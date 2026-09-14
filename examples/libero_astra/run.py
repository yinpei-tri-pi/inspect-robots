# Keep model prompts and tool descriptions unchanged.
# ruff: noqa: E501
"""A vision-language model controls LIBERO's Panda through Cartesian pose primitives.

The model sees RGB images, robot proprioception, and camera calibration aids.
No object poses, goal-region coordinates, demonstrations, or trained robot policy
are supplied. Every motion is executed by robosuite's OSC_POSE motor controller.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from inspect_robots_agent._capture import WireCapture
from inspect_robots_agent._llm import Provider
from inspect_robots_agent._responses import ResponsesClient
from inspect_robots_agent.policy import _evicted_view
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
CAMERAS = ["agentview", "robot0_eye_in_hand", "birdview"]
WIDTH = 384


def write_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def append_json(path, data):
    with path.open("a") as handle:
        handle.write(json.dumps(data, allow_nan=False) + "\n")


def image_url(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def state(obs):
    keys = [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "robot0_gripper_qvel",
        "robot0_joint_pos",
        "robot0_joint_vel",
    ]
    return {key: np.asarray(obs[key]).tolist() for key in keys if key in obs}


def camera_image(obs, name):
    # MuJoCo's framebuffer origin is bottom-left. Use upright images for the VLM.
    return Image.fromarray(np.ascontiguousarray(obs[name + "_image"][::-1]))


def grid_image(env, obs):
    """Calibrated table-plane grid; does not access object identities or poses."""
    image = camera_image(obs, "birdview").convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    camera = env.sim.model.camera_name2id("birdview")
    position = env.sim.data.cam_xpos[camera]
    rotation = env.sim.data.cam_xmat[camera].reshape(3, 3)
    focal = WIDTH / (2 * math.tan(math.radians(env.sim.model.cam_fovy[camera]) / 2))

    def project(x, y):
        point = rotation.T @ (np.array([x, y, 0.90]) - position)
        return (
            WIDTH / 2 + focal * point[0] / (-point[2]),
            WIDTH / 2 - focal * point[1] / (-point[2]),
        )

    for val in np.arange(-0.4, 0.401, 0.1):
        a, b = project(val, -0.4), project(val, 0.4)
        draw.line([a, b], fill=(110, 180, 215), width=1)
        draw.text(
            (b[0] + 2, b[1] + 1),
            f"x{val:+.1f}",
            fill="white",
            font=font,
            stroke_width=1,
            stroke_fill="black",
        )
        a, b = project(-0.4, val), project(0.4, val)
        draw.line([a, b], fill=(110, 180, 215), width=1)
        draw.text(
            (a[0] + 2, a[1] + 1),
            f"y{val:+.1f}",
            fill="white",
            font=font,
            stroke_width=1,
            stroke_fill="black",
        )
    return image


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "move_to",
            "description": "Move the Panda's gripper grasp-center to a world-frame XYZ target in meters. Keep or change the gripper opening. A feedback controller executes motor actions until arrival or max_steps; then fresh images show the result. Unreachable/contact-limited targets may not be reached.",
            "parameters": {
                "type": "object",
                "properties": {
                    "position": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3,
                        "description": "World [x,y,z] in meters; table top z=0.90.",
                    },
                    "yaw_deg": {
                        "type": "number",
                        "description": "World-Z rotation relative to initial gripper orientation, in degrees; default keeps previous target yaw.",
                    },
                    "gripper": {"type": "string", "enum": ["keep", "open", "close"]},
                    "max_steps": {
                        "type": "integer",
                        "minimum": 8,
                        "maximum": 60,
                        "description": "Maximum control steps at 20 Hz, default 40.",
                    },
                    "note": {
                        "type": "string",
                        "description": "Describe the visible situation and why this motion helps.",
                    },
                },
                "required": ["position", "gripper", "note"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Ask to end the task; success is checked by LIBERO, not your self-report.",
            "parameters": {
                "type": "object",
                "properties": {"note": {"type": "string"}},
                "required": ["note"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "give_up",
            "description": "End an unsuccessful attempt, explaining the obstacle.",
            "parameters": {
                "type": "object",
                "properties": {"note": {"type": "string"}},
                "required": ["note"],
            },
        },
    },
]

SYSTEM = """You are directly controlling a simulated Franka Panda robot in LIBERO to complete a manipulation task. You are the only task planner. Choose one tool call at a time, observe its result, and adapt.

You see an upright external RGB camera, a wrist RGB camera, and an overhead RGB camera with a calibrated x/y grid on the table plane. Grid labels are world meters, table height z=0.90 m. The grid is a camera calibration aid, not an object detector: you must identify objects visually. Exact object positions and task predicates are not given to you. Use the overhead grid to estimate x,y and the side/wrist views to judge z and contact. Objects above the table can project away from the grid plane.

move_to.position is the gripper grasp-center [x,y,z] in world meters. Stay in x=[-0.50,0.40], y=[-0.50,0.50], z=[0.895,1.40]. Gripper open/close actuates real simulated fingers. yaw_deg rotates the initial gripper orientation about world Z; 0 means the initial orientation. A generic feedback controller turns your target into bounded Cartesian motor commands; no object is teleported or automatically grasped. A motion may stop early when the pose and gripper settle. Inspect the actual final position and new images after each call.

For pick-and-place, approach above the object with an open gripper, descend until the fingers straddle it, close, lift and verify it actually follows, then carry and release over the destination. Keep the gripper closed during transfer. Avoid moving sideways through tall objects; lift first. Use small corrections near contact. For pushing, use a closed gripper and approach the object's side at the correct height. You may use up to 60 physical steps per call (3 s). Do not repeatedly issue an unreachable target without changing your approach.

Every move needs a short note explaining the visual evidence and next action. A simulator success check ends the episode automatically. If done is rejected, continue or give_up. Your budget is 40 calls and 800 control steps. Be efficient but verify grasping and placement from images.
"""


def run_task(args, task_id):
    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(task_id)
    out = Path(args.output) / f"task_{task_id:02d}_attempt_{args.attempt:02d}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists() or (out / "decisions.jsonl").exists():
        raise RuntimeError(f"Refusing to overwrite {out}")
    (out / "frames").mkdir(exist_ok=True)
    capture = WireCapture()
    capture.begin_trial(str(out), "api", "episode")
    client = ResponsesClient(
        Provider(
            base_url="https://api.openai.com/v1",
            api_key=os.environ["OPENAI_API_KEY"],
            model=args.model,
        ),
        capture=capture,
    )
    continuation = getattr(args, "continue_from", None)
    system_prompt = SYSTEM
    if continuation:
        system_prompt += "\nPOST-SUCCESS FINISHING PHASE: The saved benchmark episode already satisfied LIBERO's predicate while the gripper still held the object. You are continuing from that exact saved simulator state. Finish the physical placement: open the fingers gently, then withdraw the empty gripper above z=1.08 m while keeping the placed object in position. Only success with an open, withdrawn gripper completes this phase. You have 8 calls and 160 motor steps. No object poses are provided.\n"
    meta = {
        "task_id": task_id,
        "suite": args.suite,
        "instruction": task.language,
        "attempt": args.attempt,
        "init_state_index": args.init_state,
        "seed": args.seed,
        "model": args.model,
        "reasoning_effort": "medium",
        "observability": "RGB + robot proprioception + calibrated table-plane grid; no object poses or goal-region coordinates",
        "controller": "robosuite OSC_POSE with generic Cartesian target tracking",
        "control_hz": 20,
        "rendering": "CPU OSMesa",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }
    write_json(out / "metadata.json", meta)
    write_json(out / "tools.json", TOOLS)
    source = Path(__file__).read_bytes()
    (out / "runner_source.py").write_bytes(source)
    meta["runner_sha256"] = hashlib.sha256(source).hexdigest()
    (out / "system_prompt.txt").write_text(system_prompt)
    if continuation:
        meta["phase"] = "post_success_release_and_withdraw"
        meta["continued_from"] = str(Path(continuation).resolve())
        meta["termination_criterion"] = (
            "LIBERO success AND open gripper AND grasp-center z >= 1.08 m"
        )
    env = None
    start = time.perf_counter()
    step_index = 0
    frame_index = 0
    simulation_states = []
    initial_rotation = None
    gripper_command = -1.0
    yaw = 0.0
    success = False
    reason = "budget"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Task: " + task.language},
    ]
    try:
        env = OffScreenRenderEnv(
            bddl_file_name=str(
                Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            ),
            camera_heights=WIDTH,
            camera_widths=WIDTH,
            camera_names=CAMERAS,
        )
        env.seed(args.seed)
        env.reset()
        camera_id = env.sim.model.camera_name2id("birdview")
        env.sim.model.cam_pos[camera_id] = [0.0, 0.0, 2.1]
        env.sim.model.cam_quat[camera_id] = [1.0, 0.0, 0.0, 0.0]
        initial_state = (
            np.load(Path(continuation) / "sim_states.npz")["states"][-1]
            if continuation
            else suite.get_task_init_states(task_id)[args.init_state]
        )
        obs = env.set_init_state(initial_state)
        if continuation:
            np.testing.assert_array_equal(env.get_sim_state(), initial_state)
            meta["restored_state_exact_match"] = True
        warmup_states = [env.get_sim_state().copy()]
        (out / "warmup").mkdir(exist_ok=True)
        warmup_count = 0 if continuation else 10
        for warmup_step in range(warmup_count):
            before_warmup = state(obs)
            obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
            warmup_states.append(env.get_sim_state().copy())
            for camera in CAMERAS:
                camera_image(obs, camera).save(out / "warmup" / f"{warmup_step:03d}_{camera}.png")
            append_json(
                out / "warmup.jsonl",
                {
                    "step": warmup_step,
                    "action": [0.0] * 6 + [-1.0],
                    "before": before_warmup,
                    "after": state(obs),
                },
            )
        np.savez_compressed(out / "warmup_states.npz", states=np.asarray(warmup_states))
        (out / "warmup.jsonl").touch(exist_ok=True)
        if continuation:
            previous_meta = json.loads((Path(continuation) / "result.json").read_text())
            initial_rotation = np.asarray(previous_meta["initial_gripper_rotation"])
            gripper_command = 1.0
        else:
            initial_rotation = Rotation.from_quat(obs["robot0_eef_quat"]).as_matrix()
        meta["initial_robot_state"] = state(obs)
        meta["initial_gripper_rotation"] = initial_rotation.tolist()
        meta["controller_output_max"] = env.robots[0].controller.output_max.tolist()
        assert np.allclose(env.robots[0].controller.output_max, [0.05, 0.05, 0.05, 0.5, 0.5, 0.5])
        meta["warmup_steps"] = warmup_count
        if args.reference_root and not continuation:
            reference = Path(args.reference_root) / f"task_{task_id:02d}_attempt_{args.attempt:02d}"
            expected = np.load(reference / "sim_states.npz")["states"][0]
            np.testing.assert_allclose(env.get_sim_state(), expected, rtol=0, atol=1e-10)
            meta["comparison_reference"] = str(reference.resolve())
            meta["initial_state_matches_reference"] = True
            meta["initial_state_max_absolute_difference"] = float(
                np.max(np.abs(env.get_sim_state() - expected))
            )
        meta["camera_calibration"] = {
            name: {
                "position": env.sim.data.cam_xpos[env.sim.model.camera_name2id(name)].tolist(),
                "rotation": env.sim.data.cam_xmat[env.sim.model.camera_name2id(name)].tolist(),
                "fovy": float(env.sim.model.cam_fovy[env.sim.model.camera_name2id(name)]),
            }
            for name in CAMERAS
        }
        (out / "model.xml").write_text(env.sim.model.get_xml())
        meta["setup_time_s"] = time.perf_counter() - start
        write_json(out / "metadata.json", meta)
        rollout_start = time.perf_counter()

        def completion(observation):
            task_success = bool(env.check_success())
            if not continuation:
                return task_success
            return bool(
                task_success
                and gripper_command < 0
                and float(observation["robot0_eef_pos"][2]) >= 1.08
                and np.linalg.norm(observation["robot0_gripper_qpos"]) > 0.045
            )

        def save_frame(observation):
            nonlocal frame_index
            for camera in CAMERAS:
                camera_image(observation, camera).save(
                    out / "frames" / f"{frame_index:06d}_{camera}.png"
                )
            simulation_states.append(env.get_sim_state().copy())
            append_json(
                out / "observations.jsonl",
                {
                    "frame": frame_index,
                    "step": step_index,
                    "wall_time_unix": time.time(),
                    "robot": state(observation),
                },
            )
            frame_index += 1

        save_frame(obs)
        for decision_index in range(args.max_calls):
            if step_index >= args.max_steps:
                break
            parts = [
                {
                    "type": "text",
                    "text": f"Decision {decision_index}, physical step {step_index}. Robot state:\n"
                    + json.dumps(state(obs))
                    + f"\nPrevious target yaw={yaw} deg, gripper command={'closed' if gripper_command > 0 else 'open'}.",
                }
            ]
            for name, image in [
                ("External camera", camera_image(obs, "agentview")),
                ("Wrist camera", camera_image(obs, "robot0_eye_in_hand")),
                ("Overhead camera with world x/y grid", grid_image(env, obs)),
            ]:
                parts.extend(
                    [
                        {"type": "text", "text": name},
                        {"type": "image_url", "image_url": {"url": image_url(image)}},
                    ]
                )
            grid_image(env, obs).save(out / "frames" / f"decision_{decision_index:03d}_grid.png")
            messages.append({"role": "user", "content": parts})
            before = state(obs)
            before_frame = frame_index - 1
            request_start = time.time()
            timer = time.perf_counter()
            response = client.complete(_evicted_view(messages, 2), TOOLS, reasoning_effort="medium")
            api_s = time.perf_counter() - timer
            messages.append(response.raw())
            decision = {
                "decision": decision_index,
                "step_start": step_index,
                "frame_before": before_frame,
                "started_at_unix": request_start,
                "api_latency_s": api_s,
                "state_before": before,
                "assistant": response.raw(),
            }
            call = response.tool_calls[0] if response.tool_calls else None
            result = {}
            if call is None:
                messages.append({"role": "user", "content": "Please issue exactly one tool call."})
                result = {"error": "No tool call returned"}
            else:
                try:
                    arguments = json.loads(call.arguments)
                    decision["tool"] = {
                        "name": call.name,
                        "arguments": arguments,
                        "call_id": call.id,
                    }
                    print(
                        f"[{task_id}:{decision_index}] {call.name} {json.dumps(arguments)}",
                        flush=True,
                    )
                    if call.name == "give_up":
                        reason = "model_give_up"
                        result = {"success": False, "note": arguments.get("note", "")}
                    elif call.name == "done":
                        success = completion(obs)
                        result = {
                            "success": success,
                            "note": "LIBERO success check"
                            if success
                            else "LIBERO has not detected success. Continue acting or give_up.",
                        }
                    elif call.name == "move_to":
                        target = np.asarray(arguments["position"], dtype=float)
                        if (
                            target.shape != (3,)
                            or not np.isfinite(target).all()
                            or np.any(target < [-0.50, -0.50, 0.895])
                            or np.any(target > [0.40, 0.50, 1.40])
                        ):
                            raise ValueError("Target outside declared workspace bounds")
                        new_yaw = float(arguments.get("yaw_deg", yaw))
                        if not np.isfinite(new_yaw) or abs(new_yaw) > 360:
                            raise ValueError("yaw_deg must be finite and between -360 and 360")
                        grip = arguments.get("gripper", "keep")
                        if grip not in {"keep", "open", "close"}:
                            raise ValueError("Unknown gripper command")
                        gripper_command = {"keep": gripper_command, "open": -1.0, "close": 1.0}[
                            grip
                        ]
                        yaw = new_yaw
                        target_rotation = (
                            Rotation.from_euler("z", yaw, degrees=True).as_matrix()
                            @ initial_rotation
                        )
                        maximum = min(60, max(8, int(arguments.get("max_steps", 40))))
                        stable = 0
                        actions = []
                        motion_start = time.perf_counter()
                        for j in range(min(maximum, args.max_steps - step_index)):
                            current = np.asarray(obs["robot0_eef_pos"])
                            rotation = Rotation.from_quat(obs["robot0_eef_quat"]).as_matrix()
                            rotation_error = Rotation.from_matrix(
                                target_rotation @ rotation.T
                            ).as_rotvec()
                            # OSC_POSE scales normalized position by 0.05 m and rotation by 0.5 rad.
                            action = np.r_[
                                np.clip((target - current) / 0.05, -0.5, 0.5),
                                np.clip(rotation_error / 0.5, -0.5, 0.5),
                                gripper_command,
                            ]
                            previous = state(obs)
                            obs, reward, done, _info = env.step(action.tolist())
                            step_index += 1
                            save_frame(obs)
                            append_json(
                                out / "actions.jsonl",
                                {
                                    "step": step_index,
                                    "decision": decision_index,
                                    "action": action.tolist(),
                                    "before": previous,
                                    "after": state(obs),
                                    "reward": float(reward),
                                    "done": bool(done),
                                    "frame": frame_index - 1,
                                },
                            )
                            actions.append(action.tolist())
                            success = completion(obs)
                            if success:
                                break
                            pos_error = np.linalg.norm(target - np.asarray(obs["robot0_eef_pos"]))
                            grip_velocity = np.linalg.norm(obs.get("robot0_gripper_qvel", [0, 0]))
                            stable = (
                                stable + 1
                                if pos_error < 0.003
                                and np.linalg.norm(rotation_error) < 0.04
                                and grip_velocity < 0.01
                                else 0
                            )
                            if j >= 9 and stable >= 4:
                                break
                        result = {
                            "executed_steps": len(actions),
                            "target": target.tolist(),
                            "actual_position": np.asarray(obs["robot0_eef_pos"]).tolist(),
                            "position_error_m": float(
                                np.linalg.norm(target - obs["robot0_eef_pos"])
                            ),
                            "gripper_qpos": np.asarray(obs["robot0_gripper_qpos"]).tolist(),
                            "success": success,
                            "benchmark_success": bool(env.check_success()),
                        }
                        decision["executed_actions"] = actions
                        decision["simulation_time_s"] = time.perf_counter() - motion_start
                    else:
                        raise ValueError("Unknown tool")
                except (ValueError, KeyError, TypeError) as error:
                    result = {"error": str(error)}
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)}
                )
                for extra in response.tool_calls[1:]:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": extra.id,
                            "content": "Not executed: only one action tool per decision.",
                        }
                    )
            decision.update(
                {
                    "result": result,
                    "step_end": step_index,
                    "frame_after": frame_index - 1,
                    "state_after": state(obs),
                }
            )
            append_json(out / "decisions.jsonl", decision)
            # Keep media in image files and in the native wire capture; readable transcript omits payloads.
            clean = []
            for msg in messages:
                item = dict(msg)
                if isinstance(item.get("content"), list):
                    item["content"] = [
                        p
                        if p.get("type") != "image_url"
                        else {
                            "type": "text",
                            "text": "[camera image saved in frames and wire capture]",
                        }
                        for p in item["content"]
                    ]
                clean.append(item)
            write_json(out / "transcript.json", clean)
            print(f"[{task_id}:{decision_index}] result {json.dumps(result)}", flush=True)
            if success or reason == "model_give_up":
                reason = "success" if success else reason
                break
        meta.update(
            {
                "success": success,
                "termination_reason": reason,
                "control_steps": step_index,
                "decisions": len((out / "decisions.jsonl").read_text().splitlines()),
                "rollout_wall_time_s": time.perf_counter() - rollout_start,
                "status": "completed",
            }
        )
    except Exception as error:
        meta.update(
            {
                "success": False,
                "status": "error",
                "error": str(error),
                "traceback": traceback.format_exc(),
                "control_steps": step_index,
            }
        )
        print(traceback.format_exc(), flush=True)
    finally:
        capture.end_trial()
        client.close()
        if simulation_states:
            np.savez_compressed(out / "sim_states.npz", states=np.asarray(simulation_states))
        if env is not None:
            try:
                env.close()
            except Exception as close_error:
                meta["close_error"] = str(close_error)
        meta["total_wall_time_s"] = time.perf_counter() - start
        meta["completed_at"] = datetime.now(timezone.utc).isoformat()
        wire = out / "wire/api/episode/calls.jsonl"
        if wire.exists():
            rows = [json.loads(line) for line in wire.read_text().splitlines()]
            totals = {
                key: sum((r.get("response") or {}).get("usage", {}).get(key, 0) for r in rows)
                for key in ["input_tokens", "output_tokens", "total_tokens"]
            }
            totals["reasoning_tokens"] = sum(
                (r.get("response") or {})
                .get("usage", {})
                .get("output_tokens_details", {})
                .get("reasoning_tokens", 0)
                for r in rows
            )
            totals["cached_input_tokens"] = sum(
                (r.get("response") or {})
                .get("usage", {})
                .get("input_tokens_details", {})
                .get("cached_tokens", 0)
                for r in rows
            )
            meta["usage"] = totals
            meta["api_request_time_s"] = sum(r["duration_s"] for r in rows)
            meta["api_attempts"] = len(rows)
        write_json(out / "result.json", meta)
        print(json.dumps(meta, indent=2), flush=True)
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, nargs="+", default=[5, 6, 8])
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument(
        "--reference-root",
        help="Verify each initial simulator state against a saved comparison run before requesting the model.",
    )
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--init-state", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-calls", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--output", default=str(ROOT / "artifacts/astra_libero"))
    parser.add_argument(
        "--continue-from",
        default=None,
        help="Continue a successful saved episode to release the object and withdraw; logs go to the specified new output root.",
    )
    args = parser.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY must be set")
    for task_id in args.tasks:
        run_task(args, task_id)
