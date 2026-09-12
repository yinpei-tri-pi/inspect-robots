"""One-time recovery of a NumPy-bool logging failure from original saved evidence.

Does not instantiate a simulator or make API requests. Raw API, action, state,
and image evidence are untouched. Original incomplete logs are archived.
"""

import json
import shutil
from pathlib import Path

import numpy as np
from inspect_robots_agent._responses import _parse_response

root = (
    Path(__file__).resolve().parents[2]
    / "artifacts/astra_libero/task_08_attempt_00/post_success/task_08_attempt_00"
)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


meta = json.loads((root / "result.json").read_text())
assert (
    meta["status"] == "error" and meta["error"] == "Object of type bool_ is not JSON serializable"
)
decisions = rows(root / "decisions.jsonl")
actions = rows(root / "actions.jsonl")
observations = rows(root / "observations.jsonl")
wire = rows(root / "wire/api/episode/calls.jsonl")
assert len(decisions) == 1 and len(wire) == 2
last = actions[-1]
assert last["done"] and last["action"][-1] == -1 and last["after"]["robot0_eef_pos"][2] >= 1.08
assert np.linalg.norm(last["after"]["robot0_gripper_qpos"]) > 0.045
archive = root / "recovery_originals"
archive.mkdir()
for name in ["result.json", "decisions.jsonl", "transcript.json", "review.mp4"]:
    if (root / name).exists():
        shutil.copyfile(root / name, archive / name)
w = wire[-1]
response, _ = _parse_response(w["response"])
call = response.tool_calls[0]
args = json.loads(call.arguments)
executed = [a for a in actions if a["decision"] == 1]
before = executed[0]["step"] - 1
after = executed[-1]["step"]
result = {
    "executed_steps": len(executed),
    "target": args["position"],
    "actual_position": last["after"]["robot0_eef_pos"],
    "position_error_m": float(
        np.linalg.norm(np.array(args["position"]) - last["after"]["robot0_eef_pos"])
    ),
    "gripper_qpos": last["after"]["robot0_gripper_qpos"],
    "success": True,
    "benchmark_success": True,
}
decision = {
    "decision": 1,
    "step_start": before,
    "step_end": after,
    "frame_before": before,
    "frame_after": after,
    "started_at_unix": w["t"],
    "api_latency_s": w["duration_s"],
    "simulation_time_s": observations[after]["wall_time_unix"] - (w["t"] + w["duration_s"]),
    "state_before": observations[before]["robot"],
    "state_after": observations[after]["robot"],
    "assistant": response.raw(),
    "tool": {"name": call.name, "arguments": args, "call_id": call.id},
    "executed_actions": [a["action"] for a in executed],
    "result": result,
    "timing_reconstructed": True,
    "recovery_note": (
        "Final record reconstructed from original API/action/frame logs after NumPy bool "
        "serialization failed. API latency is measured wire duration; physics/render/log "
        "duration is approximate, derived from response-end and final-frame timestamps."
    ),
}
with (root / "decisions.jsonl").open("a") as handle:
    handle.write(json.dumps(decision) + "\n")
transcript = json.loads((root / "transcript.json").read_text())
transcript.extend(
    [
        {
            "role": "user",
            "content": (
                "[Recovered final observation: exact request is call 1 in "
                "wire/api/episode/calls.jsonl; frame 17 and decision_001_grid.png.]"
            ),
            "_log_recovery": True,
        },
        response.raw(),
        {
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(result),
            "_log_recovery": True,
        },
    ]
)
(root / "transcript.json").write_text(json.dumps(transcript, indent=2))
provenance = {
    "error": meta.pop("error"),
    "traceback": meta.pop("traceback"),
    "original_logs": "recovery_originals/",
    "method": decision["recovery_note"],
    "no_new_api_calls": True,
    "no_new_simulator_steps": True,
    "success_evidence": {
        "last_action_done": last["done"],
        "gripper_command": last["action"][-1],
        "grasp_center_z": last["after"]["robot0_eef_pos"][2],
        "gripper_qpos": last["after"]["robot0_gripper_qpos"],
    },
    "time_and_tokens": "Original total wall time and provider token usage are unchanged.",
}
(root / "recovery_provenance.json").write_text(json.dumps(provenance, indent=2))
meta.update(
    status="completed",
    success=True,
    termination_reason="success",
    decisions=2,
    logging_recovery=provenance,
)
(root / "result.json").write_text(json.dumps(meta, indent=2))
if (root / "review.mp4").exists():
    (root / "review.mp4").unlink()
print(
    json.dumps(
        {
            "success": meta["success"],
            "decisions": 2,
            "motor_steps": len(actions),
            "usage": meta["usage"],
            "total_wall_time_s": meta["total_wall_time_s"],
            "provenance": str(root / "recovery_provenance.json"),
        },
        indent=2,
    )
)
