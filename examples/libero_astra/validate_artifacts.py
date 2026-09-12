"""Check evidence alignment, raw usage totals, frames, and controller bounds."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def validate(folder):
    result = json.loads((folder / "result.json").read_text())
    assert result["status"] == "completed", result
    decisions = rows(folder / "decisions.jsonl")
    actions = rows(folder / "actions.jsonl")
    obs = rows(folder / "observations.jsonl")
    wire = rows(folder / "wire/api/episode/calls.jsonl")
    states = np.load(folder / "sim_states.npz")["states"]
    if result.get("continued_from"):
        np.testing.assert_array_equal(
            states[0], np.load(Path(result["continued_from"]) / "sim_states.npz")["states"][-1]
        )
    assert len(decisions) == result["decisions"]
    assert len(actions) == result["control_steps"]
    assert len(obs) == len(states) == len(actions) + 1
    assert (
        hashlib.sha256((folder / "runner_source.py").read_bytes()).hexdigest()
        == result["runner_sha256"]
    )
    assert [x["frame"] for x in obs] == list(range(len(obs)))
    assert [x["step"] for x in actions] == list(range(1, len(actions) + 1))
    assert all(
        set(x["robot"]).issubset(
            {
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
                "robot0_gripper_qvel",
                "robot0_joint_pos",
                "robot0_joint_vel",
            }
        )
        for x in obs
    )
    for a in actions:
        command = np.asarray(a["action"])
        assert command.shape == (7,) and np.isfinite(command).all()
        assert np.max(np.abs(command[:6])) <= 0.5 and abs(command[6]) == 1
        assert np.allclose(a["after"]["robot0_eef_pos"], obs[a["frame"]]["robot"]["robot0_eef_pos"])
    for d in decisions:
        aa = [a for a in actions if a["decision"] == d["decision"]]
        assert len(aa) == d["step_end"] - d["step_start"]
        assert len(aa) == d["frame_after"] - d["frame_before"]
        assert d["api_latency_s"] > 0
        assert any(w["call"] == d["decision"] and w["status"] == 200 for w in wire)
        assert (folder / "frames" / f"decision_{d['decision']:03d}_grid.png").exists()
    for i in range(len(obs)):
        for camera in ["agentview", "robot0_eye_in_hand", "birdview"]:
            assert (folder / "frames" / f"{i:06d}_{camera}.png").stat().st_size > 100
    total = dict.fromkeys(["input_tokens", "output_tokens", "total_tokens"], 0)
    for w in wire:
        assert w["request"]["model"] == "gpt-6-astra"
        response = w.get("response") or {}
        if w["status"] == 200:
            assert response["model"].startswith("gpt-6-astra")
        for key in total:
            total[key] += (response.get("usage") or {}).get(key, 0)
        for blob in set(re.findall(r"\$blob:([0-9a-f]{64})", json.dumps(w["request"]))):
            payload = (folder / "wire/api/blobs" / f"{blob}.png").read_bytes()
            assert hashlib.sha256(payload).hexdigest() == blob
    assert total == {key: result["usage"][key] for key in total}
    assert total["input_tokens"] + total["output_tokens"] == total["total_tokens"]
    if result["success"]:
        assert decisions[-1]["result"]["success"] is True
        assert actions[-1]["done"] is True
    return {
        "folder": folder.name,
        "success": result["success"],
        "decisions": len(decisions),
        "motor_steps": len(actions),
        "frames": len(obs),
        "tokens": total["total_tokens"],
        "checked": (
            "Evidence counts, state/action alignment, bounds, model identity, "
            "usage totals, image hashes, source hash"
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2] / "artifacts/astra_libero"
    )
    args = parser.parse_args()
    results = []
    for p in sorted(args.root.rglob("result.json")):
        if p.parent.name.startswith("task_"):
            result = validate(p.parent)
            result["folder"] = p.parent.relative_to(args.root).as_posix()
            results.append(result)
    assert results, "No completed attempts"
    (args.root / "validation.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
