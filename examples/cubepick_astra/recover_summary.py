"""Recover the final summary from the completed rollout; makes no API calls."""

import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from statistics import mean

import numpy as np
from inspect_robots_agent.policy import AgentPolicyConfig

from inspect_robots.log import EvalLog, EvalResults, EvalSpec, EvalStats, SceneResult
from inspect_robots.mock import CubePickEmbodiment

ROOT = Path(__file__).resolve().parents[2] / "artifacts/astra_cubepick"


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def main():
    manifest = json.loads((ROOT / "run_manifest.json").read_text())
    predictions = lines(ROOT / "predictions.jsonl")
    steps = lines(ROOT / "steps.jsonl")
    wire_path = next((ROOT / "logs/wire").rglob("calls.jsonl"))
    calls = lines(wire_path)
    run_id = wire_path.parent.parent.name
    transcript_path = ROOT / "logs/transcripts" / run_id / "layout-0-e0.jsonl"
    usage = json.loads((ROOT / "usage.json").read_text())["totals"]
    assert len(steps) == len(predictions) == len(calls) == 15
    assert all(c["status"] == 200 and c["response"]["model"] == "gpt-6-astra" for c in calls)
    assert steps[-1]["terminated"] and steps[-1]["termination_reason"] == "success"
    for i, step in enumerate(steps):
        np.testing.assert_allclose(
            np.clip(np.array(step["before"]["eef_pos"]) + step["executed_action"], 0, 1),
            step["after"]["eef_pos"],
            atol=1e-12,
        )
        distance = np.linalg.norm(np.array(step["after"]["eef_pos"]) - step["after"]["cube_pos"])
        assert abs(distance - step["info"]["distance"]) < 1e-12
        np.testing.assert_allclose(
            predictions[i]["predicted_actions"][0], step["executed_action"], atol=1e-12
        )
        if i:
            assert step["before"] == steps[i - 1]["after"]
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        assert usage[key] == sum(c["response"]["usage"][key] for c in calls)
    started = predictions[0]["started_at_unix"]
    completed = steps[-1]["recorded_at_unix"]

    def iso(t):
        return datetime.fromtimestamp(t, timezone.utc).isoformat()

    metrics = {
        "success_at_end": 1.0,
        "episode_length": float(len(steps)),
        "min_distance_to_goal": min(s["info"]["distance"] for s in steps),
    }
    recovery = {
        "summary_rebuilt_from_durable_records": True,
        "reason": (
            "Custom sink omitted JsonLogSink; final summary writer failed after successful rollout."
        ),
        "source_files": [
            "predictions.jsonl",
            "steps.jsonl",
            "usage.json",
            str(wire_path.relative_to(ROOT)),
            str(transcript_path.relative_to(ROOT)),
        ],
        "timing_definition": (
            "First policy decision start to final completed simulator step. "
            "Excludes setup and final scoring/report generation."
        ),
        "additional_model_calls": 0,
    }
    metadata = {
        "wire_capture": str(wire_path.relative_to(ROOT / "logs")),
        "transcript": str(transcript_path.relative_to(ROOT / "logs")),
        "action_log": f"actions/{run_id}/layout-0-e0.jsonl",
        "llm_usage": {"llm_calls": len(calls), **usage},
        "api_timing": {
            "attempts": len(calls),
            "total_request_s": sum(c["duration_s"] for c in calls),
            "usage_reported_attempts": len(calls),
        },
        "controller": {"name": "DefaultController", "replan_interval": 1},
        "summary_provenance": recovery,
    }
    info = CubePickEmbodiment().info
    config = AgentPolicyConfig(
        model="gpt-6-astra",
        base_url="https://api.openai.com/v1",
        wire="responses",
        effort="medium",
        max_llm_calls=30,
        image_horizon=None,
        transcript_echo=True,
        replan_interval=1,
    )
    log = EvalLog(
        version=1,
        status="success",
        eval=EvalSpec(
            task="astra-cubepick-reach",
            policy="agent",
            embodiment="cubepick",
            created=iso(started),
            inspect_robots_version=version("inspect-robots"),
            git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            policy_config=asdict(config),
            embodiment_info={
                "control_hz": info.control_hz,
                "is_simulated": True,
                "capabilities": sorted(info.capabilities),
            },
            seed=0,
            max_steps=80,
        ),
        results=EvalResults(total_scenes=1, total_trials=1, metrics=metrics),
        stats=EvalStats(
            started_at=iso(started),
            completed_at=iso(completed),
            duration_s=completed - started,
            total_steps=len(steps),
            mean_inference_latency_s=mean(p["duration_s"] for p in predictions),
            frames_dir=str(ROOT / "logs/frames" / run_id),
        ),
        samples=(
            SceneResult(
                scene_id="layout-0",
                status="success",
                reduced=metrics,
                epochs=(metrics,),
                instruction="reach the cube",
                operator_judgements=(None,),
                judgement_sources=(None,),
                operator_notes=(None,),
                operator_messages=((),),
                trial_metadata=(metadata,),
                termination_reasons=("success",),
                policy_transcripts=(lines(transcript_path),),
            ),
        ),
    )
    write(ROOT / "logs/eval.json", log.to_dict())
    write(ROOT / "eval.json", log.to_dict())
    manifest.update(
        {
            "status": "success",
            "rollout_started_at": iso(started),
            "completed_at": iso(completed),
            "wall_time_s": completed - started,
            "metrics": metrics,
            "total_steps": len(steps),
            "mean_inference_latency_s": log.stats.mean_inference_latency_s,
            "trial_metadata": metadata,
            "log_path": "logs/eval.json",
            "summary_provenance": recovery,
        }
    )
    write(ROOT / "run_manifest.json", manifest)
    write(
        ROOT / "data_validation.json",
        {
            "passed": True,
            "recorded_transitions_verified": len(steps),
            "actual_api_calls": len(calls),
            "usage_matches_raw_api": True,
            "model": "gpt-6-astra",
            "additional_api_calls": 0,
        },
    )
    print(
        json.dumps(
            {"wall_time_s": completed - started, "metrics": metrics, "usage": usage}, indent=2
        )
    )


if __name__ == "__main__":
    main()
