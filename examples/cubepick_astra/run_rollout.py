"""Run one real Astra API rollout, retaining observations, predictions, and usage."""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from inspect_robots_agent import LLMAgentPolicy

from inspect_robots import eval
from inspect_robots.approver import ChainApprover, ClampApprover, DeltaLimitApprover
from inspect_robots.controller import DefaultController
from inspect_robots.logging.json_log import JsonLogSink
from inspect_robots.logging.sink import NullSink
from inspect_robots.mock import CubePickEmbodiment
from inspect_robots.scene import Scene
from inspect_robots.scorer import episode_length, min_distance_to_goal, success_at_end
from inspect_robots.task import Task

ROOT = Path(__file__).resolve().parents[2] / "artifacts/astra_cubepick"


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def append(path, value):
    with path.open("a") as handle:
        handle.write(json.dumps(value, allow_nan=False) + "\n")


def state_dict(observation):
    return {key: np.asarray(value).tolist() for key, value in observation.state.items()}


class RecordedAgent(LLMAgentPolicy):
    """Add instrumentation while preserving the plugin's prompts and actions."""

    def act(self, observation):
        before_messages = len(self.transcript() or [])
        first_call = self._calls_used
        start = time.time()
        timer = time.perf_counter()
        chunk = super().act(observation)
        elapsed = time.perf_counter() - timer
        row = {
            "env_step": observation.extra["env_step"],
            "started_at_unix": start,
            "duration_s": elapsed,
            "first_api_call": first_call,
            "api_calls": self._calls_used - first_call,
            "observation": state_dict(observation),
            "predicted_actions": [np.asarray(action.data).tolist() for action in chunk.actions],
            "action_metadata": [dict(action.meta) for action in chunk.actions],
            "chunk_length": len(chunk),
            "control_hz": chunk.control_hz,
            "messages": (self.transcript() or [])[before_messages:],
        }
        append(ROOT / "predictions.jsonl", row)
        return replace(chunk, inference_latency_s=elapsed)

    def on_trial_end(self, record, log_dir, run_id):
        super().on_trial_end(record, log_dir, run_id)
        pointer = record.metadata.get("wire_capture")
        if not pointer:
            return
        rows = [json.loads(line) for line in (Path(log_dir) / pointer).read_text().splitlines()]
        totals = {}
        usages = []
        for row in rows:
            response = row.get("response")
            usage = response.get("usage") if isinstance(response, dict) else None
            if not isinstance(usage, dict):
                continue
            usages.append({"call": row["call"], "attempt": row["attempt"], "usage": usage})
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                value = usage.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    totals[key] = totals.get(key, 0) + value
            for group, key, destination in (
                ("input_tokens_details", "cached_tokens", "cached_input_tokens"),
                ("output_tokens_details", "reasoning_tokens", "reasoning_tokens"),
            ):
                details = usage.get(group)
                value = details.get(key) if isinstance(details, dict) else None
                if isinstance(value, int) and not isinstance(value, bool):
                    totals[destination] = totals.get(destination, 0) + value
        record.metadata.setdefault("llm_usage", {}).update(totals)
        record.metadata["api_timing"] = {
            "attempts": len(rows),
            "total_request_s": sum(row["duration_s"] for row in rows),
            "usage_reported_attempts": len(usages),
        }
        record.metadata["controller"] = {"name": "DefaultController", "replan_interval": 1}
        dump(ROOT / "usage.json", {"totals": totals, "per_attempt": usages})


class StepRecorder(NullSink):
    def log_step(self, t, observation, action, result):
        append(
            ROOT / "steps.jsonl",
            {
                "env_step": t,
                "recorded_at_unix": time.time(),
                "before": state_dict(observation),
                "executed_action": np.asarray(action.data).tolist(),
                "action_metadata": dict(action.meta),
                "after": state_dict(result.observation),
                "reward": result.reward,
                "terminated": result.terminated,
                "termination_reason": result.termination_reason,
                "info": dict(result.info),
            },
        )


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")
    if (ROOT / "run_manifest.json").exists():
        raise SystemExit("This run already exists; preserve it and use another output directory.")
    ROOT.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    manifest = {
        "model": "gpt-6-astra",
        "wire": "responses",
        "effort": "medium",
        "started_at": started,
        "seed": 0,
        "replan_interval": 1,
        "max_steps": 80,
        "max_llm_calls": 30,
        "task": "reach the cube",
        "note": (
            "Unmodified CubePick: 2D reaching; both effector and target coordinates are observed."
        ),
        "status": "running",
    }
    dump(ROOT / "run_manifest.json", manifest)
    embodiment = CubePickEmbodiment()
    policy = RecordedAgent(
        model="openai/gpt-6-astra",
        wire="responses",
        effort="medium",
        max_llm_calls=30,
        images="always",
        image_horizon=None,
        wire_capture=True,
        transcript_echo=True,
    )
    policy.config = replace(policy.config, replan_interval=1)
    space = embodiment.info.action_space
    task = Task(
        name="astra-cubepick-reach",
        scenes=[Scene(id="layout-0", instruction="reach the cube", init_seed=0)],
        scorer=[success_at_end(), episode_length(), min_distance_to_goal()],
        max_steps=80,
    )
    timer = time.perf_counter()
    json_sink = JsonLogSink(str(ROOT / "logs"))
    try:
        (log,) = eval(
            task,
            policy,
            embodiment,
            seed=0,
            log_dir=str(ROOT / "logs"),
            controller=DefaultController(replan_interval=1),
            approver=ChainApprover(ClampApprover(space), DeltaLimitApprover(space)),
            sinks=[StepRecorder(), json_sink],
            store_frames=True,
            store_actions=True,
        )
        dump(ROOT / "eval.json", log.to_dict())
        manifest.update(
            {
                "status": log.status,
                "wall_time_s": time.perf_counter() - timer,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "metrics": log.results.metrics,
                "total_steps": log.stats.total_steps,
                "mean_inference_latency_s": log.stats.mean_inference_latency_s,
                "trial_metadata": log.samples[0].trial_metadata[0],
                "log_path": str(json_sink.path.relative_to(ROOT)),
            }
        )
    finally:
        policy._client.close()
        embodiment.close()
        dump(ROOT / "run_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
