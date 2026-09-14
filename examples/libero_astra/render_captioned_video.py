"""Render recorded LIBERO frames with call captions and a final result card."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from build_report import read_lines
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

WIDTH = 1152
PANEL = 384
PAD = 18
TOP_PAD = 12
LINE = 27
BACKGROUND = (13, 19, 31)
BLUE = (129, 187, 255)
MUTED = (170, 187, 208)
PRICING = {
    "model": "gpt-6-astra",
    "currency": "USD",
    "service_tier": "default",
    "verified_on": "2026-09-11",
    "source": "https://developers.openai.com/api/docs/models/gpt-6-astra",
    "cache_accounting_source": "https://developers.openai.com/api/docs/guides/prompt-caching",
    "usd_per_million_tokens": {
        "ordinary_input": 10.0,
        "cached_input": 1.0,
        "cache_write": 12.5,
        "output": 50.0,
    },
    "long_context_threshold": 272_000,
    "long_context_input_multiplier": 2.0,
    "long_context_output_multiplier": 1.5,
    "note": (
        "Standard list-price estimate; includes cache writes; excludes account discounts and taxes."
    ),
}
CAMERAS = [
    ("agentview", "External · Panda arm"),
    ("robot0_eye_in_hand", "Wrist camera"),
    ("birdview", "Overhead camera"),
]


def pricing_for(model):
    if model == "gpt-6-astra":
        return PRICING
    if model == "gpt-5.6-luna":
        return {
            **PRICING,
            "model": model,
            "source": "https://developers.openai.com/api/docs/models/gpt-5.6-luna",
            "usd_per_million_tokens": {
                "ordinary_input": 0.20,
                "cached_input": 0.02,
                "cache_write": 0.25,
                "output": 1.20,
            },
        }
    raise ValueError(f"No verified pricing configured for {model}")


def font(size):
    candidates = list(
        Path(sys.prefix).glob(
            "lib/python*/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf"
        )
    )
    return (
        ImageFont.truetype(str(candidates[0]), size)
        if candidates
        else ImageFont.load_default(size=size)
    )


def wrap(text, face, width=WIDTH - 2 * PAD):
    """Wrap using actual glyph widths, including unusually long words."""
    lines = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}" if current else word
        if face.getlength(candidate) <= width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = ""
        for char in word:
            if face.getlength(current + char) > width:
                lines.append(current)
                current = ""
            current += char
    if current:
        lines.append(current)
    return lines


def caption(decision, finishing=False):
    tool = decision["tool"]
    args = tool["arguments"]
    heading = f"Call {decision['decision'] + 1} · {tool['name']}"
    if finishing:
        heading = "Finishing phase · " + heading
    parameters = ", ".join(
        f"{key}={json.dumps(value, ensure_ascii=False, separators=(',', ':'))}"
        for key, value in args.items()
        if key not in ("note", "max_steps")
    )
    heading += f"({parameters})"
    note = "note: " + args["note"] if "note" in args else ""
    return heading, note


def yaw_caption(decision, initial_rotation, previous_yaw, target_yaw):
    measured = []
    for state in (decision["state_before"], decision["state_after"]):
        rotation = Rotation.from_quat(state["robot0_eef_quat"]).as_matrix()
        relative = rotation @ initial_rotation.T
        measured.append(float(np.degrees(np.arctan2(relative[1, 0], relative[0, 0]))))
    if target_yaw == previous_yaw:
        target_text = f"Yaw target held at {target_yaw:g}°"
    else:
        target_text = f"Yaw target {previous_yaw:g}° → {target_yaw:g}°"
    return (
        f"{target_text} · Recorded yaw {measured[0]:.1f}° → {measured[1]:.1f}° "
        "(world Z, relative to initial pose)."
    )


def video_filename(result):
    task_goal = "_".join(result["instruction"].lower().split())
    label = "SUCCESS" if result["success"] else "FAILURE"
    prefix = "" if result["model"] == "gpt-6-astra" else result["model"] + "__"
    return f"{prefix}{task_goal}_{label}.mp4"


def request_metrics(record):
    """Estimate one saved API response, counting reasoning within output once."""
    response = record["response"]
    pricing = pricing_for(response["model"])
    if response["service_tier"] != "default":
        raise ValueError("Pricing is only configured for standard requests")
    usage = response["usage"]
    details = usage["input_tokens_details"]
    cached = details["cached_tokens"]
    written = details["cache_write_tokens"]
    ordinary = usage["input_tokens"] - cached - written
    if min(ordinary, cached, written) < 0:
        raise ValueError("Invalid input token accounting")
    rates = pricing["usd_per_million_tokens"]
    input_cost = (
        ordinary * rates["ordinary_input"]
        + cached * rates["cached_input"]
        + written * rates["cache_write"]
    )
    output_cost = usage["output_tokens"] * rates["output"]
    if usage["input_tokens"] > pricing["long_context_threshold"]:
        input_cost *= pricing["long_context_input_multiplier"]
        output_cost *= pricing["long_context_output_multiplier"]
    return {
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "reasoning_tokens": usage["output_tokens_details"]["reasoning_tokens"],
        "total_tokens": usage["total_tokens"],
        "cached_input_tokens": cached,
        "cache_write_tokens": written,
        "estimated_cost_usd": (input_cost + output_cost) / 1_000_000,
    }


def render(folder, overwrite=False):
    folders = [
        folder,
        *sorted(p.parent for p in (folder / "post_success").glob("task_*/result.json")),
    ]
    results = [json.loads((p / "result.json").read_text()) for p in folders]
    model = results[0]["model"]
    pricing = pricing_for(model)
    if any(r["model"] != model for r in results):
        raise ValueError("All phases must use the same model")
    destination = folder / video_filename(results[0])
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Use --overwrite to replace {destination}")
    fps = results[0]["control_hz"]
    if any(r["control_hz"] != fps for r in results):
        raise ValueError("All phases must use the same control frequency")
    face, small = font(22), font(17)
    goal_face = font(24)
    goal_caption = f"Task goal: {results[0]['instruction']} · {model}"
    goal_lines = wrap(goal_caption, goal_face)
    goal_height = len(goal_lines) * 31 + 8
    entries = []
    cumulative = {}
    for segment, source in enumerate(folders):
        decisions = read_lines(source / "decisions.jsonl")
        if not decisions:
            raise ValueError(f"No decisions in {source}")
        wire = read_lines(source / "wire/api/episode/calls.jsonl")
        cursor = 0
        # Each recorded runner phase initializes the target yaw to zero.
        previous_yaw = 0.0
        initial_rotation = np.asarray(results[segment]["initial_gripper_rotation"])
        for decision in decisions:
            before, after = decision["frame_before"], decision["frame_after"]
            if before != cursor or after < before:
                raise ValueError(f"Noncontiguous frames in {source}")
            args = decision["tool"]["arguments"]
            heading, note = caption(decision, finishing=segment > 0)
            orientation = None
            if decision["tool"]["name"] == "move_to":
                target_yaw = float(args.get("yaw_deg", previous_yaw))
                orientation = yaw_caption(decision, initial_rotation, previous_yaw, target_yaw)
                previous_yaw = target_yaw
            lines = [(line, BLUE) for line in wrap(heading, face)]
            if orientation:
                lines += [(line, (132, 227, 176)) for line in wrap(orientation, face)]
            lines += [(line, "white") for line in wrap(note, face)]
            requests = [w for w in wire if w["call"] == decision["decision"]]
            if not requests:
                raise ValueError(f"No saved API usage for call {decision['decision']}")
            metrics = [request_metrics(w) for w in requests]
            call_usage = {key: sum(m[key] for m in metrics) for key in metrics[0]}
            call_usage["api_latency_s"] = decision["api_latency_s"]
            for key, value in call_usage.items():
                cumulative[key] = cumulative.get(key, 0) + value
            details = [f"max_steps={args['max_steps']}"] if "max_steps" in args else []
            details += [
                f"API latency {decision['api_latency_s']:.2f} s",
                f"Input {call_usage['input_tokens']:,}",
                f"Generated {call_usage['output_tokens']:,} "
                f"({call_usage['reasoning_tokens']:,} reasoning)",
            ]
            detail_lines = wrap(" · ".join(details), small)
            detail_lines += wrap(
                f"Accumulated: {cumulative['input_tokens']:,} input / "
                f"{cumulative['output_tokens']:,} generated tokens · "
                f"Estimated API cost ${cumulative['estimated_cost_usd']:.4f}",
                small,
            )
            # Keep all physical frames once; show calls without motion as explicit holds.
            frames = list(
                range(
                    before if not entries or entries[-1]["segment"] != segment else before + 1,
                    after + 1,
                )
            )
            if before == after:
                frames = [before] * fps
            entries.append(
                {
                    "source": source,
                    "segment": segment,
                    "decision": decision,
                    "heading": heading,
                    "note": note,
                    "orientation_caption": orientation,
                    "lines": lines,
                    "detail_lines": detail_lines,
                    "usage": call_usage,
                    "cumulative_usage": dict(cumulative),
                    "frames": frames,
                }
            )
            cursor = after
        source_count = len(list((source / "frames").glob("[0-9]*_agentview.png")))
        if source_count != cursor + 1:
            raise ValueError(f"Frame count disagrees with decisions in {source}")
    for key in ("input_tokens", "output_tokens", "total_tokens", "reasoning_tokens"):
        if cumulative[key] != sum(r["usage"][key] for r in results):
            raise ValueError(f"Saved results and API records disagree on {key}")
    details_height = 24 * max(len(e["detail_lines"]) for e in entries)
    banner = max(164, 2 * TOP_PAD + goal_height + LINE * max(len(e["lines"]) for e in entries))
    banner += banner % 2
    images_bottom = banner + 30 + PANEL
    metrics_y = images_bottom + 10
    height = metrics_y + details_height + 26

    def canvas_for(entry, frame_index):
        canvas = Image.new("RGB", (WIDTH, height), BACKGROUND)
        draw = ImageDraw.Draw(canvas)
        for n, line in enumerate(goal_lines):
            draw.text(
                (PAD, TOP_PAD + n * 31), line, font=goal_face, fill=(255, 211, 137), anchor="lt"
            )
        draw.line(
            (PAD, TOP_PAD + goal_height - 7, WIDTH - PAD, TOP_PAD + goal_height - 7),
            fill=(43, 59, 80),
        )
        for n, (camera, label) in enumerate(CAMERAS):
            with Image.open(
                entry["source"] / "frames" / f"{frame_index:06d}_{camera}.png"
            ) as frame:
                if frame.size != (PANEL, PANEL):
                    raise ValueError(f"Unexpected camera dimensions: {frame.size}")
                canvas.paste(frame.convert("RGB"), (n * PANEL, banner + 30))
            draw.text((n * PANEL + 9, banner + 5), label, font=small, fill="white")
        for n, (line, color) in enumerate(entry["lines"]):
            draw.text(
                (PAD, TOP_PAD + goal_height + n * LINE),
                line,
                font=face,
                fill=color,
                anchor="lt",
            )
        for n, line in enumerate(entry["detail_lines"]):
            draw.text((PAD, metrics_y + n * 24), line, font=small, fill=MUTED, anchor="lt")
        phase = "Finishing phase (restored saved state)" if entry["segment"] else "Rollout"
        footer = f"{phase} · Physics step {frame_index} · Simulation {frame_index / fps:.2f} s"
        if entry["decision"]["frame_before"] == entry["decision"]["frame_after"]:
            footer += " · No-motion call: 1 s display hold"
        draw.text((10, height - 24), footer, font=small, fill=MUTED)
        return canvas

    temporary = destination.with_suffix(".part.mp4")
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{WIDTH}x{height}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-threads",
        "4",
        "-crf",
        "19",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    metadata = {
        "video": destination.name,
        "task_goal": results[0]["instruction"],
        "task_goal_caption": goal_caption,
        "yaw_caption_definition": (
            "Target yaw is a world-Z rotation relative to the initial gripper orientation. "
            "Recorded yaw shows measured call endpoints, not an additional model prediction. "
            "When yaw_deg is omitted, the previous target remains active."
        ),
        "model": model,
        "pricing": pricing,
        "totals": cumulative,
        "cumulative_scope": "This task, including saved-state finishing phases when present.",
        "generated_token_definition": "API output_tokens; includes reasoning_tokens, counted once.",
        "fps": fps,
        "resolution": [WIDTH, height],
        "layout": {
            "top_caption_height": banner,
            "camera_bottom_y": images_bottom,
            "metrics_y": metrics_y,
            "metrics_location": "below cameras",
            "bottom_tool_parameters": ["max_steps"],
        },
        "sources": [str(p.resolve()) for p in folders],
        "calls": [],
        "success": results[0]["success"],
        "result_card_seconds": 4,
        "playback": (
            "Motion at recorded control rate; API waiting time omitted; "
            "no-motion calls held for 1 second."
        ),
    }
    frame_count = 0
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for entry in entries:
            start = frame_count
            for index in entry["frames"]:
                canvas = canvas_for(entry, index)
                process.stdin.write(canvas.tobytes())
                frame_count += 1
            metadata["calls"].append(
                {
                    "segment": entry["segment"],
                    "call": entry["decision"]["decision"] + 1,
                    "caption": entry["heading"] + " " + entry["note"],
                    "orientation_caption": entry["orientation_caption"],
                    "arguments": entry["decision"]["tool"]["arguments"],
                    "usage": entry["usage"],
                    "cumulative_usage": entry["cumulative_usage"],
                    "metric_captions": entry["detail_lines"],
                    "video_frames": [start, frame_count - 1],
                    "source_frames": [
                        entry["decision"]["frame_before"],
                        entry["decision"]["frame_after"],
                    ],
                }
            )
        # Hold the final camera views while showing the native task result prominently.
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, WIDTH, banner - 1), fill=BACKGROUND)
        success = results[0]["success"]
        draw.text(
            (PAD, TOP_PAD),
            "SUCCESS" if success else "FAILURE",
            font=font(56),
            fill=(132, 227, 176) if success else (255, 157, 135),
            anchor="lt",
        )
        draw.text(
            (PAD, 72),
            results[0]["instruction"].capitalize(),
            font=font(24),
            fill="white",
            anchor="lt",
        )
        failure_reason = {
            "model_give_up": "model gave up",
            "budget": "step or call budget exhausted",
        }.get(results[0].get("termination_reason"), "task condition not met")
        outcome = (
            "LIBERO task success verified." if success else f"LIBERO failure: {failure_reason}."
        )
        draw.text((PAD, 109), outcome, font=font(18), fill=MUTED, anchor="lt")
        if len(results) > 1 and results[-1]["success"]:
            draw.text(
                (PAD, 139),
                "Includes the saved-state continuation: gripper release and withdrawal.",
                font=font(17),
                fill=MUTED,
                anchor="lt",
            )
        wall_time = sum(r["total_wall_time_s"] for r in results)
        draw.rectangle((0, images_bottom, WIDTH, height), fill=BACKGROUND)
        final_details = [
            f"{model} · {len(entries)} calls · Wall time {wall_time:.1f} s · "
            f"API time {cumulative['api_latency_s']:.2f} s",
            f"Accumulated: {cumulative['input_tokens']:,} input / "
            f"{cumulative['output_tokens']:,} generated tokens · "
            f"Estimated total API cost ${cumulative['estimated_cost_usd']:.4f}",
        ]
        for n, line in enumerate(final_details):
            draw.text((PAD, metrics_y + n * 24), line, font=small, fill="white", anchor="lt")
        draw.text(
            (10, height - 24),
            "Standard list-price estimate including cache writes. "
            "Generated tokens include reasoning.",
            font=small,
            fill=MUTED,
        )
        metadata["result_card_start_frame"] = frame_count
        final_bytes = canvas.tobytes()
        for _ in range(4 * fps):
            process.stdin.write(final_bytes)
            frame_count += 1
    finally:
        process.stdin.close()
        return_code = process.wait()
    if return_code:
        raise RuntimeError(f"ffmpeg exited with {return_code}")
    temporary.replace(destination)
    metadata.update(
        frame_count=frame_count, duration_s=frame_count / fps, size_bytes=destination.stat().st_size
    )
    (folder / "captioned_video.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        json.dumps(
            {
                "path": str(destination.resolve()),
                **{k: metadata[k] for k in ("duration_s", "size_bytes", "success")},
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    render(args.task_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
