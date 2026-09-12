# GPT-6 Astra: one CubePick rollout

Open `artifacts/astra_cubepick/report.html` in a browser. It is self-contained and works offline.
The review video, all API calls, predictions, state transitions, and token
counts are embedded. Click any API-call row to seek to its observation.

The actual rollout succeeded after 15 model calls and 15 executed actions.
The final target distance was 0.0000566864; success requires distance ≤ 0.05.

| Measurement | Value |
| --- | ---: |
| Rollout wall time | 46.395628 s |
| Total HTTP request time | 46.044298 s |
| Mean policy inference time | 3.081052 s |
| Input tokens | 25,968 |
| Output tokens | 1,136 |
| Total tokens | 27,104 |
| Cached input tokens | 0 |
| Reasoning tokens, included in output | 68 |

Wall time runs from the first policy decision's start through the final
completed simulator step. It excludes dependency installation, setup, final
scoring, and report generation. All 15 requests returned HTTP 200, reported
the model as `gpt-6-astra`, and included usage. No retries were needed.

This uses the original CubePick environment: a 2D point reaches a target,
with both coordinates and a 32×32 camera image available to the model. It
has no grasping or rigid-body physics. The policy uses the Responses API,
medium reasoning effort, and full image history. The controller executes
one action from each predicted chunk, then requests a fresh prediction.

## Files

- `report.html`: interactive report with video, timing, usage, and exact calls.
- `full_log.html`: native Inspect Robots transcript and API-wire report.
- `review.mp4`: annotated 1200×720 video; each observation lasts 1.5 seconds.
- `camera_nominal_10hz.mp4`: original camera frames enlarged with nearest-neighbor sampling, at the declared 10 Hz.
- `summary.json`, `usage.json`, `calls.csv`: aggregate and per-request measurements.
- `predictions.jsonl`: full predicted chunks, observation state, and conversation additions.
- `steps.jsonl`: actual executed actions and before/after states.
- `logs/eval.json`: schema-compatible evaluation summary.
- `logs/wire/`: complete API request/response capture and exact image blobs.
- `logs/frames/`: reset and all 15 post-action camera frames as NumPy arrays.
- `logs/transcripts/`: the policy conversation, including tool arguments and results.
- `logs/actions/`: the framework's executed-action log.
- `console.log`: original rollout console output.
- `data_validation.json`, `browser_validation.json`: data and browser verification.
- `checksums.json`: SHA-256 hashes of the retained files.

There are 16 camera observations: one initial frame and 15 post-action
frames. The review video is 24 seconds long. Nominal simulated action time
is 1.5 seconds; the 10 Hz camera video is 1.6 seconds because it also displays
the initial frame. Neither video reproduces API waiting time.

## Summary provenance

The rollout completed successfully and all underlying data was flushed.
The custom logging sink initially omitted the framework's default JSON sink,
so summary finalization failed after the final action. `recover_summary.py`
rebuilt the summary from the original API captures and recorded transitions,
checking every executed action, state update, distance, and usage total.
It made no model calls and did not run another rollout. The provenance and
the exact timing definition are also recorded in `run_manifest.json`.
The runner now explicitly includes the JSON sink.

## Rebuild the reports

From the repository root, using the installed local environment:

```bash
bash examples/libero_astra/env.sh examples/cubepick_astra/build_report.py
```

The renderer makes no API calls. It uses imageio-ffmpeg, with an optional
`FFMPEG_BINARY` override. The isolated LIBERO environment already includes the
required packages. Generated reports and large logs remain local under
`artifacts/` and are excluded from Git.

`run_rollout.py` is the instrumented runner. It refuses to overwrite this
run. On a fresh checkout, run it with `OPENAI_API_KEY` set:

```bash
bash examples/libero_astra/env.sh examples/cubepick_astra/run_rollout.py
```

The measurements above document the original local run; its media and raw
logs are not included in the source checkout.
