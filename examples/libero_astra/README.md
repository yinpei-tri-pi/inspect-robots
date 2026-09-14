# Vision-language models + LIBERO

Runs GPT-6-astra (default) or a selected model as the sole task planner for a simulated Panda arm. Each API
decision sees external, wrist, and overhead RGB images, robot proprioception,
and an overhead table-plane calibration grid. No object poses, hidden goal
regions, learned robot policy, or demonstration trajectories are supplied.

This is a custom LIBERO demo added in this fork. Its system prompt, including
generic pick-and-place advice, and calibrated overhead grid were written for
this experiment. They are not copied from the upstream real-robot experiments.
Astra and Luna receive the same prompt and grid, but these runs do not reproduce
Robocurve's YAM setup.

`move_to` requests an absolute grasp-center XYZ, relative world-Z yaw, gripper
command, and maximum duration. A generic feedback loop compares the current
robot pose with that target and issues normalized 7D OSC_POSE actions at 20 Hz.
Robosuite computes arm torques and finger actuation; MuJoCo computes motion and
contacts. Objects are never teleported. LIBERO's native predicate decides success.

## Isolated environment

The existing installation is `.venv-libero` at the repo root, created by `uv`
with `include-system-site-packages = false`. CPU Torch, MuJoCo, robosuite, and
matching OSMesa / GLAPI native libraries are local to that environment. LIBERO
source and assets are imported read-only from OpenPI's third-party checkout.
`PYTHONDONTWRITEBYTECODE=1` keeps Python from writing caches into that source.
No packages were installed into OpenPI's environment or system Python.

On a fresh Ubuntu 22.04 checkout with LIBERO source/assets available:

```bash
bash examples/libero_astra/setup.sh /path/to/LIBERO
```

The installer refuses to modify an existing `.venv-libero`. `env.sh` scopes all
configuration, native-library paths, and CPU thread limits to the launched
process. It uses CPU OSMesa and disables CUDA visibility.

## Run

From the repository root, with `OPENAI_API_KEY` already set:

```bash
bash examples/libero_astra/env.sh examples/libero_astra/smoke_env.py
bash examples/libero_astra/env.sh examples/libero_astra/run.py --tasks 5 6 8
bash examples/libero_astra/env.sh examples/libero_astra/build_report.py
```

If your key is initialized only by `.bashrc`, launch the run from your ordinary
interactive shell. The runner does not save authorization headers or the key.

Open `artifacts/astra_libero/index.html`. HTML embeds the data and works directly
from a local file; keep the adjacent videos and frames in their directories.
Use `--attempt 1` for another attempt. Existing attempts are never overwritten.

Compare GPT-5.6 Luna using the same prompt, tools, medium reasoning, seed,
initial state, and control budgets. The reference check rejects a mismatched
initial simulator state before the first API call:

```bash
bash examples/libero_astra/env.sh examples/libero_astra/run.py \
  --model gpt-5.6-luna --tasks 5 6 8 \
  --reference-root artifacts/astra_libero --output artifacts/luna_libero
bash examples/libero_astra/env.sh examples/libero_astra/build_report.py \
  --root artifacts/luna_libero
```

`render_captioned_video.py --task-dir <attempt-folder>` renders saved frames
with the task goal, all tool arguments, public notes, target and measured yaw,
API latency, token usage, estimated cumulative cost, and the native result.
Pricing uses each model's documented standard rates, including cache writes.
The renderer uses existing logs and does not make API requests.

After rendering the three captioned videos for each model, build the comparison:

```bash
bash examples/libero_astra/env.sh examples/libero_astra/compare_models.py
```

`artifacts/libero_model_comparison/index.html` embeds all six videos and their
call records, so it can be downloaded and opened by itself. It compares native
success, wall time, API latency, tokens, and estimated cost; finishing phases
are excluded from the default totals and can be included with a checkbox.
Adjacent JSON and CSV files contain the comparison data. The builder also
checks initial states, camera images, prompts, tools, and first API requests.

If native success occurs before the model releases an object, a separate
finishing phase can restore the last saved MuJoCo state and let GPT release
the object and withdraw. This phase has its own API calls, logs, video, and
completion check (native success plus open fingers and grasp-center z >= 1.08 m):

```bash
bash examples/libero_astra/env.sh examples/libero_astra/run.py --tasks 8 \
  --continue-from artifacts/astra_libero/task_08_attempt_00 \
  --output artifacts/astra_libero/task_08_attempt_00/post_success \
  --max-calls 8 --max-steps 160
```

The saved MuJoCo state is restored exactly and checked against the original.
The controller is freshly initialized; this is a separately identified
continuation, not a claim that the original episode ran uninterrupted. There
are no scripted settling actions before the continuation's first observation.

## Evidence and limitations

Each attempt saves all API requests/responses (including retries and usage),
public action notes and tool arguments, every 7D action, robot states, images
from every control step, calibrated images shown to GPT, timestamps, simulator
states, XML, warmup actions/frames, and the exact runner source. API image data
are deduplicated as PNG blobs; `$blob:<sha256>` identifies the payload.

The video plays physical steps at 20 Hz, without API waiting time. Wall time,
API latency, and physics/render/log time are recorded separately. Reasoning
tokens are included in output tokens; cached tokens are included in input
tokens. Private chain-of-thought text is not exposed by the API; encrypted
reasoning is retained as returned, and public notes are shown in the viewer.

These are interactive demos, not standard benchmark scores: there is an added
calibrated overhead camera, generic pose primitives, up to 40 model decisions,
and an 800-control-step budget. Initial state index 0 and seed 7 are used.
The 10 standard settling steps precede each episode and are logged separately.
Completed attempts and unsuccessful attempts both remain in the index.

`sim_states.npz` aligns one state per saved frame and includes full simulator
state for later inspection; it is not given to GPT. Exact replay of control also
requires restoring controller state or replaying actions from the initial state.
`model.xml` retains local mesh paths, so state replay needs the same asset tree.
