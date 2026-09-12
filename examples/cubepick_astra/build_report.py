# Inline HTML/CSS/JavaScript preserves intentional typography and template layout.
# ruff: noqa: E501, RUF001
"""Build a portable HTML replay and videos from the recorded rollout only."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import subprocess
import textwrap
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from inspect_robots._html import render_html
from inspect_robots.log import read_eval_log

ROOT = Path(__file__).resolve().parents[2] / "artifacts/astra_cubepick"
FFMPEG = os.environ.get("FFMPEG_BINARY") or imageio_ffmpeg.get_ffmpeg_exe()
REVIEW_HOLD = 1.5


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def png_data(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def encode_video(path, images, fps, repeats=1):
    first = images[0]
    with subprocess.Popen(
        [
            FFMPEG,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{first.width}x{first.height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        stdin=subprocess.PIPE,
    ) as process:
        for image in images:
            for _ in range(repeats):
                process.stdin.write(image.convert("RGB").tobytes())
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("Video encoding failed")


def function_calls(response):
    return [item for item in response.get("output", []) if item.get("type") == "function_call"]


def main():
    manifest = json.loads((ROOT / "run_manifest.json").read_text())
    log_path = ROOT / manifest["log_path"]
    log = read_eval_log(log_path)
    predictions = read_lines(ROOT / "predictions.jsonl")
    steps = read_lines(ROOT / "steps.jsonl")
    usage = json.loads((ROOT / "usage.json").read_text())
    wire_path = log_path.parent / manifest["trial_metadata"]["wire_capture"]
    wire = read_lines(wire_path)
    frame_paths = sorted(Path(log.stats.frames_dir).glob("layout-0-e0_top_*.npy"))
    arrays = [np.load(path) for path in frame_paths]
    if len(arrays) != len(steps) + 1:
        raise AssertionError("Expected reset frame and one frame per executed action")
    calls = []
    for row in wire:
        prediction = next(
            (
                p
                for p in predictions
                if p["first_api_call"] <= row["call"] < p["first_api_call"] + p["api_calls"]
            ),
            None,
        )
        response = row.get("response") or {}
        calls.append(
            {
                **row,
                "env_step": prediction["env_step"] if prediction else None,
                "tools": function_calls(response),
                "usage": response.get("usage", {}),
            }
        )
    api_time = sum(row["duration_s"] for row in calls)
    success = log.results.metrics.get("success_at_end") == 1.0
    summary = {
        **manifest,
        "success": success,
        "api_attempts": len(calls),
        "llm_calls": manifest["trial_metadata"]["llm_usage"]["llm_calls"],
        "api_request_time_s": api_time,
        "nominal_simulation_time_s": len(steps) / 10,
        "review_video_time_s": len(arrays) * REVIEW_HOLD,
        "tokens": usage["totals"],
        "final_distance": steps[-1]["info"]["distance"],
        "frame_count": len(arrays),
        "usage_note": "Cached input is included in input tokens; reasoning is included in output tokens. Totals sum provider usage across all attempts that returned usage.",
    }
    save_json(ROOT / "summary.json", summary)
    with (ROOT / "calls.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "call",
                "attempt",
                "env_step",
                "duration_s",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "cached_input_tokens",
                "reasoning_tokens",
                "tools",
            ],
        )
        writer.writeheader()
        for call in calls:
            u = call["usage"]
            writer.writerow(
                {
                    "call": call["call"] + 1,
                    "attempt": call["attempt"] + 1,
                    "env_step": call["env_step"],
                    "duration_s": call["duration_s"],
                    **{
                        key: u.get(key) for key in ("input_tokens", "output_tokens", "total_tokens")
                    },
                    "cached_input_tokens": u.get("input_tokens_details", {}).get("cached_tokens"),
                    "reasoning_tokens": u.get("output_tokens_details", {}).get("reasoning_tokens"),
                    "tools": json.dumps(call["tools"]),
                }
            )

    font_path = "/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf"
    font = ImageFont.truetype(font_path, 23)
    small = ImageFont.truetype(font_path, 19)
    title_font = ImageFont.truetype(font_path, 30)
    video_frames = []
    for index, array in enumerate(arrays):
        image = Image.new("RGB", (1200, 720), "#111827")
        draw = ImageDraw.Draw(image)
        draw.text((32, 24), "GPT-6 Astra / CubePick", font=title_font, fill="white")
        draw.text(
            (32, 70),
            f"Observation {index} of {len(steps)}  |  2D reaching",
            font=font,
            fill="#a5b4fc",
        )
        image.paste(Image.fromarray(array).resize((480, 480), Image.Resampling.NEAREST), (32, 118))
        draw.rectangle((31, 117, 512, 598), outline="#64748b", width=1)
        draw.text((32, 615), "Red: end effector   /   Green: target", font=small, fill="#e2e8f0")
        draw.text((32, 648), "Original 32 x 32 camera, enlarged", font=small, fill="#94a3b8")
        if index < len(steps):
            step = steps[index]
            call = next(call for call in calls if call["env_step"] == index)
            prediction = next(p for p in predictions if p["env_step"] == index)
            tool = call["tools"][0] if call["tools"] else {}
            args = json.loads(tool.get("arguments", "{}"))
            text = [
                f"API call {call['call'] + 1}: {tool.get('name', 'text')}",
                f"Request time: {call['duration_s']:.2f} s",
                f"Tokens: {call['usage'].get('input_tokens', '?')} in / {call['usage'].get('output_tokens', '?')} out",
                "",
                "Tool parameters: " + json.dumps({k: v for k, v in args.items() if k != "note"}),
                f"Predicted chunk: {prediction['chunk_length']} actions",
                "Executed step: " + str([round(x, 5) for x in step["executed_action"]]),
                "",
                args.get("note", ""),
            ]
        else:
            text = [
                "SUCCESS" if success else "Trial ended",
                "",
                f"Final distance: {summary['final_distance']:.5f}",
                "Success threshold: 0.05",
                f"Executed steps: {len(steps)}",
                f"Rollout wall time: {summary['wall_time_s']:.2f} s",
                f"API calls: {summary['llm_calls']}",
                f"Tokens: {summary['tokens'].get('total_tokens', '?'):,}",
            ]
        y = 123
        for line in text:
            for wrapped in textwrap.wrap(line, width=49) or [""]:
                draw.text((548, y), wrapped, font=font, fill="#e2e8f0")
                y += 30
        draw.text((548, 661), "Review playback: 1.5 s per observation", font=small, fill="#94a3b8")
        video_frames.append(image)
    encode_video(ROOT / "review.mp4", video_frames, fps=10, repeats=15)
    encode_video(
        ROOT / "camera_nominal_10hz.mp4",
        [Image.fromarray(a).resize((512, 512), Image.Resampling.NEAREST) for a in arrays],
        fps=10,
    )

    # Keep the repository's own full transcript / wire report as a second view.
    tools_dir = ROOT / "bin"
    tools_dir.mkdir(exist_ok=True)
    link = tools_dir / "ffmpeg"
    if not link.exists():
        link.symlink_to(FFMPEG)
    os.environ["PATH"] = str(tools_dir) + os.pathsep + os.environ.get("PATH", "")
    os.environ["TMPDIR"] = "/dev/shm"
    import tempfile

    tempfile.tempdir = "/dev/shm"
    native = render_html(
        log,
        title="GPT-6 Astra — full Inspect Robots log",
        log_path=log_path,
        frames_dir=Path(log.stats.frames_dir),
    )
    (ROOT / "full_log.html").write_text(native)

    data = {
        "summary": summary,
        "calls": calls,
        "predictions": predictions,
        "steps": steps,
        "frames": [png_data(a) for a in arrays],
        "reviewHold": REVIEW_HOLD,
        "eval": log.to_dict(),
    }
    payload = (
        json.dumps(data, allow_nan=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    page = TEMPLATE.replace("__DATA__", payload).replace(
        "__VIDEO__", base64.b64encode((ROOT / "review.mp4").read_bytes()).decode()
    )
    (ROOT / "report.html").write_text(page)
    save_json(
        ROOT / "checksums.json",
        {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in ROOT.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and "__pycache__" not in path.parts
            and path.name != "checksums.json"
        },
    )
    print(json.dumps(summary, indent=2))


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GPT-6 Astra · CubePick rollout</title>
<style>
:root{color-scheme:dark;--bg:#0b1020;--panel:#131c30;--line:#2b3853;--text:#edf2ff;--muted:#a1b0cb;--accent:#9cadff;--green:#73e3b2}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}main{max-width:1500px;margin:auto;padding:32px}h1{font-size:32px;margin:6px 0}h2{font-size:20px;margin:0 0 14px}h3{font-size:16px;margin:16px 0 8px}p{margin:8px 0}a{color:var(--accent)}.muted{color:var(--muted)}.eyebrow{color:var(--accent);font-size:12px;letter-spacing:.12em;text-transform:uppercase}.badge{display:inline-block;background:#133e33;color:var(--green);border-radius:20px;padding:4px 12px;font-size:13px;margin-left:12px}.tiles{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin:24px 0}.tile,.panel{border:1px solid var(--line);border-radius:12px;background:var(--panel)}.tile{padding:16px}.tile b{display:block;font-size:24px;margin:4px 0}.tile span,.tile small{color:var(--muted);font-size:12px}.panel{padding:20px;margin-bottom:18px}.replay{display:grid;grid-template-columns:1.5fr 1fr;gap:18px}.replay .panel{min-width:0}video{display:block;width:100%;border-radius:8px;background:#000}.controls{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:16px 0 6px}button,select{background:#1b2945;color:var(--text);border:1px solid #435376;padding:7px 12px;border-radius:7px;cursor:pointer}button:hover{border-color:var(--accent)}input[type=range]{flex:1;min-width:100px;accent-color:var(--accent)}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}.pair img{width:100%;image-rendering:pixelated;border-radius:6px;background:#000}.label{font-size:12px;color:var(--muted);margin:5px 0}.note{padding:12px 14px;border-left:3px solid var(--accent);background:#19253f;border-radius:0 8px 8px 0;min-height:60px}.code,pre{font:12px/1.5 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;background:#0b1324;border-radius:7px;padding:12px;margin:8px 0;max-height:380px;overflow:auto}.tablewrap{max-height:440px;overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}th{text-align:left;color:var(--muted);font-weight:500;position:sticky;top:0;background:var(--panel)}td,th{padding:9px 11px;border-bottom:1px solid var(--line);white-space:nowrap}tbody tr{cursor:pointer}tbody tr:hover,tbody tr.active{background:#233359}td:first-child{color:var(--accent)}details{margin:12px 0}summary{cursor:pointer;color:var(--accent)}.footer{font-size:13px;color:var(--muted);margin:24px 0}.download{display:flex;gap:10px;flex-wrap:wrap}.spaced{margin:18px 0}.metricline{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;color:var(--muted)}@media(max-width:1000px){.tiles{grid-template-columns:repeat(3,1fr)}.replay{grid-template-columns:1fr}}@media(max-width:600px){main{padding:16px}.tiles{grid-template-columns:repeat(2,1fr)}h1{font-size:25px}.panel{padding:14px}}
</style></head><body><main>
<div class="eyebrow">Recorded robot policy evaluation</div>
<h1>GPT-6 Astra · CubePick <span id="outcome" class="badge"></span></h1>
<p class="muted">One real API rollout · “reach the cube” · Responses API · medium reasoning effort</p>
<p class="muted">Toy 2D environment. The model receives the camera image and both object coordinates. Red is the end effector; green is the target.</p>
<div class="tiles" id="tiles"></div>
<div class="replay">
<section class="panel"><h2>Video replay</h2>
<video id="video" controls preload="metadata" src="data:video/mp4;base64,__VIDEO__"></video>
<div class="controls"><button id="prev">← Previous</button><input id="seek" type="range" min="0" value="0" aria-label="Observation step"><button id="next">Next →</button><span id="position"></span></div>
<p class="label">Review video holds each observation for 1.5 seconds. API waiting time is omitted from playback and reported separately. Use the slider or click any API call below.</p>
<div class="controls"><label>Playback speed <select id="speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label><button id="downloadVideo">Download video</button></div>
</section>
<section class="panel"><h2 id="decisionTitle">Prediction</h2><div id="callStats" class="metricline"></div>
<h3>Model’s note</h3><div id="note" class="note"></div>
<h3>Tool call and parameters</h3><pre id="tool"></pre>
<div class="pair"><div><div class="label">Observation before action</div><img id="before" alt="Original camera before action"></div><div><div class="label">Observation after action</div><img id="after" alt="Original camera after action"></div></div>
<details open><summary>Predicted chunk versus executed action</summary><pre id="actions"></pre></details>
</section></div>
<section class="panel"><h2>Every API call: time and token usage</h2>
<p class="muted">Click a row to inspect that call and seek to its observation. Input counts include cached tokens; output counts include reasoning tokens. These subsets are not added again.</p>
<div class="tablewrap"><table><thead><tr><th>Call / attempt</th><th>Step</th><th>Seconds</th><th>Input</th><th>Cached input</th><th>Output</th><th>Reasoning</th><th>Total</th><th>Tool</th></tr></thead><tbody id="calls"></tbody></table></div>
<div class="spaced metricline" id="timing"></div>
</section>
<section class="panel"><h2>Selected call: full recorded data</h2>
<p class="muted">Requests retain image blob references; the exact observation images are embedded above, and the original blobs are included with the raw logs. Responses include all output items returned by the provider.</p>
<details><summary>Raw API request</summary><pre id="request"></pre></details>
<details><summary>Raw API response and usage</summary><pre id="response"></pre></details>
<details><summary>Tool results and conversation additions</summary><pre id="messages"></pre></details>
<details><summary>Exact state transition</summary><pre id="state"></pre></details>
<details><summary>Run configuration, result, and timing</summary><pre id="config"></pre></details>
<div class="download"><button data-download="summary">Download summary JSON</button><button data-download="calls">Download API calls JSON</button><button data-download="predictions">Download predictions JSON</button><button data-download="steps">Download executed steps JSON</button><a href="full_log.html">Open native Inspect Robots report</a></div>
</section>
<p class="footer">The controller executes one action from each predicted chunk before replanning. The original environment terminates automatically when distance ≤ 0.05. All replay media and selected-call data are embedded in this HTML; it works offline.</p>
</main><script type="application/json" id="data">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent), S=D.summary;
const $=id=>document.getElementById(id), pretty=x=>JSON.stringify(x,null,2), fmt=x=>Number.isFinite(x)?x.toLocaleString('en-US'):'—';
const video=$('video');let chosenCall=-1, current=-1;
$('outcome').textContent=S.success?'Success':'Trial ended';
const tileData=[['Rollout wall time',S.wall_time_s.toFixed(2)+' s','Includes inference and simulation'],['API calls',S.llm_calls,S.api_attempts+' HTTP attempts'],['Input tokens',fmt(S.tokens.input_tokens),fmt(S.tokens.cached_input_tokens)+' cached'],['Output tokens',fmt(S.tokens.output_tokens),fmt(S.tokens.reasoning_tokens)+' reasoning'],['Total tokens',fmt(S.tokens.total_tokens),'Provider-reported usage'],['Executed steps',S.total_steps,'Final distance '+S.final_distance.toFixed(5)]];
for(const [label,value,sub] of tileData){const div=document.createElement('div');div.className='tile';for(const [tag,text] of [['span',label],['b',value],['small',sub]]){const el=document.createElement(tag);el.textContent=text;div.append(el)}$('tiles').append(div)}
$('config').textContent=pretty(S);$('seek').max=D.frames.length-1;
$('timing').textContent=`API request time: ${S.api_request_time_s.toFixed(2)} s · Mean policy inference: ${S.mean_inference_latency_s.toFixed(2)} s · Nominal simulation time: ${S.nominal_simulation_time_s.toFixed(1)} s · Review video: ${S.review_video_time_s.toFixed(1)} s`;
D.calls.forEach((c,i)=>{const u=c.usage||{},tr=document.createElement('tr');tr.dataset.call=i;const vals=[`${c.call+1} / ${c.attempt+1}`,c.env_step,c.duration_s.toFixed(3),fmt(u.input_tokens),fmt(u.input_tokens_details?.cached_tokens),fmt(u.output_tokens),fmt(u.output_tokens_details?.reasoning_tokens),fmt(u.total_tokens),c.tools.map(t=>t.name).join(', ')||'Text'];for(const v of vals){const td=document.createElement('td');td.textContent=v;tr.append(td)}tr.addEventListener('click',()=>select(c.env_step,true,i));$('calls').append(tr)});
function select(index,seekVideo=false,callIndex=null){index=Math.max(0,Math.min(D.frames.length-1,index));if(seekVideo){video.pause();video.currentTime=index*D.reviewHold+0.01}current=index;$('seek').value=index;$('position').textContent=`Step ${index} / ${D.steps.length}`;
let ci=callIndex??D.calls.findIndex(c=>c.env_step===index);chosenCall=ci;document.querySelectorAll('#calls tr').forEach((r,i)=>r.classList.toggle('active',i===ci));
$('before').src=D.frames[index];$('after').src=D.frames[Math.min(index+1,D.frames.length-1)];
const step=D.steps[index],p=D.predictions.find(p=>p.env_step===index),c=D.calls[ci];
if(!c){$('decisionTitle').textContent='Final observation';$('callStats').textContent='Environment-reported termination';$('note').textContent=S.success?'The simulator detected success. No additional model call was needed.':'The rollout has ended.';for(const name of ['tool','request','response','messages'])$(name).textContent='No API call at the final observation.';$('actions').textContent=pretty({final_state:D.steps.at(-1).after,distance:S.final_distance,success:S.success});$('state').textContent=pretty(D.steps.at(-1));return}
$('decisionTitle').textContent=`Call ${c.call+1} · observation ${index}`;$('callStats').textContent=`${c.duration_s.toFixed(3)} s · ${fmt(c.usage.input_tokens)} input · ${fmt(c.usage.output_tokens)} output tokens`;
const parsed=c.tools.map(t=>{let arguments;try{arguments=JSON.parse(t.arguments)}catch{arguments=t.arguments}return {name:t.name,arguments,call_id:t.call_id}});
$('note').textContent=parsed.map(t=>t.arguments?.note).filter(Boolean).join('\n')||'(No motion note returned)';$('tool').textContent=pretty(parsed);$('actions').textContent=pretty({predicted_actions:p?.predicted_actions,executed_action:step?.executed_action,execution_metadata:step?.action_metadata,controller:'Execute first action, then replan'});$('request').textContent=pretty(c.request);$('response').textContent=pretty(c.response);$('messages').textContent=pretty(p?.messages);$('state').textContent=pretty(step)}
video.addEventListener('timeupdate',()=>{const index=Math.min(D.frames.length-1,Math.floor(video.currentTime/D.reviewHold));if(index!==current)select(index)});
$('seek').addEventListener('input',e=>select(Number(e.target.value),true));$('prev').onclick=()=>select(current-1,true);$('next').onclick=()=>select(current+1,true);$('speed').onchange=e=>video.playbackRate=Number(e.target.value);
function download(blob,name){const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}
document.querySelectorAll('[data-download]').forEach(b=>b.onclick=()=>download(new Blob([pretty(D[b.dataset.download])],{type:'application/json'}),b.dataset.download+'.json'));
$('downloadVideo').onclick=()=>{const bytes=Uint8Array.from(atob(video.src.split(',')[1]),c=>c.charCodeAt(0));download(new Blob([bytes],{type:'video/mp4'}),'astra-cubepick-review.mp4')};select(0);
</script></body></html>"""


if __name__ == "__main__":
    main()
