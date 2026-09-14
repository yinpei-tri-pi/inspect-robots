# Inline HTML/CSS/JavaScript preserves intentional typography and template layout.
# ruff: noqa: E501, RUF001
"""Build offline, interactive LIBERO rollout reports from saved evidence."""

from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont


def read_lines(path):
    return (
        [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []
    )


def encode_video(folder, decisions, overwrite=False):
    frames = sorted((folder / "frames").glob("[0-9]*_agentview.png"))
    dest = folder / "review.mp4"
    if not frames or (dest.exists() and not overwrite):
        return
    font = ImageFont.load_default(size=19)
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
        "1152x440",
        "-pix_fmt",
        "rgb24",
        "-r",
        "20",
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
        str(dest),
    ]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    for index, _path in enumerate(frames):
        canvas = Image.new("RGB", (1152, 440), (13, 19, 31))
        draw = ImageDraw.Draw(canvas)
        decision = next(
            (d for d in decisions if d["frame_before"] <= index <= d["frame_after"]), None
        )
        for n, (camera, label) in enumerate(
            [
                ("agentview", "External · Panda arm"),
                ("robot0_eye_in_hand", "Wrist camera"),
                ("birdview", "Overhead camera"),
            ]
        ):
            frame = folder / "frames" / f"{index:06d}_{camera}.png"
            canvas.paste(Image.open(frame).convert("RGB"), (n * 384, 30))
            draw.text((n * 384 + 9, 4), label, font=font, fill="white")
        line = f"Physics step {index}  |  Simulation {index / 20:.2f} s"
        if decision:
            tool = decision.get("tool", {})
            line += f"  |  GPT call {decision['decision'] + 1}: {tool.get('name', 'text')}"
        draw.text((10, 416), line, font=font, fill="white")
        proc.stdin.write(canvas.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError("ffmpeg failed")


STYLE = """
:root{color-scheme:dark;--bg:#0c1220;--panel:#151f30;--line:#2b3b50;--text:#e5edf6;--muted:#a2b1c4;--blue:#81bbff;--green:#84e3b0;--orange:#ffbf85}
*{box-sizing:border-box}body{background:var(--bg);color:var(--text);font:15px/1.55 system-ui,sans-serif;margin:0}main{max-width:1500px;margin:auto;padding:28px}h1{font-size:29px;margin:10px 0}h2{font-size:20px;margin:0 0 12px}p{color:var(--muted)}a{color:var(--blue)}button,select{background:#24364f;color:var(--text);border:1px solid #527296;border-radius:7px;padding:8px 12px;cursor:pointer}button:hover{background:#365374}button:focus-visible,a:focus-visible{outline:3px solid var(--blue)}.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:20px;margin:16px 0}.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}.metric{background:var(--panel);padding:14px;border-radius:9px}.metric b{font-size:23px;display:block}.metric span,.muted{color:var(--muted);font-size:13px}.ok{color:var(--green)}.fail{color:var(--orange)}video{display:block;width:100%;max-height:580px;background:#070b12;border-radius:8px}.layout{display:grid;grid-template-columns:minmax(240px,340px) minmax(0,1fr);gap:18px}.calls{max-height:780px;overflow:auto}.call{display:block;text-align:left;width:100%;margin-bottom:8px;background:var(--panel);padding:12px;line-height:1.45;border-color:var(--line)}.call.active{border-color:var(--blue);background:#263f5d}.call small{display:block;color:var(--muted)}.photos{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.photos img{width:100%;border-radius:6px}.photos figure{margin:0}.photos figcaption{font-size:12px;color:var(--muted)}pre{background:#0b1423;border:1px solid var(--line);padding:14px;border-radius:8px;white-space:pre-wrap;overflow-wrap:anywhere;max-height:500px;overflow:auto;font:13px/1.5 ui-monospace,monospace}details{margin:12px 0}summary{cursor:pointer;color:var(--blue)}.controls{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:14px 0}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid var(--line);padding:9px;text-align:left;vertical-align:top}.scroll{overflow:auto}input[type=range]{flex:1;min-width:200px}.files{display:flex;gap:16px;flex-wrap:wrap}.taskcard{display:block;text-decoration:none;color:var(--text)}.taskcard:hover{border-color:var(--blue)}.note{border-left:3px solid var(--blue);padding-left:14px}.tag{font-size:12px;border-radius:30px;padding:4px 9px;background:#2b3b50}.inline{display:flex;justify-content:space-between;gap:14px;align-items:center}
@media(max-width:850px){main{padding:15px}.layout{grid-template-columns:1fr}.calls{max-height:260px}.photos{grid-template-columns:1fr}.photos img{max-width:480px}.metric b{font-size:20px}.panel{padding:13px}}
"""


SCRIPT = r"""
const data=JSON.parse(document.getElementById('data').textContent), ds=data.decisions, actions=data.actions;
const $=id=>document.getElementById(id), fmt=o=>JSON.stringify(o,null,2), esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let selected=0; const video=$('video');
const framePath=(i,cam)=>`frames/${String(i).padStart(6,'0')}_${cam}.png`;
function photos(id,frame,grid=false){$(id).innerHTML=['agentview','robot0_eye_in_hand','birdview'].map((cam,i)=>`<figure><a href="${grid&&i===2?`frames/decision_${String(selected).padStart(3,'0')}_grid.png`:framePath(frame,cam)}" target="_blank"><img loading="lazy" src="${grid&&i===2?`frames/decision_${String(selected).padStart(3,'0')}_grid.png`:framePath(frame,cam)}" alt="${['External Panda view','Wrist view','Overhead view'][i]} at frame ${frame}"></a><figcaption>${['External Panda view','Wrist view',grid?'Calibrated overhead sent to GPT':'Overhead view'][i]}</figcaption></figure>`).join('')}
function select(i,seek=true){if(!ds.length)return;selected=Math.max(0,Math.min(ds.length-1,i));const d=ds[selected],t=d.tool||{name:'text',arguments:{}},wire=data.wire.filter(w=>w.call===d.decision),u=wire.map(w=>w.response?.usage||{}).reduce((a,u)=>({input_tokens:a.input_tokens+(u.input_tokens||0),output_tokens:a.output_tokens+(u.output_tokens||0),total_tokens:a.total_tokens+(u.total_tokens||0),reasoning_tokens:a.reasoning_tokens+(u.output_tokens_details?.reasoning_tokens||0),cached_input_tokens:a.cached_input_tokens+(u.input_tokens_details?.cached_tokens||0)}),{input_tokens:0,output_tokens:0,total_tokens:0,reasoning_tokens:0,cached_input_tokens:0});
$('calltitle').textContent=`Call ${d.decision+1}: ${t.name}`;$('video-call').textContent=`Call ${d.decision+1} · ${t.name} ${t.arguments.position?JSON.stringify(t.arguments.position):''} ${t.arguments.gripper||''} — ${t.arguments.note||''}`;$('note').textContent=t.arguments.note||d.assistant.content||'No public explanation returned.';$('params').textContent=fmt(t.arguments);$('result').textContent=fmt(d.result);$('callstats').textContent=`API ${d.api_latency_s.toFixed(2)} s · Physics/render/log ${Number(d.simulation_time_s||0).toFixed(2)} s · ${u.total_tokens.toLocaleString()} tokens · Steps ${d.step_start} → ${d.step_end}${d.timing_reconstructed?' · recovered timing (physics duration approximate)':''}`;
$('usage').textContent=fmt(u);$('statebefore').textContent=fmt(d.state_before);$('stateafter').textContent=fmt(d.state_after);$('raw').textContent=fmt(wire);$('prediction').textContent=fmt(d.assistant);photos('before',d.frame_before,true);photos('after',d.frame_after);
const aa=actions.filter(a=>a.decision===d.decision);$('actionrows').innerHTML=aa.map(a=>`<tr><td>${a.step}</td><td>${a.action.map(v=>v.toFixed(5)).join(', ')}</td><td>${a.after.robot0_eef_pos.map(v=>v.toFixed(4)).join(', ')}</td><td>${a.reward}</td><td>${a.done}</td></tr>`).join('');$('actionjson').textContent=fmt(aa);
document.querySelectorAll('.call').forEach((b,j)=>b.classList.toggle('active',j===selected));if(seek)video.currentTime=d.frame_before/20;
}
$('calls').innerHTML=ds.map((d,i)=>`<button class="call" data-index="${i}"><b>${i+1}. ${esc(d.tool?.name||'text')}</b><small>${esc(d.tool?.arguments.note||'No tool returned')}</small><small>${d.api_latency_s.toFixed(1)} s API · ${d.step_end-d.step_start} motor steps</small></button>`).join('');
$('calls').addEventListener('click',e=>{const b=e.target.closest('button');if(b)select(Number(b.dataset.index));});
$('prev').onclick=()=>select(selected-1);$('next').onclick=()=>select(selected+1);$('playcall').onclick=()=>{video.currentTime=ds[selected].frame_before/20;video.play();};
video.addEventListener('timeupdate',()=>{$('videotime').textContent=`Simulation ${video.currentTime.toFixed(2)} s`;if($('follow').checked&&!video.paused){const f=Math.floor(video.currentTime*20+1e-5);const i=ds.findIndex(d=>d.frame_before<=f&&f<d.frame_after);if(i>=0&&i!==selected)select(i,false);}});
$('speed').onchange=e=>video.playbackRate=Number(e.target.value);
$('frame').max=Math.max(0,data.observations.length-1);$('frame').oninput=e=>{const i=Number(e.target.value);$('frametext').textContent=`Physics frame ${i} / ${data.observations.length-1}`;photos('allframes',i);$('framestate').textContent=fmt(data.observations[i]);};$('frame').dispatchEvent(new Event('input'));
select(0,false);
"""


def build_task(folder, overwrite_video=False):
    meta = json.loads((folder / "result.json").read_text())
    decisions = read_lines(folder / "decisions.jsonl")
    data = {
        "metadata": meta,
        "decisions": decisions,
        "actions": read_lines(folder / "actions.jsonl"),
        "observations": read_lines(folder / "observations.jsonl"),
        "wire": read_lines(folder / "wire/api/episode/calls.jsonl"),
    }
    if not data["observations"]:
        (folder / "report.html").write_text(
            f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Attempt setup error</title><style>{STYLE}</style><main><a href="../index.html">← All attempts</a><h1>Attempt ended before the first frame</h1><pre>{html.escape(json.dumps(meta, indent=2))}</pre><a href="result.json">Raw result</a></main></html>'
        )
        return meta
    encode_video(folder, decisions, overwrite_video)
    u = meta.get("usage", {})
    success = bool(meta.get("success"))
    termination = meta.get("termination_reason", meta.get("status"))
    termination = {
        "model_give_up": "Model stopped after unsuccessful recovery",
        "budget": "Step or call budget exhausted",
        "error": "Execution error",
    }.get(termination, termination)
    status = "SUCCESS · LIBERO predicate satisfied" if success else f"UNSUCCESSFUL · {termination}"
    cards = [
        ("Result", "Success" if success else "Unsuccessful"),
        ("GPT decisions", len(decisions)),
        ("Motor steps", meta.get("control_steps", 0)),
        ("Wall time", f"{meta.get('total_wall_time_s', 0):.1f} s"),
        ("API time", f"{meta.get('api_request_time_s', 0):.1f} s"),
        ("Tokens", f"{u.get('total_tokens', 0):,}"),
    ]
    metrics = "".join(
        f'<div class="metric"><span>{label}</span><b>{value}</b></div>' for label, value in cards
    )
    payload = json.dumps(data).replace("<", "\\u003c")
    title = html.escape(meta["instruction"])
    model_label = html.escape(meta["model"])
    finishing = ""
    if meta.get("phase"):
        finishing = '<section class="panel"><h2>Finishing continuation</h2><p>This phase starts from the exact saved simulator state after benchmark success. GPT must release the object and withdraw the open gripper above 1.08 m while preserving LIBERO success. It is logged separately from the benchmark episode and uses no settling actions before the first model observation.</p></section>'
    if meta.get("logging_recovery"):
        finishing += '<section class="panel"><h2>Log recovery note</h2><p>The release and withdrawal completed, but saving the last tool result failed because of a NumPy Boolean. The final decision record was recovered from the original API response, motor actions, and saved frames. No model requests or physics steps were repeated. Total wall time and token usage are unchanged; the last call’s physics/render/log duration is an estimate from saved timestamps.</p><a href="recovery_provenance.json">Recovery evidence and original error</a> · <a href="recovery_originals/result.json">Original incomplete result</a></section>'
    for extra in sorted((folder / "post_success").glob("task_*/result.json")):
        extra_meta = json.loads(extra.read_text())
        relative = extra.parent.relative_to(folder).as_posix()
        finishing += f'<section class="panel"><h2>After success: release and withdraw</h2><p>The original episode stopped at first LIBERO success while the object was still held. GPT then continued from the exact saved state to finish releasing it and moving the hand clear. This separate phase used {extra_meta.get("decisions", 0)} calls, {extra_meta.get("total_wall_time_s", 0):.1f} seconds, and {extra_meta.get("usage", {}).get("total_tokens", 0):,} tokens. Completion: {"verified" if extra_meta.get("success") else "not achieved"}.</p><video controls preload="metadata" src="{relative}/review.mp4"></video><p><a href="{relative}/report.html">Inspect all finishing-phase observations, calls, actions, and timing →</a></p></section>'
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title} · {model_label} LIBERO</title><style>{STYLE}</style><main>
<a href="../index.html">← All attempts</a><h1>{title}</h1><p><span class="tag">{model_label}</span> LIBERO Goal · task {meta["task_id"]} · attempt {meta["attempt"]} · initial state {meta["init_state_index"]}</p><p class="{"ok" if success else "fail"}">{status}</p><div class="metrics">{metrics}</div>
<section class="panel"><h2>Robot rollout</h2><video id="video" controls preload="metadata" src="review.mp4" poster="frames/000000_agentview.png"></video><div class="controls"><span id="videotime">Simulation 0.00 s</span><label><input id="follow" type="checkbox" checked> Follow video calls</label><label>Speed <select id="speed"><option value="0.5">0.5×</option><option selected value="1">1×</option><option value="2">2×</option></select></label><a href="review.mp4" download>Download MP4</a></div><p id="video-call" class="note"></p><p>Video runs at the simulator’s 20 Hz. API waiting time is omitted from playback and recorded below. The Panda moves objects through MuJoCo contacts and actuated fingers.</p></section>
<p class="note">GPT sees the three camera images and robot state. The overhead grid uses camera calibration and a known table height; no exact object poses or goal regions are supplied. GPT chooses Cartesian targets; a generic feedback loop converts each target to OSC_POSE commands. Success comes from LIBERO’s task predicate. Public model notes and complete API outputs are shown; encrypted reasoning is retained as returned, and private reasoning text is unavailable.</p>
<div class="layout"><aside class="panel"><h2>Every model decision</h2><div id="calls" class="calls"></div></aside><section class="panel"><div class="inline"><h2 id="calltitle"></h2><div><button id="prev" aria-label="Previous call">←</button> <button id="next" aria-label="Next call">→</button></div></div><p id="callstats"></p><button id="playcall">Play from this call</button><h3>What GPT observed</h3><div class="photos" id="before"></div><h3>Prediction / public action note</h3><p id="note"></p><h3>Tool parameters</h3><pre id="params"></pre><h3>Tool result</h3><pre id="result"></pre><h3>After the action</h3><div class="photos" id="after"></div><details><summary>Token usage for this call</summary><pre id="usage"></pre></details><details><summary>Robot state before and after</summary><pre id="statebefore"></pre><pre id="stateafter"></pre></details><details><summary>Every executed motor action</summary><p>Action = normalized Δx, Δy, Δz, Δrotation x/y/z, gripper. Position scale 0.05 m; rotation scale 0.5 rad. Gripper −1 opens, +1 closes.</p><div class="scroll"><table><thead><tr><th>Step</th><th>Action (7D)</th><th>Actual grasp-center XYZ</th><th>Reward</th><th>Done</th></tr></thead><tbody id="actionrows"></tbody></table></div><details><summary>Full motor action JSON</summary><pre id="actionjson"></pre></details></details><details><summary>Assistant message</summary><pre id="prediction"></pre></details><details><summary>Exact API request / response, timing, retries, usage</summary><p>Image references use $blob hashes. Original PNGs are in wire/api/blobs/. Authorization headers are never recorded.</p><pre id="raw"></pre></details></section></div>
<section class="panel"><h2>Inspect every simulation frame</h2><div class="controls"><label id="frametext" for="frame">Frame</label><input type="range" id="frame" min="0" value="0" step="1"></div><div class="photos" id="allframes"></div><details><summary>Frame timestamp and robot state</summary><pre id="framestate"></pre></details></section>
<section class="panel"><h2>Saved evidence</h2><div class="files">{"".join(f'<a href="{name}" download>{name}</a>' for name in ["result.json", "decisions.jsonl", "actions.jsonl", "observations.jsonl", "transcript.json", "wire/api/episode/calls.jsonl", "sim_states.npz", "model.xml", "warmup.jsonl", "warmup_states.npz", "system_prompt.txt", "tools.json", "runner_source.py"])}</div><details><summary>Full run metadata</summary><pre>{html.escape(json.dumps(meta, indent=2))}</pre></details><p>Reasoning tokens are included in output tokens; cached tokens are included in input tokens. Wall time includes environment setup and cleanup. Physics/render/log time includes saving images. These are interactive demonstrations with an 800-step budget, not standard LIBERO benchmark scores.</p></section>
</main><script type="application/json" id="data">{payload}</script><script>{SCRIPT}</script></html>'''
    page = page.replace('<p class="note">GPT sees', finishing + '<p class="note">GPT sees')
    (folder / "report.html").write_text(page)
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2] / "artifacts/astra_libero"
    )
    parser.add_argument("--overwrite-video", action="store_true")
    args = parser.parse_args()
    records = []
    continuations = []
    for result in sorted(args.root.glob("task_*/result.json")):
        meta = build_task(result.parent, args.overwrite_video)
        records.append((result.parent.name, meta))
        for extra in sorted((result.parent / "post_success").glob("task_*/result.json")):
            extra_meta = build_task(extra.parent, args.overwrite_video)
            continuations.append((extra.parent.relative_to(args.root).as_posix(), extra_meta))
            (extra.parent.parent / "index.html").write_text(
                f'<!doctype html><html lang="en"><meta charset="utf-8"><title>Post-success finishing phase</title><style>{STYLE}</style><main><h1>Post-success finishing phase</h1><p>This continues an already successful benchmark episode to release the object and withdraw.</p><a href="{extra.parent.name}/report.html">Open the finishing-phase report</a><p><a href="../report.html">Return to the original episode</a></p></main></html>'
            )
    model_label = html.escape(" / ".join(sorted({m["model"] for _, m in records})))
    all_records = records + continuations
    total_tokens = sum(m.get("usage", {}).get("total_tokens", 0) for _, m in all_records)
    total_wall = sum(m.get("total_wall_time_s", 0) for _, m in all_records)
    success = sum(bool(m.get("success")) for _, m in records)
    with (args.root / "per_call_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task_id",
                "attempt",
                "phase",
                "decision",
                "tool",
                "api_latency_s",
                "physics_render_log_s",
                "timing_reconstructed",
                "motor_steps",
                "simulation_s",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "reasoning_tokens",
                "cached_input_tokens",
            ],
        )
        writer.writeheader()
        for name, meta in all_records:
            wire = read_lines(args.root / name / "wire/api/episode/calls.jsonl")
            for d in read_lines(args.root / name / "decisions.jsonl"):
                usage = [
                    (w.get("response") or {}).get("usage") or {}
                    for w in wire
                    if w["call"] == d["decision"]
                ]
                steps = d["step_end"] - d["step_start"]
                row = {
                    "task_id": meta["task_id"],
                    "attempt": meta["attempt"],
                    "phase": meta.get("phase", "benchmark"),
                    "decision": d["decision"],
                    "tool": d.get("tool", {}).get("name", "text"),
                    "api_latency_s": d["api_latency_s"],
                    "physics_render_log_s": d.get("simulation_time_s", 0),
                    "motor_steps": steps,
                    "simulation_s": steps / 20,
                }
                row.update(
                    {
                        key: sum(u.get(key, 0) for u in usage)
                        for key in ["input_tokens", "output_tokens", "total_tokens"]
                    }
                )
                row["reasoning_tokens"] = sum(
                    u.get("output_tokens_details", {}).get("reasoning_tokens", 0) for u in usage
                )
                row["cached_input_tokens"] = sum(
                    u.get("input_tokens_details", {}).get("cached_tokens", 0) for u in usage
                )
                row["timing_reconstructed"] = d.get("timing_reconstructed", False)
                writer.writerow(row)
    display_records = sorted(records, key=lambda item: (not item[1].get("success", False), item[0]))
    cards = "".join(
        f'''<a class="panel taskcard" href="{name}/report.html"><div class="inline"><h2>{html.escape(m["instruction"])}</h2><b class="{"ok" if m.get("success") else "fail"}">{"Success" if m.get("success") else "Unsuccessful"}</b></div><p>Task {m["task_id"]} · attempt {m["attempt"]} · {m.get("decisions", 0)} GPT decisions · {m.get("control_steps", 0)} motor steps · {m.get("total_wall_time_s", 0):.1f} s wall · {m.get("usage", {}).get("total_tokens", 0):,} tokens</p><img src="{name}/frames/000000_agentview.png" alt="Initial Panda robot scene" style="width:190px;border-radius:8px"></a>'''
        for name, m in display_records
    )
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{model_label} controls LIBERO</title><style>{STYLE}</style><main><p>INSPECT ROBOTS / REAL PHYSICS DEMO</p><h1>{model_label} controls a Panda in LIBERO</h1><p>Explore the videos, camera observations, model tool calls, motor actions, timing, and token usage. Every saved attempt is listed, including failures.</p><div class="metrics"><div class="metric"><span>Successful attempts</span><b>{success} / {len(records)}</b></div><div class="metric"><span>Total tokens</span><b>{total_tokens:,}</b></div><div class="metric"><span>Summed run wall time</span><b>{total_wall:.1f} s</b></div></div><section class="panel"><h2>How the arm is controlled</h2><p>Camera images + robot state → {model_label} chooses a pose and gripper command → feedback loop produces 7D OSC_POSE actions → MuJoCo computes motor torques and object contacts → new images return to GPT.</p><p>The model receives RGB views and a calibrated table grid. It receives no exact object locations or goal-region coordinates. A real simulated Panda arm and actuated fingers perform the motion. LIBERO’s own success check determines the result.</p><p>Runs use an isolated uv environment with CPU physics and OSMesa rendering. An additional overhead camera and an 800-step budget make these interactive demos rather than standard benchmark evaluations.</p></section>{cards}<section class="panel"><h2>Reproduce and inspect</h2><div class="files"><a href="README.md">README / method</a><a href="environment.freeze.txt">Environment versions</a><a href="summary.json">Machine-readable summary</a><a href="per_call_metrics.csv">Per-call timing / tokens (CSV)</a><a href="source/run.py">Runner source</a></div><p>Each attempt includes its exact runner source, prompts, raw API evidence, all action frames, a MuJoCo XML model, replay states, and warmup logs. Token totals come from provider responses; reasoning tokens are a subset of output tokens.</p></section></main></html>"""
    (args.root / "index.html").write_text(page)
    if continuations:
        page = page.replace(
            '<section class="panel"><h2>Reproduce and inspect',
            '<p>Overall timing and token totals include the separately logged post-success finishing phase. The success count covers only the three original benchmark attempts.</p><section class="panel"><h2>Reproduce and inspect',
        )
        (args.root / "index.html").write_text(page)
    (args.root / "summary.json").write_text(
        json.dumps(
            {
                "successful_attempts": success,
                "attempts": len(records),
                "total_tokens": total_tokens,
                "summed_run_wall_time_s": total_wall,
                "runs": [dict(folder=n, **m) for n, m in records],
                "continuations": [dict(folder=n, **m) for n, m in continuations],
            },
            indent=2,
        )
    )
    (args.root / "source").mkdir(exist_ok=True)
    for source in Path(__file__).parent.iterdir():
        if source.is_file() and source.suffix in {".py", ".sh", ".md"}:
            shutil.copyfile(source, args.root / "source" / source.name)
    outcome = "\n\n## Recorded outcomes\n\n| Task | Result | GPT calls | Motor steps | Wall seconds | Tokens |\n|---|---|---:|---:|---:|---:|\n"
    for _, m in records:
        outcome += f"| {m['instruction']} | {'Success' if m.get('success') else 'Unsuccessful'} | {m.get('decisions', 0)} | {m.get('control_steps', 0)} | {m.get('total_wall_time_s', 0):.2f} | {m.get('usage', {}).get('total_tokens', 0):,} |\n"
    for _, m in continuations:
        outcome += f"\nPost-success release and withdrawal continuation: {'verified' if m.get('success') else 'unsuccessful'}, {m.get('decisions', 0)} GPT calls, {m.get('total_wall_time_s', 0):.2f} seconds, {m.get('usage', {}).get('total_tokens', 0):,} tokens. Restored from the exact final simulator state; logged separately.\n"
    (args.root / "README.md").write_text(
        (Path(__file__).parent / "README.md").read_text() + outcome
    )
    print(f"Built {len(records)} reports: {success} successful attempts; {total_tokens} tokens.")


if __name__ == "__main__":
    main()
