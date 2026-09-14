# Offline report embeds its videos, styles, and interaction code.
# ruff: noqa: E501
"""Build a portable Astra/Luna comparison from saved LIBERO evidence."""

import argparse
import base64
import csv
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
from build_report import read_lines
from render_captioned_video import pricing_for, request_metrics


def collect(folder):
    result = json.loads((folder / "result.json").read_text())
    decisions = read_lines(folder / "decisions.jsonl")
    wire = read_lines(folder / "wire/api/episode/calls.jsonl")
    calls = []
    for d in decisions:
        metrics = [request_metrics(w) for w in wire if w["call"] == d["decision"]]
        usage = {key: sum(m[key] for m in metrics) for key in metrics[0]}
        calls.append(
            {
                "call": d["decision"] + 1,
                "phase": result.get("phase", "main episode"),
                "tool": d.get("tool"),
                "result": d.get("result"),
                "api_latency_s": d["api_latency_s"],
                "physics_render_log_s": d.get("simulation_time_s", 0),
                "frame_before": d["frame_before"],
                "frame_after": d["frame_after"],
                "state_before": d["state_before"],
                "state_after": d["state_after"],
                **usage,
            }
        )
    latencies = [c["api_latency_s"] for c in calls]
    return {
        "model": result["model"],
        "task_id": result["task_id"],
        "goal": result["instruction"],
        "success": result["success"],
        "phase": result.get("phase", "main episode"),
        "termination": result["termination_reason"],
        "calls_count": len(calls),
        "motor_steps": result["control_steps"],
        "wall_s": result["total_wall_time_s"],
        "api_s": sum(latencies),
        "api_mean_s": statistics.mean(latencies),
        "api_median_s": statistics.median(latencies),
        "input_tokens": result["usage"]["input_tokens"],
        "output_tokens": result["usage"]["output_tokens"],
        "total_tokens": result["usage"]["total_tokens"],
        "estimated_cost_usd": sum(c["estimated_cost_usd"] for c in calls),
        "calls": calls,
    }


def audit_initial(astra, luna):
    same_files = {}
    for name in [
        "system_prompt.txt",
        "tools.json",
        *[f"frames/000000_{cam}.png" for cam in ("agentview", "robot0_eye_in_hand", "birdview")],
    ]:
        same_files[name] = (
            hashlib.sha256((astra / name).read_bytes()).hexdigest()
            == hashlib.sha256((luna / name).read_bytes()).hexdigest()
        )
    a = np.load(astra / "sim_states.npz")["states"][0]
    b = np.load(luna / "sim_states.npz")["states"][0]
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-10)
    requests = []
    for folder in (astra, luna):
        request = dict(read_lines(folder / "wire/api/episode/calls.jsonl")[0]["request"])
        request.pop("model")
        requests.append(request)
    assert all(same_files.values()) and requests[0] == requests[1]
    return {
        "identical_files": same_files,
        "initial_sim_state_max_abs_difference": float(np.max(np.abs(a - b))),
        "first_api_request_identical_except_model": requests[0] == requests[1],
    }


PAGE = r"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LIBERO: GPT-6 Astra vs GPT-5.6 Luna</title>
<style>
:root{color-scheme:dark;font:16px/1.5 system-ui;background:#0d131f;color:#e8eef6}body{margin:0}main{max-width:1500px;margin:auto;padding:24px}h1{font-size:30px;margin:8px 0}h2{font-size:22px}.muted{color:#abbace}.panel{background:#172235;border:1px solid #35455d;border-radius:12px;padding:20px;margin:18px 0}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;white-space:nowrap}th,td{padding:10px;text-align:left;border-bottom:1px solid #35455d}th{color:#a4bddd}.ok{color:#86e0b0}.fail{color:#ffb296}.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}video{width:100%;background:black;border-radius:8px}button,select{background:#263e5a;color:white;border:1px solid #6683a7;border-radius:6px;padding:9px;font:inherit;margin:4px}button{cursor:pointer}a{color:#a2c9ff}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap}label{display:inline-flex;gap:8px;align-items:center}.stats{color:#bccce0;font-size:14px}summary{cursor:pointer;font-weight:600}@media(max-width:850px){.pair{grid-template-columns:1fr}main{padding:12px}h1{font-size:25px}}
</style><main>
<h1>LIBERO: GPT-6 Astra vs GPT-5.6 Luna</h1>
<p class="muted">Three matched tasks · same initial states, prompts, cameras, robot state, tools, and medium reasoning · one attempt per model per task</p>
<section class="panel"><h2>Results and cost</h2>
<label><input id="finishing" type="checkbox"> Include separately recorded finishing phases in usage totals</label>
<div class="scroll"><table><thead><tr><th>Model</th><th>Tasks succeeded</th><th>Calls</th><th>Motor steps</th><th>Wall time</th><th>API time</th><th>Mean API latency</th><th>Input tokens</th><th>Generated tokens</th><th>Estimated cost</th></tr></thead><tbody id="totals"></tbody></table></div>
<p class="muted">Success counts always refer to the three main episodes. Finishing phases release a held object after native task success; they are not additional benchmark attempts.</p>
<div class="scroll"><table><thead><tr><th>Task</th><th>Model</th><th>Result</th><th>Calls</th><th>Steps</th><th>Wall time</th><th>Mean / median API latency</th><th>Generated tokens</th><th>Estimated cost</th></tr></thead><tbody id="tasks"></tbody></table></div></section>
<section class="panel"><div class="controls"><label>Task <select id="task"></select></label><button id="both">Restart and play both</button><button id="pause">Pause both</button></div>
<p class="muted">Playback uses simulation time and omits API waiting. Videos include all call parameters and notes, measured yaw, latency, tokens, cost, and a final result card. Any finishing continuation is labeled in the video.</p>
<div class="pair"><article><h2>GPT-6 Astra</h2><video id="astra" controls preload="metadata"></video><p id="astra-stats" class="stats"></p><a id="astra-download" download>Download Astra MP4</a><label>Call <select id="astra-call"></select></label><details><summary>Selected call: parameters, result, timing, usage, robot state</summary><pre id="astra-details"></pre></details></article>
<article><h2>GPT-5.6 Luna</h2><video id="luna" controls preload="metadata"></video><p id="luna-stats" class="stats"></p><a id="luna-download" download>Download Luna MP4</a><label>Call <select id="luna-call"></select></label><details><summary>Selected call: parameters, result, timing, usage, robot state</summary><pre id="luna-details"></pre></details></article></div></section>
<section class="panel"><details><summary>Comparison protocol and verification</summary>
<p>LIBERO Goal tasks 5, 6, and 8; initial state index 0; seed 7; 10 settling steps; 20 Hz OSC_POSE; maximum 40 model calls and 800 motor steps. Models see external, wrist, and calibrated overhead images plus robot proprioception. Exact object poses and hidden goal regions are not supplied.</p>
<p>Initial simulator states, first camera PNGs, system prompts, tool schemas, and the first API request (apart from model name) are verified against Astra. The later runner includes a redundant benchmark_success field alongside success in move results; the controller and main stopping criterion are unchanged.</p>
<p>These are individual interactive demonstrations, not an official benchmark score or a statistical model ranking. Latency and wall time can vary with API service and host load. Wall time includes setup, simulation, rendering, and logging.</p>
<p>Costs use published standard token rates and recorded cache reads/writes; generated tokens include reasoning once. Taxes and account-specific discounts are excluded. <a href="https://developers.openai.com/api/docs/models/gpt-6-astra">Astra pricing</a> · <a href="https://developers.openai.com/api/docs/models/gpt-5.6-luna">Luna pricing</a></p>
<pre id="audit"></pre></details></section>
</main><script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id),fmt=n=>Number(n).toLocaleString(),secs=n=>Number(n).toFixed(2)+' s',cost=n=>'$'+Number(n).toFixed(4),esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const sides=['astra','luna'],models={'astra':'gpt-6-astra','luna':'gpt-5.6-luna'};
function phases(run){return $('finishing').checked?[run,...run.finishing]:[run]}
function aggregate(list){const out={calls_count:0,motor_steps:0,wall_s:0,api_s:0,input_tokens:0,output_tokens:0,estimated_cost_usd:0};for(const r of list)for(const k in out)out[k]+=r[k];out.mean=out.api_s/out.calls_count;return out}
function renderTables(){
$('totals').innerHTML=sides.map(side=>{const runs=data.tasks.map(t=>t[side]),a=aggregate(runs.flatMap(phases));return `<tr><td>${models[side]}</td><td>${runs.filter(r=>r.success).length} / 3</td><td>${a.calls_count}</td><td>${a.motor_steps}</td><td>${secs(a.wall_s)}</td><td>${secs(a.api_s)}</td><td>${secs(a.mean)}</td><td>${fmt(a.input_tokens)}</td><td>${fmt(a.output_tokens)}</td><td>${cost(a.estimated_cost_usd)}</td></tr>`}).join('');
$('tasks').innerHTML=data.tasks.flatMap(t=>sides.map(side=>{const r=t[side],a=aggregate(phases(r)),lat=phases(r).flatMap(p=>p.calls.map(c=>c.api_latency_s)).sort((a,b)=>a-b),mid=Math.floor(lat.length/2),median=lat.length%2?lat[mid]:(lat[mid-1]+lat[mid])/2;return `<tr><td>${esc(t.goal)}</td><td>${models[side]}</td><td class="${r.success?'ok':'fail'}">${r.success?'SUCCESS':'FAILURE'}</td><td>${a.calls_count}</td><td>${a.motor_steps}</td><td>${secs(a.wall_s)}</td><td>${secs(a.mean)} / ${secs(median)}</td><td>${fmt(a.output_tokens)}</td><td>${cost(a.estimated_cost_usd)}</td></tr>`})).join('')}
function showCall(side,seek=false){const r=data.tasks[Number($('task').value)][side],i=Number($(side+'-call').value),c=[...r.calls,...r.finishing.flatMap(p=>p.calls)][i];$(side+'-details').textContent=JSON.stringify(c,null,2);if(seek)$(side).currentTime=r.video_calls[i].video_frames[0]/20}
function selectTask(){const t=data.tasks[Number($('task').value)];for(const side of sides){const r=t[side],v=$(side);v.pause();v.src=r.video_data;v.load();v.ontimeupdate=()=>{const f=Math.floor(v.currentTime*20+1e-6),i=r.video_calls.findIndex(c=>c.video_frames[0]<=f&&f<=c.video_frames[1]);if(i>=0&&Number($(side+'-call').value)!==i){$(side+'-call').value=String(i);showCall(side)}};const d=$(side+'-download');d.href=r.video_data;d.download=r.video_filename.startsWith(models[side]+'__')?r.video_filename:models[side]+'__'+r.video_filename;$(side+'-stats').textContent=`${r.success?'SUCCESS':'FAILURE'} · Main: ${r.calls_count} calls · ${r.motor_steps} steps · ${secs(r.wall_s)} wall · ${cost(r.estimated_cost_usd)} estimated${r.finishing.length?' · Video includes separately logged finishing phase':''}`;$(side+'-call').innerHTML=[...r.calls,...r.finishing.flatMap(p=>p.calls)].map((c,i)=>`<option value="${i}">${c.phase==='main episode'?'Main':'Finishing'} ${c.call}: ${esc(c.tool?.name||'no tool')}</option>`).join('');showCall(side)}}
$('task').innerHTML=data.tasks.map((t,i)=>`<option value="${i}">${esc(t.goal)}</option>`).join('');$('task').onchange=selectTask;$('finishing').onchange=renderTables;$('audit').textContent=JSON.stringify(data.audit,null,2);
for(const side of sides)$(side+'-call').onchange=()=>showCall(side,true);
$('both').onclick=()=>{for(const side of sides){$(side).currentTime=0;$(side).play().catch(()=>{})}};$('pause').onclick=()=>sides.forEach(side=>$(side).pause());renderTables();selectTask();
</script></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2] / "artifacts"
    parser.add_argument("--astra", type=Path, default=root / "astra_libero")
    parser.add_argument("--luna", type=Path, default=root / "luna_libero")
    parser.add_argument("--output", type=Path, default=root / "libero_model_comparison")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = {
        "tasks": [],
        "audit": {},
        "pricing": {m: pricing_for(m) for m in ("gpt-6-astra", "gpt-5.6-luna")},
    }
    csv_rows = []
    for task in (5, 6, 8):
        folders = {
            "astra": args.astra / f"task_{task:02d}_attempt_00",
            "luna": args.luna / f"task_{task:02d}_attempt_00",
        }
        data["audit"][str(task)] = audit_initial(folders["astra"], folders["luna"])
        item = {"task_id": task}
        for side, folder in folders.items():
            r = collect(folder)
            r["finishing"] = [
                collect(p.parent)
                for p in sorted((folder / "post_success").glob("task_*/result.json"))
            ]
            captioned = json.loads((folder / "captioned_video.json").read_text())
            r["video_filename"] = captioned["video"]
            r["video_path"] = str((folder / captioned["video"]).resolve())
            r["video_calls"] = captioned["calls"]
            for phase in [r, *r["finishing"]]:
                csv_rows.append(
                    {
                        k: phase[k]
                        for k in (
                            "model",
                            "task_id",
                            "goal",
                            "phase",
                            "success",
                            "calls_count",
                            "motor_steps",
                            "wall_s",
                            "api_s",
                            "api_mean_s",
                            "api_median_s",
                            "input_tokens",
                            "output_tokens",
                            "total_tokens",
                            "estimated_cost_usd",
                        )
                    }
                )
            item[side] = r
            item["goal"] = r["goal"]
        data["tasks"].append(item)
    (args.output / "comparison.json").write_text(json.dumps(data, indent=2) + "\n")
    with (args.output / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    for task in data["tasks"]:
        for side in ("astra", "luna"):
            r = task[side]
            r["video_data"] = (
                "data:video/mp4;base64,"
                + base64.b64encode(Path(r["video_path"]).read_bytes()).decode()
            )
    (args.output / "index.html").write_text(
        PAGE.replace("__DATA__", json.dumps(data).replace("<", "\\u003c"))
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
