"""Static HTML explorer for scripts/patch/ar_patch_attention.py --save-media outputs.

Reads existing shards/media only (no model run). Writes <root>/report.html and one
time-averaged pixel-RMSE map (impact.png) per branch.
Run after summarize_ar_patch_attention.py (uses its summary.json and figures):
  python scripts/patch/build_ar_patch_importance_report.py --root outputs/patch_importance/ar_patch_importance_20261026
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.patch.patch_noise_strength import scale_name
from scripts.patch.summarize_ar_patch_attention import load_case
from scripts.patch.summarize_ar_patch_rollout import PIXEL_MAP_LIMIT, colorize

METRICS = {"rmse": ("pixel", "all_video_rmse", "全视频像素 RMSE（21 帧）"),
           "later": ("pixel", "later_frames_rmse", "后段像素 RMSE（帧 12–20）"),
           "next": ("rollout", "next_chunk_mse", "下一 chunk latent MSE")}


def r6(value):
    return float(f"{value:.6g}")


def impact_png(branch_dir, overwrite):
    target = branch_dir / "impact.png"
    if target.is_file() and not overwrite:
        return
    with np.load(branch_dir / "impact_map.npz") as archive:
        mse = archive["pixel_mse_downsample"]
    if mse.ndim != 3 or not np.isfinite(mse).all():
        raise ValueError(f"Invalid impact map: {branch_dir}")
    image = Image.fromarray(colorize(np.sqrt(mse.mean(axis=0)), PIXEL_MAP_LIMIT))
    image.resize((image.width * 2, image.height * 2), Image.Resampling.NEAREST).save(target)


def build_case(root, case_dir, summary_entry, overwrite, workers):
    protocol, by_scale, _ = load_case(case_dir)
    scales = sorted(by_scale, reverse=True)
    branch_dirs = [case_dir / f"source_{r['source_id']:03d}" / f"scale_{scale_name(s)}"
                   for s in scales for r in by_scale[s]]
    for path in [case_dir / "baseline" / "final.mp4", case_dir / "baseline_source_frame.png"] + [
            d / name for d in branch_dirs for name in ("final.mp4", "final.jpg", "impact_map.npz")]:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing media (run with --save-media): {path}")
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(lambda d: impact_png(d, overwrite), branch_dirs))
    stats = {float(k): v for k, v in summary_entry["importance_by_scale"].items()}
    return {"dir": case_dir.relative_to(root).as_posix(), "prompt_index": protocol["prompt_index"],
            "prompt": protocol["prompt"], "seed": protocol["seed"], "step": protocol["step_index"],
            "base_sigma": protocol["base_sigma"], "scales": scales, "scale_names": [scale_name(s) for s in scales],
            "latent": protocol["shape"][-2:],
            "regions": [{k: r[k] for k in ("id", "row", "col", "y0", "y1", "x0", "x1")} for r in protocol["regions"]],
            "tokens": [r["num_tokens"] for r in by_scale[scales[0]]],
            "metrics": {m: [[r6(r[group][key]) for r in by_scale[s]] for s in scales]
                        for m, (group, key, _) in METRICS.items()},
            "per_frame": [[[r6(v) for v in r["pixel"]["per_frame_rmse"]] for r in by_scale[s]] for s in scales],
            "stats": [stats[s] for s in scales],
            "attention_rho": {k: v["layer_mean"] for k, v in summary_entry["spearman"].items()},
            "attention_scale": summary_entry["attention_scale"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/patch_importance/ar_patch_importance_20261026"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = args.root
    summary = json.loads((root / "summary.json").read_text())
    entries = {(e["prompt_index"], e["seed"], e["step_index"]): e for e in summary["cases"]}
    cases = []
    for case_dir in sorted(root.glob("prompt_*_seed_*/step_*")):
        protocol = json.loads(next(case_dir.glob("importance_shard_*.json")).read_text())["protocol"]
        entry = entries[protocol["prompt_index"], protocol["seed"], protocol["step_index"]]
        cases.append(build_case(root, case_dir, entry, args.overwrite, args.workers))
        print(f"CASE {cases[-1]['dir']} ok", flush=True)
    cases.sort(key=lambda c: (c["prompt_index"], c["step"]))
    data = {"cases": cases, "metrics": {k: v[2] for k, v in METRICS.items()}, "impact_max": PIXEL_MAP_LIMIT,
            "pooled_attention_rho": summary.get("pooled_within_prompt_rank_spearman", {})}
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    (root / "report.html").write_text(TEMPLATE.replace("__DATA__", encoded), encoding="utf-8")
    print(f"REPORT {root / 'report.html'} cases={len(cases)}", flush=True)


TEMPLATE = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>AR patch importance · 噪声等级干预</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f2f6f5;color:#16373b;font:14px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1480px;margin:auto;padding:28px 24px}h1{font-size:28px;margin:6px 0}h2{font-size:19px;margin:4px 0 10px}
.tag{font-size:11px;letter-spacing:1px;color:#087f7a;font-weight:700}.intro,.note{color:#5d7477}.note{font-size:12px}
.card{background:#fff;border:1px solid #dce7e5;border-radius:12px;padding:16px;margin:16px 0}
.controls{display:flex;flex-wrap:wrap;gap:18px;align-items:center}select,button{font:inherit;padding:6px 9px;border:1px solid #bcd0cb;border-radius:7px;background:#fff;color:inherit;cursor:pointer}
button.on{background:#087f7a;color:#fff;border-color:#087f7a}.layout{display:grid;grid-template-columns:minmax(0,1.05fr) minmax(0,1fr);gap:18px}
.grid{position:relative;width:100%;aspect-ratio:104/60;background:#000;border-radius:8px;overflow:hidden;user-select:none}
.grid img{position:absolute;inset:0;width:100%;height:100%;object-fit:fill}
.cell{position:absolute;border:1px solid rgba(255,255,255,.18);cursor:pointer}.cell:hover{outline:2px solid #fff;z-index:2}
.cell.sel{outline:3px solid #38f2d8;z-index:3}#tip{position:fixed;pointer-events:none;background:#16373b;color:#fff;padding:4px 8px;border-radius:5px;font-size:12px;display:none;z-index:9}
.legend{display:flex;align-items:center;gap:8px;margin-top:8px;font-size:12px}#legend{width:260px;height:12px;border-radius:3px}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin-top:10px}th,td{border-bottom:1px solid #e3ecea;padding:5px 7px;text-align:right}th:first-child,td:first-child{text-align:left}
th{color:#5d7477;font-weight:600}tr.click{cursor:pointer}tr.click:hover,tr.cur{background:#eaf5f2}
.videos,.row2{display:grid;grid-template-columns:1fr 1fr;gap:10px}figure{margin:0}figcaption{font-size:12px;color:#5d7477;margin-bottom:3px}
video,.row2 img{width:100%;border-radius:6px;background:#000;display:block}.row2 img{image-rendering:pixelated}svg{width:100%;background:#fafcfc;border:1px solid #e3ecea;border-radius:6px}
.figs{display:grid;grid-template-columns:1fr;gap:12px}.figs img{width:100%;border:1px solid #e3ecea;border-radius:6px}a{color:#087f7a}
footer{font-size:12px;color:#5d7477;margin-top:20px}@media(max-width:1000px){.layout{grid-template-columns:1fr}}
</style></head><body><main>
<div class="tag">SELF-FORCING · CHECKPOINT 600 (generator) · CHUNK 0 · LATENT FRAME 1 · 12×16 PATCH GRID</div>
<h1>逐 patch 降低噪声等级，对完整生成结果的影响</h1>
<p class="intro">在一次去噪调用（step_index 1 或 2，从 0 编号）中，只把一个 patch 的实际噪声等级改为 σ<sub>local</sub> = σ<sub>base</sub> × scale，同时把该 patch 的 token timestep 设为 σ<sub>local</sub>×1000、flow→x₀ 换算使用同一 σ map；其余 token 与之后的所有调用保持原 schedule 与原噪声，继续生成完整 21 帧。数值是相对未干预 baseline 的变化。scale 0 表示该 patch 直接取上一步模型预测的 x₀（不是真值）。</p>
<div class="card controls">
<label>场景 <select id="case"></select></label>
<span>局部噪声 <span id="scales"></span></span>
<label>指标 <select id="metric"></select></label>
<label><input type="checkbox" id="shared" checked> 4 档共用色标</label>
</div>
<div class="layout">
<div class="card"><h2 id="ctitle"></h2><p class="note" id="cinfo"></p>
<div class="grid" id="grid"><img id="frame" alt="baseline"><div id="cells"></div></div>
<div class="legend"><span>0</span><canvas id="legend" width="260" height="12"></canvas><span id="lmax"></span><span class="note">底图为 baseline 解码帧 4（latent 帧 1）；色块 = 干预该 patch 后的结果变化</span></div>
<table id="stats"></table></div>
<div class="card"><h2 id="ptitle">点击左图选择 patch</h2>
<div class="videos"><figure><figcaption>Baseline</figcaption><video id="vb" muted loop playsinline controls></video></figure>
<figure><figcaption id="vcap">干预后</figcaption><video id="vi" muted loop playsinline controls></video></figure></div>
<p><button id="sync">从头同步播放</button> <span class="note">两段视频使用相同初始噪声与后续噪声，仅该 patch 在一次调用中被改动。</span></p>
<div class="row2"><figure><figcaption>像素 RMSE 空间分布（21 帧时间平均，色标 0–<span id="imax"></span>）</figcaption><img id="impact" alt=""></figure>
<figure><figcaption>逐帧像素 RMSE（该 patch，4 档）</figcaption><svg id="curve" viewBox="0 0 440 210"></svg></figure></div>
<table id="ptable"></table></div>
</div>
<div class="card"><h2>跨场景汇总</h2><p class="note">点击行切换场景。均值为 192 个 patch 的全视频像素 RMSE；ρ 为与 scale 0 排序的 Spearman；attention ρ 为 scale <span id="ascale"></span> 下 importance 与 30 层平均 attention（received_all）的 Spearman。</p><table id="overview"></table></div>
<div class="card"><h2>静态图（当前场景）</h2><div class="figs"><figure><figcaption>4 档同色标热图</figcaption><img id="fig1" alt=""></figure>
<figure><figcaption>importance vs attention</figcaption><img id="fig2" alt=""></figure></div>
<p><a href="per_layer_spearman.png" target="_blank">逐层 attention 相关性</a> · <a href="summary.json" target="_blank">summary.json</a></p></div>
<footer><p>限制：每个 prompt 只有 1 个 seed；patch 含 6/8/9/12 个 token，面积与输入能量不一致；192 次干预相互独立，数值不是可相加的因果贡献；只在 chunk 0 的 latent 帧 1 干预；RMSE 衡量结果变化，不衡量生成质量。</p></footer>
</main><div id="tip"></div>
<script id="data" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);
const ST=[[0,0,4],[59,15,112],[140,41,129],[222,73,104],[254,159,109],[252,253,191]];
const LC=['#2a9d8f','#e9c46a','#f4a261','#c2410c'];
function col(t){t=Math.max(0,Math.min(1,t))*(ST.length-1);const i=Math.min(Math.floor(t),ST.length-2),f=t-i;return'rgb('+ST[i].map((v,k)=>Math.round(v+(ST[i+1][k]-v)*f))+')'}
const pad=n=>String(n).padStart(3,'0'),fmt=v=>v==null?'–':(+v).toPrecision(4);
let S={c:0,s:0,m:'rmse',sel:null};const C=()=>D.cases[S.c];
function bdir(c,sid,si){return`${c.dir}/source_${pad(sid)}/scale_${c.scale_names[si]}`}
function init(){
 D.cases.forEach((c,i)=>$('case').add(new Option(`#${c.prompt_index} · step_index ${c.step} · ${c.prompt.slice(0,40)}`,i)));
 Object.entries(D.metrics).forEach(([k,v])=>$('metric').add(new Option(v,k)));
 $('case').onchange=e=>{S.c=+e.target.value;render(true)};$('metric').onchange=e=>{S.m=e.target.value;render()};$('shared').onchange=()=>render();
 $('sync').onclick=()=>{for(const v of[$('vb'),$('vi')]){v.currentTime=0;v.play()}};
 $('imax').textContent=D.impact_max;$('ascale').textContent=D.cases[0].attention_scale;
 const ov=$('overview'),sc=D.cases[0].scales;
 ov.innerHTML='<tr><th>场景</th><th>step_index</th><th>σ base</th>'+sc.map(s=>`<th>×${s} 均值</th><th>ρ</th>`).join('')+'<th>attention ρ</th></tr>'+D.cases.map((c,i)=>`<tr class="click" data-i="${i}"><td>#${c.prompt_index} ${c.prompt.slice(0,34)}</td><td>${c.step}</td><td>${fmt(c.base_sigma)}</td>`+c.stats.map(x=>`<td>${fmt(x.mean_rmse)}</td><td>${x.spearman_vs_lowest_scale.toFixed(2)}</td>`).join('')+`<td>${c.attention_rho.received_all.toFixed(2)}</td></tr>`).join('');
 ov.querySelectorAll('tr.click').forEach(r=>r.onclick=()=>{S.c=+r.dataset.i;$('case').value=S.c;render(true);scrollTo({top:0,behavior:'smooth'})});
 S.s=D.cases[0].scales.length-1;render(true)}
function render(newCase){
 const c=C();if(newCase){S.sel=null;$('frame').src=c.dir+'/baseline_source_frame.png';$('vb').src=c.dir+'/baseline/final.mp4';
  $('fig1').src=c.dir+'/importance_by_scale.png';$('fig2').src=c.dir+'/importance_vs_attention.png';
  $('scales').innerHTML=c.scales.map((s,i)=>`<button data-i="${i}">×${s}</button>`).join(' ');
  $('scales').querySelectorAll('button').forEach(b=>b.onclick=()=>{S.s=+b.dataset.i;render();if(S.sel!=null)select(S.sel)});
  document.querySelectorAll('#overview tr.click').forEach(r=>r.classList.toggle('cur',+r.dataset.i===S.c));
  $('ptitle').textContent='点击左图选择 patch';$('vi').removeAttribute('src');$('impact').removeAttribute('src');$('curve').innerHTML='';$('ptable').innerHTML=''}
 $('scales').querySelectorAll('button').forEach(b=>b.classList.toggle('on',+b.dataset.i===S.s));
 const vals=c.metrics[S.m][S.s],vmax=Math.max(...($('shared').checked?c.metrics[S.m].flat():vals))||1;
 $('ctitle').textContent=`#${c.prompt_index} · step_index ${c.step} · σ_local = ${fmt(c.base_sigma)} × ${c.scales[S.s]} = ${fmt(c.base_sigma*c.scales[S.s])}`;
 $('cinfo').textContent=`“${c.prompt}” · seed ${c.seed} · 当前指标：${D.metrics[S.m]}`;
 const [H,W]=c.latent,box=$('cells');box.innerHTML='';
 c.regions.forEach(r=>{const d=document.createElement('div');d.className='cell'+(r.id===S.sel?' sel':'');
  Object.assign(d.style,{left:r.x0/W*100+'%',top:r.y0/H*100+'%',width:(r.x1-r.x0)/W*100+'%',height:(r.y1-r.y0)/H*100+'%',background:col(vals[r.id]/vmax),opacity:.72});
  d.onmousemove=e=>{const t=$('tip');t.style.display='block';t.style.left=e.clientX+12+'px';t.style.top=e.clientY+12+'px';t.textContent=`patch ${pad(r.id)} (行${r.row},列${r.col}, ${c.tokens[r.id]} tokens): ${fmt(vals[r.id])}`};
  d.onmouseleave=()=>$('tip').style.display='none';d.onclick=()=>select(r.id);box.appendChild(d)});
 const g=$('legend').getContext('2d');for(let x=0;x<260;x++){g.fillStyle=col(x/259);g.fillRect(x,0,1,12)}$('lmax').textContent=fmt(vmax);
 $('stats').innerHTML='<tr><th>scale</th><th>σ local</th><th>均值</th><th>中位数</th><th>最大</th><th>Top-5 patch</th><th>ρ vs ×0</th></tr>'+c.stats.map((x,i)=>`<tr class="${i===S.s?'cur':''}"><td>×${c.scales[i]}</td><td>${fmt(x.local_sigma)}</td><td>${fmt(x.mean_rmse)}</td><td>${fmt(x.median_rmse)}</td><td>${fmt(x.max_rmse)}</td><td>${x.top_source_ids.map(pad).join(' ')}</td><td>${x.spearman_vs_lowest_scale.toFixed(2)}</td></tr>`).join('')+'<tr><td colspan="7" class="note">本表统计量固定为全视频像素 RMSE。</td></tr>'}
function select(sid){
 const c=C();S.sel=sid;document.querySelectorAll('.cell').forEach((d,i)=>d.classList.toggle('sel',c.regions[i].id===sid));
 const r=c.regions[sid],dir=bdir(c,sid,S.s);
 $('ptitle').textContent=`patch ${pad(sid)} · 行 ${r.row} 列 ${r.col} · ${c.tokens[sid]} tokens · latent y${r.y0}–${r.y1}, x${r.x0}–${r.x1}`;
 $('vcap').textContent=`干预后 · ×${c.scales[S.s]}（σ_local=${fmt(c.base_sigma*c.scales[S.s])}）`;
 $('vi').src=dir+'/final.mp4';$('impact').src=dir+'/impact.png';
 for(const v of[$('vb'),$('vi')]){v.currentTime=0;v.play().catch(()=>{})}
 const ser=c.per_frame.map(s=>s[sid]),ym=Math.max(...ser.flat())||1,n=ser[0].length,X=i=>40+i*(390/(n-1)),Y=v=>185-v/ym*165;
 let s=`<line x1="40" y1="185" x2="430" y2="185" stroke="#9bb"/><line x1="40" y1="20" x2="40" y2="185" stroke="#9bb"/><text x="4" y="24" font-size="10" fill="#577">${fmt(ym)}</text><text x="30" y="200" font-size="10" fill="#577">帧 0</text><text x="410" y="200" font-size="10" fill="#577">帧 ${n-1}</text>`;
 ser.forEach((v,i)=>{s+=`<polyline fill="none" stroke="${LC[i%4]}" stroke-width="${i===S.s?2.6:1.3}" points="${v.map((y,j)=>X(j)+','+Y(y)).join(' ')}"/><text x="${60+i*90}" y="14" font-size="11" fill="${LC[i%4]}">×${c.scales[i]}</text>`});
 $('curve').innerHTML=s;
 const rank=(arr,v)=>1+arr.filter(x=>x>v).length;
 $('ptable').innerHTML='<tr><th>scale</th><th>σ local</th>'+Object.values(D.metrics).map(m=>`<th>${m}</th>`).join('')+'<th>RMSE 排名 /192</th><th>文件</th></tr>'+c.scales.map((x,i)=>`<tr class="${i===S.s?'cur':''}"><td>×${x}</td><td>${fmt(c.base_sigma*x)}</td>`+Object.keys(D.metrics).map(m=>`<td>${fmt(c.metrics[m][i][sid])}</td>`).join('')+`<td>${rank(c.metrics.rmse[i],c.metrics.rmse[i][sid])}</td><td><a href="${bdir(c,sid,i)}/final.mp4" target="_blank">mp4</a> · <a href="${bdir(c,sid,i)}/final.jpg" target="_blank">末帧</a></td></tr>`).join('')}
init();
</script></body></html>"""


if __name__ == "__main__":
    main()
