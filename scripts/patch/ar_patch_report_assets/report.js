"use strict";
(() => {
  const data = JSON.parse(document.getElementById("report-data").textContent);
  const matched = data.protocol.intervention_mode === "matched";
  const pilot = data.report_context && data.report_context.phase === "pilot";
  const el = id => document.getElementById(id);
  const palette = [[236,245,244],[160,210,202],[65,160,153],[14,103,115],[247,178,78],[203,68,41]];
  const state = {step:data.steps.includes(1)?1:data.steps[0], scale:data.scales.includes(0)?0:data.scales[0], source:Math.min(95,data.regions.length-1), metric:"pixel_rmse", frame:data.display.frames-1, chunk:0, inspectStep:1, inspectFrame:1, latentFrame:1, playing:false};
  if(!data.branches[`${state.step}:${state.source}:${Number(state.scale)}`]) {
    const available=data.regions.find(region=>data.branches[`${state.step}:${region.id}:${Number(state.scale)}`]);
    if(available)state.source=available.id;
  }
  const images = new Map();
  const baselineUrls = new Set(Object.values(data.baseline.media).map(item=>item.url));
  let renderVersion = 0, traceVersion = 0, mapVersion = 0, selectionVersion = 0, mediaTimer = null, currentFrames = null, animation = null, lastFrameTime = 0;
  const scaleKey = value => String(Number(value));
  const key = (step,source,scale) => `${step}:${source}:${scaleKey(scale)}`;
  const current = () => data.branches[key(state.step,state.source,state.scale)] || null;
  const fmt = value => value == null ? "—" : Number(value).toPrecision(4);
  const esc = value => String(value).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const color = value => {
    const position = Math.max(0,Math.min(1,value))*(palette.length-1), low=Math.floor(position), high=Math.min(low+1,palette.length-1), p=position-low;
    return palette[low].map((v,i)=>Math.round(v*(1-p)+palette[high][i]*p));
  };
  function options(id, values, value, label = x => String(x)) {
    el(id).innerHTML=values.map(v=>`<option value="${esc(v)}">${esc(label(v))}</option>`).join("");
    el(id).value=String(value);
  }
  function link(id, href) {
    const node=el(id); node.setAttribute("aria-disabled",href?"false":"true");node.target="_blank";node.rel="noopener";
    if(href) node.href=href; else node.removeAttribute("href");
  }
  function trimImageCache() {
    if(images.size<=24)return;
    for(const [url,entry] of images){if(entry.settled&&!baselineUrls.has(url)){images.delete(url);if(images.size<=24)break;}}
  }
  function cancelPendingBranchImages() {
    for(const [url,entry] of images){if(!entry.settled&&!baselineUrls.has(url)){images.delete(url);entry.cancel();}}
  }
  function image(descriptor, priority="auto") {
    if(!descriptor) return Promise.resolve(null);
    if(images.has(descriptor.url)) {const entry=images.get(descriptor.url);images.delete(descriptor.url);images.set(descriptor.url,entry);return entry.promise;}
    const entry={img:new Image(),settled:false,promise:null,cancel:null};
    entry.promise=new Promise(resolve=>{
      const finish=value=>{if(entry.settled)return;entry.settled=true;entry.img.onload=null;entry.img.onerror=null;resolve(value);trimImageCache();};
      entry.cancel=()=>{entry.img.onload=null;entry.img.onerror=null;entry.img.removeAttribute("src");finish(null);};
      entry.img.onload=()=>finish(entry.img);entry.img.onerror=()=>finish(null);entry.img.fetchPriority=priority;
    });
    images.set(descriptor.url,entry);entry.img.src=descriptor.url;trimImageCache();
    return entry.promise;
  }
  function empty(canvas, text="该分支尚未完成") {
    const ctx=canvas.getContext("2d");ctx.fillStyle="#edf2f0";ctx.fillRect(0,0,canvas.width,canvas.height);
    ctx.fillStyle="#748983";ctx.font="14px system-ui";ctx.textAlign="center";ctx.textBaseline="middle";ctx.fillText(text,canvas.width/2,canvas.height/2);ctx.textAlign="start";
  }
  function frame(canvas, descriptor, img, index, missing) {
    if(!descriptor || !img || index<0 || index>=descriptor.count) {empty(canvas,missing);return;}
    const ctx=canvas.getContext("2d"), x=(index%descriptor.columns)*descriptor.width,y=Math.floor(index/descriptor.columns)*descriptor.height;
    ctx.drawImage(img,x,y,descriptor.width,descriptor.height,0,0,canvas.width,canvas.height);
  }
  function regionBox(region, width=832,height=480) {
    const shape=data.protocol.shape||[1,6,16,60,104];
    if(["x0","x1","y0","y1"].every(k=>region[k]!=null)) return [region.x0/shape[shape.length-1]*width,region.y0/shape[shape.length-2]*height,(region.x1-region.x0)/shape[shape.length-1]*width,(region.y1-region.y0)/shape[shape.length-2]*height];
    return [region.col/data.cols*width,region.row/data.rows*height,width/data.cols,height/data.rows];
  }
  function entriesForCondition() {
    return data.regions.map(region=>({region,branch:data.branches[key(state.step,region.id,state.scale)]||null})).map(item=>({...item,value:item.branch?item.branch.metrics[state.metric]:null}));
  }
  function metricRange() {
    const entries=entriesForCondition(), currentMax=Math.max(0,...entries.map(x=>x.value==null?0:x.value));
    return el("color-scope").value==="global" ? data.metric_maxima[state.metric] : currentMax;
  }
  async function renderMap() {
    const version=++mapVersion,canvas=el("heatmap"),ctx=canvas.getContext("2d"),entries=entriesForCondition(),maximum=metricRange(),heatmap=el("selection-mode").value==="heatmap";
    const descriptor=data.baseline.media.final,baseImage=await image(descriptor);
    if(version!==mapVersion)return;
    ctx.fillStyle="#edf2f0";ctx.fillRect(0,0,832,480);
    if(baseImage)frame(canvas,descriptor,baseImage,Math.min(4,descriptor.count-1));
    for(const item of entries) {
      const [x,y,w,h]=regionBox(item.region),valid=item.value!=null;
      let normalized=valid&&maximum>0?item.value/maximum:0;
      if(el("log-scale").checked)normalized=Math.log1p(99*normalized)/Math.log(100);
      const rgb=valid?color(normalized):[190,197,196];
      if(heatmap || !valid){ctx.fillStyle=`rgba(${rgb.join(",")},${heatmap?.74:.25})`;ctx.fillRect(x,y,w,h);}
      ctx.strokeStyle=heatmap?"#ffffff45":"#ffffff80";ctx.lineWidth=heatmap?.6:1;ctx.strokeRect(x,y,w,h);
      if(!valid) {ctx.save();ctx.beginPath();ctx.rect(x,y,w,h);ctx.clip();ctx.strokeStyle="#c9d2ce80";ctx.lineWidth=1;for(let n=-h;n<w;n+=9){ctx.beginPath();ctx.moveTo(x+n,y);ctx.lineTo(x+n+h,y+h);ctx.stroke();}ctx.restore();}
      if(heatmap){ctx.fillStyle=valid&&rgb.reduce((a,b)=>a+b,0)<360?"#fff":"#14343a";ctx.font="10px ui-monospace,monospace";ctx.textAlign="center";ctx.textBaseline="middle";ctx.fillText(String(item.region.id).padStart(3,"0"),x+w/2,y+h/2);}
    }
    const [x,y,w,h]=regionBox(data.regions[state.source]);ctx.lineWidth=5;ctx.strokeStyle="#fff";ctx.strokeRect(x+1,y+1,w-2,h-2);ctx.lineWidth=3;ctx.strokeStyle="#008f87";ctx.strokeRect(x+1,y+1,w-2,h-2);ctx.fillStyle="#008f8733";ctx.fillRect(x,y,w,h);ctx.textAlign="start";
    el("legend-high").textContent=fmt(maximum);el("legend-low").textContent=el("log-scale").checked?"0 · log":"0";
    el("metric-description").textContent=data.metrics[state.metric].description+"。单位："+data.metrics[state.metric].unit+"。热图使用最终完整生成结果。";
    el("metric-description").hidden=!heatmap;el("map-legend").hidden=!heatmap;
    for(const node of document.querySelectorAll(".heatmap-option"))node.hidden=!heatmap;
    link("heatmap-download",data.static_heatmaps[`${state.step}:${scaleKey(state.scale)}:${state.metric}`]);
    const sorted=entries.filter(x=>x.value!=null).sort((a,b)=>b.value-a.value||a.region.id-b.region.id);
    el("ranking-count").textContent=`${sorted.length} / ${data.regions.length} 完成`;
    el("rankings").innerHTML=sorted.slice(0,40).map((item,index)=>`<button class="rank-row ${item.region.id===state.source?"active":""}" data-source="${item.region.id}"><span>${index+1}</span><span>Patch ${String(item.region.id).padStart(3,"0")}<div class="rank-bar" style="width:${maximum?Math.max(1,item.value/maximum*100):0}%"></div></span><span class="rank-value">${fmt(item.value)}</span></button>`).join("")||'<p class="note">当前条件尚无已完成分支。</p>';
    for(const button of el("rankings").querySelectorAll("button"))button.onclick=()=>selectSource(Number(button.dataset.source));
  }
  function selectionDetails() {
    const branch=current(),region=data.regions[state.source],stepT=data.timesteps[state.step],baseSigma=branch?branch.base_sigma:(data.protocol.sigmas||[])[state.step];
    el("selection-title").textContent=`Patch ${String(state.source).padStart(3,"0")} · 行 ${region.row} / 列 ${region.col}${region.num_tokens!=null?` · ${region.num_tokens} tokens`:""}`;
    el("selected-patch-hint").textContent=`当前选择 Patch ${String(state.source).padStart(3,"0")} · ${branch?"已完成":"等待生成"}`;
    el("condition").textContent=`第 ${state.step+1} 次去噪（索引 ${state.step}） · t=${stepT==null?"—":fmt(stepT)} · σ=${fmt(baseSigma)} → ${baseSigma==null?"—":fmt(baseSigma*state.scale)}`;
    if(matched){
      const sourceSigma=branch?branch.source_sigma:(baseSigma==null?null:baseSigma*state.scale),localT=branch?branch.local_timestep:(sourceSigma==null?null:sourceSigma*1000);
      el("condition").textContent=`第 ${state.step+1} 次去噪（source_step ${state.step}） · patch σ=${fmt(sourceSigma)} / token t=${localT??"—"} · 背景 t=${stepT??"—"} · 仅此次调用对齐`;
    }
    el("intervention-caption").textContent=`干预结果 · σ × ${scaleKey(state.scale)}`;
    if(matched)el("intervention-caption").textContent+= " · t 同步对齐";
    el("branch-status").className="branch-status"+(branch?"":" missing");
    el("branch-status").textContent=branch?`已完成 · source_step ${state.step} / 第 ${state.step+1} 次去噪。继续点击左侧其他 patch 即可切换。`:"这个 patch 在当前条件下尚未完成，请等待生成，或点击已完成的区域。";
    if(pilot&&!branch)el("branch-status").textContent="此 patch 不在本次验证样例中，尚未生成。请点击已完成的区域；全扫描在总览页面。";
    const values=branch?branch.metrics:{};
    const cards=[["完整视频 Pixel RMSE",values.pixel_rmse],["下一 chunk Latent MSE",values.next_chunk_mse],["下一 chunk / 输入能量",values.normalized_next_chunk_energy],["实际输入扰动能量",values.input_energy]];
    el("metric-cards").innerHTML=cards.map(([label,value])=>`<div class="metric-card">${label}<strong>${fmt(value)}</strong></div>`).join("");
    link("branch-video",branch&&branch.video_url);link("branch-record",branch&&branch.record_url);link("poster-download",data.posters[`${state.step}:${state.source}`]);link("trajectory-download",branch&&branch.latent_url);
  }
  function strengthCards() {
    el("strength-cards").innerHTML=data.scales.map(scale=>{const branch=data.branches[key(state.step,state.source,scale)];return `<div class="strength-card ${scale===state.scale?"active":""}" role="button" tabindex="0" data-scale="${scale}"><div class="strength-title"><strong>σ × ${scaleKey(scale)}</strong><span>${scale===0?"最强干预":scale===.75?"较弱干预":""}</span></div><canvas id="strength-${scaleKey(scale)}" width="416" height="240"></canvas><div class="strength-value">Pixel RMSE ${branch?fmt(branch.metrics.pixel_rmse):"未完成"}<br>下一 chunk MSE ${branch?fmt(branch.metrics.next_chunk_mse):"—"}</div></div>`;}).join("");
    for(const card of el("strength-cards").querySelectorAll(".strength-card")) {card.onclick=()=>selectScale(Number(card.dataset.scale));card.onkeydown=e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();selectScale(Number(card.dataset.scale));}};}
  }
  async function loadFinalFrames() {
    const version=++renderVersion,branch=current();
    const descriptors={baseline:data.baseline.media.final,intervention:branch&&branch.media.final,difference:branch&&branch.media.difference,pixelImpact:branch&&branch.media.pixel_mse_downsample,latentImpact:branch&&branch.media.latent_mse};
    for(const scale of data.scales){const other=data.branches[key(state.step,state.source,scale)];descriptors[`strength-${scaleKey(scale)}`]=other&&other.media.final;}
    currentFrames=Object.fromEntries(Object.entries(descriptors).map(([name,desc])=>[name,{desc,img:null}]));
    drawFinalFrames();
    const primary=["baseline","intervention","difference"];
    const entries=await Promise.all(primary.map(async name=>[name,{desc:descriptors[name],img:await image(descriptors[name],"high")}]));
    if(version!==renderVersion)return;
    Object.assign(currentFrames,Object.fromEntries(entries));drawFinalFrames();
    Promise.all(Object.entries(descriptors).filter(([name])=>!primary.includes(name)).map(async([name,desc])=>[name,{desc,img:await image(desc)}])).then(auxiliary=>{
      if(version!==renderVersion)return;
      Object.assign(currentFrames,Object.fromEntries(auxiliary));drawFinalFrames();
    });
  }
  function drawFinalFrames() {
    el("frame").value=state.frame;el("frame-label").textContent=`帧 ${state.frame} / ${data.display.frames-1}`;
    if(!currentFrames)return;
    const bindings=[["video-baseline","baseline",state.frame],["video-intervention","intervention",state.frame],["video-difference","difference",state.frame],["impact-pixels","pixelImpact",state.frame],["impact-latent","latentImpact",state.latentFrame],...data.scales.map(scale=>[`strength-${scaleKey(scale)}`,`strength-${scaleKey(scale)}`,state.frame])];
    for(const [id,name,index] of bindings){const item=currentFrames[name];frame(el(id),item.desc,item.img,index,item.desc?"加载本地预览…":"该分支尚未完成");}
  }
  async function renderTrace() {
    const version=++traceVersion,branch=current(),baseIndex=data.baseline.trajectory.findIndex(t=>t.chunk===state.chunk&&t.step===state.inspectStep),baseEntry=data.baseline.trajectory[baseIndex];
    const sourceChunk=data.protocol.source_chunk||0,shared=state.chunk<sourceChunk||(state.chunk===sourceChunk&&state.inspectStep<state.step);
    const changed=branch?(shared?data.baseline:branch):null,changedIndex=changed?changed.trajectory.findIndex(t=>t.chunk===state.chunk&&t.step===state.inspectStep):-1;
    const changedEntry=changed&&changed.trajectory[changedIndex];
    const stateText=!branch?"当前干预分支尚未完成，仅显示 baseline。":shared?"这个时刻早于干预，输入和预测都共享 baseline，差异为零。":`干预后轨迹 · chunk ${state.chunk} / inspect_step ${state.inspectStep}${state.chunk>sourceChunk&&state.inspectStep===0?" · 新 chunk 初始输入成对相同，clean 预测可受 AR 上下文影响":""}`;
    el("trace-status").textContent=stateText+(baseEntry?` · t=${fmt(baseEntry.global_t)} / σ=${fmt(baseEntry.sigma)}`:"");
    if(matched&&branch&&!shared){
      el("trace-status").textContent+=changedEntry&&changedEntry.mode==="matched"?` · 本次 source 调用：仅 frame ${changedEntry.source_frame} 的选中 patch 使用 σ=${fmt(changedEntry.source_sigma)}、token t=${changedEntry.local_timestep}，flow→x0 的 sigma_map 同步对齐；其余 token 保持原值。`:" · 此次调用使用原 schedule；不再覆盖 patch timestep / sigma_map。";
    }
    el("trace-status").dataset.timestepOverride=String(matched&&!!changedEntry&&changedEntry.mode==="matched");
    el("trace-status").className="branch-status"+(!branch?" missing":"");
    const jobs=[];
    for(const kind of ["noisy","clean"]){
      const vaeName=`${kind}_pixels`;
      jobs.push({id:`trace-${kind}-base`,owner:data.baseline,name:vaeName,index:baseIndex});
      jobs.push({id:`trace-${kind}-changed`,owner:changed,name:vaeName,index:changedIndex});
      jobs.push({id:`rms-${kind}-base`,owner:data.baseline,name:`${kind}_rms`,index:baseIndex*3+state.inspectFrame});
      jobs.push({id:`rms-${kind}-changed`,owner:changed,name:`${kind}_rms`,index:changedIndex*3+state.inspectFrame});
      jobs.push({id:`rms-${kind}-delta`,owner:changed,name:`${kind}_delta_rms`,index:changedIndex*3+state.inspectFrame});
    }
    const loaded=await Promise.all(jobs.map(async job=>{const desc=job.owner&&job.owner.media[job.name];return {...job,desc,img:await image(desc)};}));
    if(version!==traceVersion)return;
    for(const job of loaded)frame(el(job.id),job.desc,job.img,job.index,job.name.endsWith("pixels")?"此结果未保存 VAE 预览":"轨迹未完成或不存在");
    el("trace-status").dataset.sharedPrefix=String(shared&&!!branch);
    el("trace-status").dataset.branchEntry=changedEntry?`${changedEntry.chunk}:${changedEntry.step}`:"missing";
  }
  function renderSelection(){
    const selected=++selectionVersion;++renderVersion;++traceVersion;
    clearTimeout(mediaTimer);cancelPendingBranchImages();currentFrames=null;
    selectionDetails();strengthCards();renderMap();
    for(const id of ["video-intervention","video-difference","impact-pixels","impact-latent",...data.scales.map(s=>`strength-${scaleKey(s)}`),"trace-noisy-changed","trace-clean-changed","rms-noisy-changed","rms-clean-changed","rms-noisy-delta","rms-clean-delta"])empty(el(id),current()?"加载选中 patch…":"该分支尚未完成");
    mediaTimer=setTimeout(async()=>{if(selected!==selectionVersion)return;await loadFinalFrames();if(selected===selectionVersion)renderTrace();},80);
  }
  function selectSource(value){const source=Math.max(0,Math.min(data.regions.length-1,Math.round(value)||0));if(source===state.source)return;state.source=source;renderSelection();}
  function selectScale(value){if(value===state.scale)return;state.scale=value;el("strength").value=String(value);renderSelection();}
  function selectStep(value){if(value===state.step)return;state.step=value;el("source-step").value=String(value);renderSelection();}
  function play(){state.playing=!state.playing;el("play").textContent=state.playing?"❚❚ 暂停":"▶ 播放";lastFrameTime=0;if(state.playing)animation=requestAnimationFrame(tick);else if(animation)cancelAnimationFrame(animation);}
  function tick(now){if(!state.playing)return;const interval=1000/(data.display.fps*Number(el("speed").value));if(!lastFrameTime)lastFrameTime=now;if(currentFrames&&now-lastFrameTime>=interval){const jump=Math.floor((now-lastFrameTime)/interval);state.frame=(state.frame+jump)%data.display.frames;lastFrameTime+=jump*interval;drawFinalFrames();}animation=requestAnimationFrame(tick);}
  function pointerSource(event){const rectangle=el("heatmap").getBoundingClientRect(),x=(event.clientX-rectangle.left)/rectangle.width*832,y=(event.clientY-rectangle.top)/rectangle.height*480;return data.regions.find(region=>{const [rx,ry,rw,rh]=regionBox(region);return x>=rx&&x<rx+rw&&y>=ry&&y<ry+rh;});}
  function initialize(){
    const promptIndex=data.protocol.prompt_index??0,seed=data.protocol.seed??0;
    el("experiment-label").textContent=`SELF-FORCING · CHECKPOINT 600 · PROMPT ${promptIndex} / SEED ${seed}`;
    document.title=`Prompt ${promptIndex} · AR Patch Atlas`;
    if(matched){
      document.title+=" · Noise-level aligned";
      el("report-heading").textContent="逐 patch 扫描 · 噪声等级与 timestep 对齐";
      el("experiment-label").textContent+=" · NOISE-LEVEL ALIGNED";
      el("intervention-explanation").textContent="倍率越小，局部噪声相对 baseline 降得越多。σ × 0 是替换为模型前一步预测，不是真实干净图像，也不是删除该区域信息。仅 source 调用中，选中 patch 的实际 σ、token timestep（σ × 1000）与 flow→x0 的 sigma_map 同步对齐；背景 token 保留原 timestep / σ，后续调用按原 schedule 继续。";
      el("conditioning-note").hidden=false;
    }
    if(data.navigation){
      el("prompt-navigation").hidden=false;
      el("overview-link").href=data.navigation.overview_url;
      el("prompt-switch").innerHTML=data.navigation.cases.map(item=>`<option value="${esc(item.url)}" ${item.prompt_index===promptIndex&&item.seed===seed?"selected":""}>${esc(item.label)}</option>`).join("");
      el("prompt-switch").onchange=event=>{window.location.href=event.target.value;};
    }
    options("source-step",data.steps,state.step,s=>`step ${s} · 第 ${s+1} 次去噪`);options("strength",data.scales,state.scale,s=>`σ × ${scaleKey(s)}${s===0?" · 最强干预":""}`);options("metric",Object.keys(data.metrics),state.metric,s=>data.metrics[s].label);
    const chunks=[...new Set(data.baseline.trajectory.map(t=>t.chunk))],inspectSteps=[...new Set(data.baseline.trajectory.map(t=>t.step))];options("inspect-chunk",chunks,state.chunk,c=>`chunk ${c}`);options("inspect-step",inspectSteps,state.inspectStep,s=>`step ${s} · 第 ${s+1} 次去噪`);options("impact-latent-frame",Array.from({length:(data.protocol.shape||[1,6])[1]},(_,i)=>i),state.latentFrame,i=>`${i}${i>=3?" · 下一 chunk":" · 首个 chunk"}`);
    el("frame").max=data.display.frames-1;
    el("progress").innerHTML=`<span>${data.complete?"扫描完成":"扫描进行中"} · ${data.completed.toLocaleString()} / ${data.expected.toLocaleString()} 个完整生成分支</span><div class="progress-track"><div class="progress-fill" style="width:${100*data.completed/data.expected}%"></div></div>${data.complete?"":'<button id="refresh-results" type="button">刷新结果 ↻</button><span class="progress-note">后台持续生成，刷新页面可查看新结果</span>'}`;
    if(pilot)el("progress").innerHTML=`<span>验证样例（pilot） · ${data.completed.toLocaleString()} / ${data.expected.toLocaleString()} 个完整生成分支已保存</span><div class="progress-track"><div class="progress-fill" style="width:${100*data.completed/data.expected}%"></div></div><span class="progress-note">仅包含 patch ${data.report_context.source_ids.map(id=>String(id).padStart(3,"0")).join(" / ")}；其余区域未生成。本页保留验证样例，<a href="${esc(data.report_context.full_scan_url)}">查看全扫描总览 →</a></span>`;
    if(!data.complete&&!pilot)el("refresh-results").onclick=()=>window.location.reload();
    el("prompt").textContent="Prompt: "+data.prompt;el("generated-at").textContent="报告生成于 "+new Date(data.created_at).toLocaleString()+"。"+(data.complete?"完整扫描的本地报告。":"这是页面打开时的快照；刷新页面可获取后台最新发布的结果。");
    if(pilot)el("generated-at").textContent="报告生成于 "+new Date(data.created_at).toLocaleString()+"。这是独立验证样例，完整扫描覆盖率仍按原协议统计。";
    link("manifest-link",data.manifest_url);link("baseline-video",data.baseline.video_url);link("baseline-trajectory",data.baseline.latent_url);
    if(data.influence_summary){const links=Object.entries(data.influence_summary).filter(([,href])=>href);el("analysis-links").hidden=links.length===0;el("analysis-links").innerHTML="本轮完整生成的影响统计："+links.map(([kind,href])=>`<a href="${esc(href)}" target="_blank" rel="noopener">${kind.toUpperCase()} ↗</a>`).join(" · ");}
    el("source-step").onchange=e=>selectStep(Number(e.target.value));el("strength").onchange=e=>selectScale(Number(e.target.value));el("metric").onchange=e=>{state.metric=e.target.value;renderMap();};
    for(const id of ["selection-mode","color-scope","log-scale"])el(id).onchange=renderMap;
    el("heatmap").onclick=e=>{const region=pointerSource(e);if(region)selectSource(region.id);};el("heatmap").onkeydown=e=>{const offsets={ArrowLeft:-1,ArrowRight:1,ArrowUp:-data.cols,ArrowDown:data.cols};if(e.key in offsets){e.preventDefault();selectSource(state.source+offsets[e.key]);}};
    el("heatmap").onmousemove=e=>{const region=pointerSource(e),tip=el("map-tooltip"),hover=el("selector-hover").getContext("2d");hover.clearRect(0,0,832,480);if(!region){tip.hidden=true;return;}const [x,y,w,h]=regionBox(region);hover.strokeStyle="#ffd268";hover.lineWidth=3;hover.strokeRect(x+1,y+1,w-2,h-2);hover.fillStyle="#ffd26822";hover.fillRect(x,y,w,h);const b=data.branches[key(state.step,region.id,state.scale)];tip.textContent=`点击 Patch ${String(region.id).padStart(3,"0")} · ${region.num_tokens||"?"} tokens · ${b?"已完成，查看对比":"等待生成"}${b&&el("selection-mode").value==="heatmap"?" · "+fmt(b.metrics[state.metric]):""}`;tip.hidden=false;const rect=el("heatmap").getBoundingClientRect();tip.style.left=Math.min(e.clientX-rect.left+12,Math.max(0,rect.width-240))+"px";tip.style.top=Math.max(0,e.clientY-rect.top-36)+"px";};el("heatmap").onmouseleave=()=>{el("map-tooltip").hidden=true;el("selector-hover").getContext("2d").clearRect(0,0,832,480);};
    el("play").onclick=play;el("frame").oninput=e=>{state.frame=Number(e.target.value);drawFinalFrames();};el("impact-latent-frame").onchange=e=>{state.latentFrame=Number(e.target.value);drawFinalFrames();};
    el("inspect-chunk").onchange=e=>{state.chunk=Number(e.target.value);renderTrace();};el("inspect-step").onchange=e=>{state.inspectStep=Number(e.target.value);renderTrace();};el("inspect-frame").onchange=e=>{state.inspectFrame=Number(e.target.value);renderTrace();};
    renderSelection();
  }
  window.ARPatchAtlas={getState:()=>({...state}),selectSource,selectScale,selectStep,branchKey:key,regionBox,refresh:renderSelection};
  initialize();
})();
