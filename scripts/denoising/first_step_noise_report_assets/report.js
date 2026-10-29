(() => {
  'use strict';
  const data = JSON.parse(document.getElementById('report-data').textContent);
  const summary = data.summary;
  const groups = summary.groups || [];
  const manifest = data.manifest || [];
  const assessment = data.assessment || {};
  const state = {prompt: null, frame: 1, stage: 'aligned_x0', metric: 'aligned_x0'};
  const $ = id => document.getElementById(id);
  const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
  const fmt = (value, digits = 3) => Number.isFinite(value) ? Number(value).toFixed(digits) : '—';
  const ci = value => Array.isArray(value) ? `[${fmt(value[0])}, ${fmt(value[1])}]` : '区间未计算';
  const percent = value => Number.isFinite(value) ? `${(100 * value).toFixed(1)}%` : '—';
  const safePath = value => typeof value === 'string' && !/^(?:[a-z][a-z\d+.-]*:|\/\/)/i.test(value) ? value : '';
  const shortNames = {42: '沙漠绿洲', 158: '人物行进', 300: '猫喝水'};
  const prompts = [...new Set(manifest.map(item => item.prompt_index))].sort((a,b) => a-b);
  const label = prompt => (assessment.comparisons || []).find(row => Number(row.prompt_index) === Number(prompt))?.label || shortNames[prompt] || `Prompt ${prompt}`;
  const promptGroup = (prompt, model) => groups.find(group => Number(group.prompt_index) === Number(prompt) && group.model === model);
  const stageLabel = stage => ({aligned_x0:'同 σ x₀',final:'最终结果',first:'初始 x₀ (σ=1)',final_alt_transition:'更换后续噪声'}[stage] || stage);
  const records = (prompt, model) => manifest.filter(item => item.prompt_index === Number(prompt) && item.model === model).sort((a,b) => a.seed-b.seed);
  const modelName = model => model === 'ar' ? 'AR' : 'Wan';
  const frameNumber = () => [0,4,8][state.frame];

  function metricRows(stage) {
    return prompts.map(prompt => {
      const ar = promptGroup(prompt, 'ar'), wan = promptGroup(prompt, 'wan');
      const source = (assessment.comparisons || []).find(row => Number(row.prompt_index) === Number(prompt) && row.stage === stage);
      const cross = (summary.cross_model || []).find(row => row.prompt_index === prompt);
      const comparison = cross?.comparisons?.[`ar_${stage}_vs_wan_${stage}`] || {};
      const arMse = source?.ar_mse ?? ar?.stages?.[stage]?.metrics?.mse?.mean;
      const wanMse = source?.wan_mse ?? wan?.stages?.[stage]?.metrics?.mse?.mean;
      const ratio = source?.ar_over_wan_ratio ?? comparison.ar_over_wan_across_seed_mse ?? (wanMse > 0 ? arMse / wanMse : null);
      const interval = source?.ratio_ci95 ?? comparison.ar_over_wan_across_seed_mse_ci95;
      return {prompt, arMse, wanMse, ratio, interval};
    });
  }

  function sampleButton(record, stage) {
    const path = safePath(record?.paths?.[stage]?.[state.frame]);
    if (!path) return '<div class="sample-missing">无此阶段</div>';
    const sigma = record.capture_stages?.[stage]?.sigma;
    const caption = `${modelName(record.model)} · ${label(record.prompt_index)} · seed ${record.seed} · ${stageLabel(stage)}${sigma == null ? '' : ` · σ=${sigma}`} · 帧 ${frameNumber()}`;
    return `<button type="button" class="sample" data-image="${esc(path)}" data-caption="${esc(caption)}" aria-label="放大：${esc(caption)}"><img src="${esc(path)}" alt="${esc(caption)}" decoding="async"><span>seed ${esc(record.seed)}</span></button>`;
  }

  const visualDefaults = {
    42: {title:'沙漠绿洲',text:'本批样本中，AR 的粗粒度 RGB 距离小于 Wan，支持它在颜色与粗布局上更集中。棕榈、沙地、水面的场景限定本身较强；这仍不等于语义模式数量更少。'},
    158: {title:'人物行进',text:'AR 多见道路或步道上的背影、下半身近景；Wan 出现不同距离、朝向及背景。RGB 距离没有给出稳定的统一排序，视觉差异更适合描述为构图偏好不同。'},
    300: {title:'猫喝水',text:'AR 多为容器边猫脸近景；Wan 还出现地面水源和较远的室内构图。这提示本批 Wan 的构图 / 场景覆盖可能更广，但还没有语义类别标注来检验这一点。'}
  };
  const observations = prompts.map(prompt => ({prompt_index:prompt,...visualDefaults[prompt],...(assessment.visual_observations || []).find(row => Number(row.prompt_index) === Number(prompt))}));

  function renderSamples() {
    const exemplar = manifest.find(record => record.prompt_index === state.prompt);
    $('prompt-text').textContent = exemplar?.prompt || '';
    $('sample-caption').textContent = `${state.stage === 'aligned_x0' ? '共同噪声水平 σ = 0.9375 上的 clean x₀ 预测' : '各模型完成采样后的 final'} · 当前显示帧 ${frameNumber()} · 两侧相同位置对应相同初始 seed（跨模型初噪经 hash 校验）`;
    $('sample-grids').innerHTML = ['ar','wan'].map(model => `<section class="model-panel ${model}" aria-label="${modelName(model)} 全部 seed"><div class="model-head"><h3>${modelName(model)}</h3><span>${model === 'ar' ? '蒸馏四步 · 无额外 CFG' : '双向 48 步 · CFG 6'} · ${esc(stageLabel(state.stage))}</span></div><div class="seed-grid">${records(state.prompt,model).map(record => sampleButton(record,state.stage)).join('')}</div></section>`).join('');
    const observation = observations.find(row => Number(row.prompt_index) === state.prompt);
    $('prompt-observation').innerHTML = `<strong>视觉观察 · </strong>${esc(observation?.text || '比较主体位置、镜头尺度、背景类别与画面布局。视觉观察仅用于描述当前样本。')}`;
    if (document.querySelector('.advanced').open) renderTrajectory();
  }

  function renderTrajectory() {
    const stages = [['ar','first'],['ar','aligned_x0'],['ar','final'],['wan','aligned_x0'],['wan','final']];
    const seeds = [...new Set(records(state.prompt,'ar').concat(records(state.prompt,'wan')).map(record => record.seed))].sort((a,b)=>a-b);
    let body = '<div class="trajectory"><div class="trajectory-head">seed</div>' + stages.map(([model,stage])=>`<div class="trajectory-head ${model}-text">${modelName(model)} · ${esc(stageLabel(stage))}</div>`).join('');
    for (const seed of seeds) {
      body += `<div class="trajectory-seed">${esc(seed)}</div>` + stages.map(([model,stage])=>sampleButton(records(state.prompt,model).find(record=>record.seed===seed),stage)).join('');
    }
    $('trajectory-grid').innerHTML = body + '</div>';
  }

  function renderMetrics() {
    const rows = metricRows(state.metric);
    $('metric-rows').innerHTML = rows.map(row => {
      const direction = row.interval?.[1] < 1 ? ['lower','AR 的 RGB 更集中'] : row.interval?.[0] > 1 ? ['higher','AR 的 RGB 更分散'] : ['unclear','尚无明确方向'];
      return `<tr><td>${esc(label(row.prompt))} <span class="muted">#${row.prompt}</span></td><td>${fmt(row.arMse,4)}</td><td>${fmt(row.wanMse,4)}</td><td><strong>${fmt(row.ratio)}</strong> <span class="muted">${ci(row.interval)}</span></td><td><span class="direction ${direction[0]}">${direction[1]}</span></td></tr>`;
    }).join('');
    const width=850, left=135, right=735, top=28, rowHeight=43;
    const upper = Math.max(1.5,...rows.map(row=>Math.min(4,row.interval?.[1] || row.ratio || 1))) * 1.08;
    const x = value => left + Math.max(0,Math.min(upper,value)) / upper * (right-left);
    const h = top + rows.length * rowHeight + 23;
    let svg = `<svg class="ratio-svg" viewBox="0 0 ${width} ${h}" role="img" aria-label="${state.metric === 'final' ? '最终结果' : '同噪声水平 x0'}跨 seed RGB 距离比值；横线表示 seed bootstrap 探索性 95% 区间"><line x1="${x(1)}" x2="${x(1)}" y1="15" y2="${h-25}" class="reference"/>`;
    for (const tick of [0,.5,1,1.5,2,2.5,3,3.5,4].filter(t=>t<=upper)) svg+=`<text x="${x(tick)}" y="${h-6}" text-anchor="middle">${tick}</text>`;
    rows.forEach((row,index)=>{
      const y=top+index*rowHeight;
      svg+=`<text x="4" y="${y+5}" class="chart-label">${esc(label(row.prompt))}</text><line x1="${left}" x2="${right}" y1="${y}" y2="${y}" stroke="#e7ecf3"/>`;
      if (row.interval && row.interval.every(Number.isFinite)) svg+=`<line x1="${x(row.interval[0])}" x2="${x(row.interval[1])}" y1="${y}" y2="${y}" class="ci-line"/><line x1="${x(row.interval[0])}" x2="${x(row.interval[0])}" y1="${y-5}" y2="${y+5}" class="ci-cap"/><line x1="${x(row.interval[1])}" x2="${x(row.interval[1])}" y1="${y-5}" y2="${y+5}" class="ci-cap"/>`;
      if (Number.isFinite(row.ratio)) svg+=`<circle cx="${x(row.ratio)}" cy="${y}" r="5" class="point"/><text x="${right+18}" y="${y+4}" class="chart-value">${fmt(row.ratio)}</text>`;
    });
    $('ratio-chart').innerHTML = svg+'</svg>';
  }

  function renderStatic() {
    let arSuccess=0,wanSuccess=0,arN=0,wanN=0;
    $('retention-rows').innerHTML=prompts.map(prompt=>{
      const ar=promptGroup(prompt,'ar')?.retention?.aligned_x0,wan=promptGroup(prompt,'wan')?.retention?.aligned_x0;
      const success = row => row ? `${Math.round(row.retrieval_accuracy * row.n_seeds)} / ${row.n_seeds}` : '—';
      if(ar){arSuccess+=ar.retrieval_accuracy*ar.n_seeds;arN+=ar.n_seeds;}
      if(wan){wanSuccess+=wan.retrieval_accuracy*wan.n_seeds;wanN+=wan.n_seeds;}
      return `<tr><td>${esc(label(prompt))}</td><td class="ar-text"><strong>${success(ar)}</strong></td><td class="wan-text"><strong>${success(wan)}</strong></td><td>${fmt(ar?.matched_over_mismatched_mse)} <span class="muted">${ci(ar?.ratio_ci95)}</span></td><td>${fmt(wan?.matched_over_mismatched_mse)} <span class="muted">${ci(wan?.ratio_ci95)}</span></td></tr>`;
    }).join('');
    $('retention-headline').textContent=`AR ${Math.round(arSuccess)}/${arN}，Wan ${Math.round(wanSuccess)}/${wanN}，均能从同 prompt 的 8 个 final 中匹配自己的 seed。这支持样本特征的早期保留。`;
    $('control-rows').innerHTML=prompts.map(prompt=>{
      const row=promptGroup(prompt,'ar')?.transition_noise_intervention;
      return `<tr><td>${esc(label(prompt))}</td><td>${fmt(row?.same_initial_changed_transition_mse,4)}</td><td>${fmt(row?.across_initial_final_mse,4)}</td><td><strong>${fmt(row?.transition_over_initial_mse)}</strong> <span class="muted">${ci(row?.ratio_ci95)}</span></td></tr>`;
    }).join('');
    $('semantic-cards').innerHTML=observations.map(row=>`<article class="semantic-card"><h4>${esc(row.title || label(row.prompt_index))}</h4><p>${esc(row.text)}</p></article>`).join('');
    const catAr=promptGroup(300,'ar')?.stages?.aligned_x0?.metrics?.centered_cosine?.mean;
    const catWan=promptGroup(300,'wan')?.stages?.aligned_x0?.metrics?.centered_cosine?.mean;
    if (Number.isFinite(catAr) && Number.isFinite(catWan)) $('cat-secondary').textContent=`例如，同 σ 猫样本的去均值 RGB 余弦相似度，AR 为 ${fmt(catAr)}、Wan 为 ${fmt(catWan)}。这是另一种像素空间描述，数值越大表示更相似；它也没有语义类别标签，仍不能单独完成 diversity 判定。`;
    if (assessment.verdict?.summary) $('verdict-detail').textContent=assessment.verdict.summary;
    if (assessment.next_experiment?.summary) $('next-experiment').innerHTML=`<p>${esc(assessment.next_experiment.summary)}</p><ul>${(assessment.next_experiment.items||[]).map(item=>`<li>${esc(item)}</li>`).join('')}</ul>`;
    $('downloads').innerHTML=(data.downloads || []).map(item=>`<a href="${esc(safePath(item.path))}" download>${esc(item.label)} ↓</a>`).join('');
    const warnings=[...(summary.warnings||[]),...(assessment.limitations||[])];
    $('warnings').innerHTML=[...new Set(warnings)].map(warning=>`<li>${esc(warning)}</li>`).join('');
    $('method-json').textContent=JSON.stringify({primary_comparison:summary.primary_comparison,metric_protocol:summary.metric_protocol,alignment_coverage:summary.alignment_coverage,problems:summary.problems},null,2);
    $('parameters-json').textContent=JSON.stringify(groups.map(group=>({prompt_index:group.prompt_index,model:group.model,seeds:group.seeds,run_parameters:group.run_parameters,capture_stages:group.capture_stages,available_stage_keys:Object.keys(group.stages||{})})),null,2);
    $('validation-json').textContent=JSON.stringify(data.validation || {note:'validation.json 尚不存在'},null,2);
    $('sheet-links').innerHTML=(data.sheets||[]).map(sheet=>`<a href="${esc(safePath(sheet.path))}">${esc(sheet.label)}</a>`).join('');
    if(data.assessment){$('assessment-details').hidden=false;$('assessment-json').textContent=JSON.stringify(data.assessment,null,2);}
    const latent=data.latent?.groups || {};
    const latentRows=prompts.map(prompt=>{
      const ar=latent[`ar/prompt_${String(prompt).padStart(6,'0')}`]?.stages,wan=latent[`wan/prompt_${String(prompt).padStart(6,'0')}`]?.stages;
      return `<tr><td>${esc(label(prompt))}</td><td>${fmt(ar?.aligned_x0?.pairwise_mse)}</td><td>${fmt(wan?.aligned_x0?.pairwise_mse)}</td><td>${fmt(ar?.final?.pairwise_mse)}</td><td>${fmt(wan?.final?.pairwise_mse)}</td></tr>`;
    }).join('');
    $('latent-table').innerHTML=`<table><thead><tr><th>Prompt</th><th>AR 同 σ x₀ MSE</th><th>Wan 同 σ x₀ MSE</th><th>AR final MSE</th><th>Wan final MSE</th></tr></thead><tbody>${latentRows}</tbody></table>`;
  }

  function bindSegment(id,attribute,key,render) {
    $(id).addEventListener('click',event=>{
      const button=event.target.closest(`button[${attribute}]`);if(!button)return;
      state[key]=key==='frame'?Number(button.getAttribute(attribute)):button.getAttribute(attribute);
      $(id).querySelectorAll('button').forEach(item=>item.setAttribute('aria-pressed',String(item===button)));
      render();
    });
  }
  $('prompt-select').innerHTML=prompts.map(prompt=>`<option value="${prompt}">${esc(label(prompt))} · prompt #${prompt}</option>`).join('');
  state.prompt=prompts.includes(300)?300:prompts[0];
  $('prompt-select').value=String(state.prompt);
  $('prompt-select').addEventListener('change',event=>{state.prompt=Number(event.target.value);renderSamples();});
  bindSegment('frame-control','data-frame','frame',renderSamples);
  bindSegment('stage-control','data-stage','stage',renderSamples);
  bindSegment('metric-control','data-metric','metric',renderMetrics);
  document.querySelector('.advanced').addEventListener('toggle',event=>{if(event.target.open)renderTrajectory();});
  const modal=$('image-modal');
  document.addEventListener('click',event=>{
    const button=event.target.closest('button[data-image]');if(!button)return;
    $('modal-image').src=button.dataset.image;$('modal-image').alt=button.dataset.caption;$('modal-caption').textContent=button.dataset.caption;
    modal.showModal();
  });
  $('modal-close').addEventListener('click',()=>modal.close());
  modal.addEventListener('click',event=>{if(event.target===modal)modal.close();});
  renderStatic();renderSamples();renderMetrics();
})();
