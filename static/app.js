const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const state = { discovery: null, telemetry: null, running: false, frozen: false, speed: Array(30).fill(0), tokens: [], activeModel: null };

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
}
function compactNumber(value) {
  if (value == null) return '—';
  return Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 }).format(value);
}
function bytes(value) {
  if (!value) return '—';
  const units = ['B','KB','MB','GB','TB']; let n = value, i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i > 2 ? 2 : 1)} ${units[i]}`;
}
function pct(value) { return `${Math.max(0, Math.min(100, value || 0)).toFixed(0)}%`; }
function showNotice(message) { $('#noticeText').textContent = message; $('#notice').classList.remove('hidden'); }
function hideNotice() { $('#notice').classList.add('hidden'); }

async function api(path, options = {}) {
  const response = await fetch(path, { headers: {'Content-Type':'application/json'}, ...options });
  if (!response.ok) { let msg = response.statusText; try { msg = (await response.json()).error || msg; } catch {} throw new Error(msg); }
  return response.json();
}

function selectedRuntime() { return state.discovery?.runtimes?.find(r => r.id === $('#runtimeSelect').value); }

function renderDiscovery(data) {
  state.discovery = data;
  state.activeModel = data.models?.find(m => /saluki/i.test(m.name)) || data.models?.[0] || null;
  const model = state.activeModel;
  $('#modelName').textContent = model?.display_name || model?.name || 'No GGUF found';
  $('#modelMeta').textContent = model ? `${model.size} · ${model.architecture || 'GGUF'}` : 'Place a GGUF in ~/Models';
  const select = $('#runtimeSelect');
  const previous = select.value;
  if (data.runtimes?.length) {
    select.innerHTML = data.runtimes.map(r => `<option value="${escapeHtml(r.id)}">${escapeHtml(r.name)} · ${escapeHtml(new URL(r.base_url).port)}</option>`).join('');
    if (data.runtimes.some(r => r.id === previous)) select.value = previous;
    $('#startRuntime').textContent = 'Runtime online'; $('#startRuntime').disabled = true;
    hideNotice();
  } else {
    select.innerHTML = '<option value="">No runtime online</option>';
    $('#startRuntime').textContent = 'Start Saluki'; $('#startRuntime').disabled = false;
    showNotice('Saluki was found, but no model runtime is online. Start it here to enable live token probabilities.');
  }
  renderAnatomy(model); renderRuntimes(data);
}

function renderAnatomy(model) {
  if (!model) return;
  const items = [
    ['ARCHITECTURE', model.architecture || '—'], ['PARAMETERS', compactNumber(model.parameters)],
    ['EMBEDDING WIDTH', compactNumber(model.embedding)], ['ATTENTION HEADS', model.attention_heads ?? '—'],
    ['KV HEADS', model.attention_kv_heads ?? '—'], ['TRAIN CONTEXT', compactNumber(model.context_train)],
    ['TENSORS', compactNumber(model.tensor_count)], ['FILE SIZE', model.size || '—']
  ];
  $('#anatomySummary').innerHTML = items.map(([a,b]) => `<div><span>${a}</span><b>${escapeHtml(b)}</b></div>`).join('');
  const layers = Number(model.layers || 0); $('#layerCount').textContent = layers ? `${layers} layers` : 'Unknown';
  $('#layers').innerHTML = layers ? Array.from({length:layers}, (_,i) => `<div class="layer" title="Transformer block ${i}">${String(i).padStart(2,'0')}</div>`).join('') : '<span class="empty-trace">Layer metadata unavailable</span>';
  $('#rawMetadata').textContent = JSON.stringify(model, null, 2);
  $('#contextMax').textContent = compactNumber(model.context_train);
}

function renderRuntimes(data) {
  const online = data.runtimes || [];
  const cards = online.map(r => `<article class="panel runtime-card"><header><h3>${escapeHtml(r.name)}</h3><span class="online-badge">● ONLINE</span></header><dl><dt>Adapter</dt><dd>${escapeHtml(r.kind)}</dd><dt>Endpoint</dt><dd>${escapeHtml(r.base_url)}</dd><dt>Version</dt><dd>${escapeHtml(r.version || 'Endpoint-reported')}</dd><dt>Models</dt><dd>${escapeHtml((r.models || []).map(m => m.name).filter(Boolean).join(', ') || 'Loaded model')}</dd></dl><div class="feature-list">${(r.features || []).map(f => `<span>${escapeHtml(f)}</span>`).join('')}</div></article>`).join('');
  const detected = Object.entries(data.binaries || {}).flatMap(([kind, paths]) => paths.map(path => `<article class="panel runtime-card"><header><h3>${escapeHtml(kind)}</h3><span class="online-badge" style="color:var(--amber)">BINARY FOUND</span></header><dl><dt>Executable</dt><dd>${escapeHtml(path)}</dd><dt>Status</dt><dd>Not serving an endpoint</dd></dl></article>`)).join('');
  $('#runtimeGrid').innerHTML = cards + detected || '<article class="panel runtime-card offline-card"><div><h3>No runtime detected</h3><p>Bluso scans llama.cpp, Ollama, LM Studio and OpenAI-compatible ports.</p></div></article>';
}

function renderTelemetry(data) {
  state.telemetry = data;
  const system = data.system || {}, cpu = system.cpu_percent || 0, ram = system.memory_percent || 0;
  $('#cpuMini').style.width = pct(cpu); $('#ramMini').style.width = pct(ram);
  $('#cpuText').textContent = pct(cpu); $('#ramText').textContent = pct(ram); $('#hostName').textContent = `${system.platform || ''} ${system.machine || ''}`.trim();
  const modelProcess = (system.processes || []).sort((a,b)=>b.rss-a.rss)[0];
  $('#modelMemory').textContent = modelProcess ? bytes(modelProcess.rss) : '—';
  $('#memoryBar').style.width = pct(modelProcess && system.memory_total ? modelProcess.rss/system.memory_total*100 : 0);
  $('#memoryCaption').textContent = system.unified_memory ? `${pct(ram)} total unified memory used` : `${pct(ram)} system memory used`;
  const runtime = data.runtimes?.find(r => r.id === $('#runtimeSelect').value) || data.runtimes?.[0];
  $('#endpointBadge').textContent = runtime ? new URL(runtime.base_url).host : 'offline';
  const slot = runtime?.slots?.[0];
  if (slot) {
    const used = slot.n_tokens != null ? slot.n_tokens : (slot.next_token?.n_remain != null ? slot.n_ctx - slot.next_token.n_remain : 0);
    const total = slot.n_ctx || state.activeModel?.context_train || 0;
    $('#techContext').textContent = total ? total.toLocaleString() : '—';
    $('#contextUsed').textContent = compactNumber(Math.max(0, used)); $('#contextMax').textContent = compactNumber(total);
    const utilization = total ? 100*Math.max(0,used)/total : 0; $('#contextPercent').textContent = `${utilization.toFixed(1)}% utilized`; renderContextMeter(utilization);
  }
  const speed = runtime?.metrics?.['llamacpp:predicted_tokens_seconds'];
  if (!state.running && speed != null) updateSpeed(speed);
}

function renderContextMeter(utilization) {
  const meter = $('#contextMeter'); if (!meter.children.length) meter.innerHTML = Array.from({length:12},()=>'<i></i>').join('');
  [...meter.children].forEach((node,i)=>node.classList.toggle('on', (i+1)/12*100 <= utilization));
}
function updateSpeed(value) {
  value = Number(value || 0); $('#tokenSpeed').textContent = value.toFixed(1); state.speed.push(value); state.speed.shift();
  const max = Math.max(1, ...state.speed), points = state.speed.map((v,i)=>`${i/(state.speed.length-1)*180},${40-v/max*36}`).join(' '); $('#speedLine').setAttribute('points',points);
}

function renderProbabilities(prob, sample) {
  if (!prob || state.frozen) return;
  const alternatives = prob.alternatives || []; const selectedP = Number(prob.probability || 0);
  $('#chosenToken').textContent = visibleToken(prob.token); $('#chosenProb').textContent = `${(selectedP*100).toFixed(2)}%`; $('#chosenId').textContent = `Token ID ${prob.id ?? '—'}`; $('#sampleNumber').textContent = sample;
  $('#probList').innerHTML = alternatives.slice(0,10).map((p,i)=>`<li><span>${String(i+1).padStart(2,'0')}</span><code title="${escapeHtml(p.token)}">${escapeHtml(visibleToken(p.token))}</code><div class="prob-bar"><i style="width:${Math.min(100,p.probability*100)}%"></i></div><b>${(p.probability*100).toFixed(2)}%</b></li>`).join('');
  const total = alternatives.reduce((s,p)=>s+p.probability,0); const entropy = alternatives.reduce((s,p)=>s-(p.probability>0?p.probability*Math.log2(p.probability):0),0);
  $('#entropy').textContent = `${entropy.toFixed(2)} bits`; $('#confidence').textContent = selectedP > .65 ? 'High' : selectedP > .25 ? 'Medium' : 'Low';
  $('#probNote').innerHTML = `<i></i> Top 10 covers ${(total*100).toFixed(1)}% of probability mass. Values are computed from real llama.cpp logits.`;
}
function visibleToken(token) { return String(token ?? '').replaceAll(' ','␠').replaceAll('\n','↵') || '∅'; }
function addTokenTrace(prob, text, index) {
  const probability = prob?.probability || 0, token = prob?.token ?? text; if (!token) return;
  if ($('#tokenRibbon .empty-trace')) $('#tokenRibbon').innerHTML = '';
  const chip = document.createElement('span'); chip.className = `token-chip ${probability>.65?'high':probability>.25?'medium':'low'}`; chip.textContent = visibleToken(token); chip.title = `Token ${index} · ${(probability*100).toFixed(2)}% selected probability`; $('#tokenRibbon').appendChild(chip); $('#tokenRibbon').scrollLeft = $('#tokenRibbon').scrollWidth;
  state.tokens.push({token,index,probability});
  const layerNodes = $$('#layers .layer'); if (layerNodes.length) { layerNodes.forEach(n=>n.classList.remove('active')); layerNodes[index % layerNodes.length].classList.add('active'); }
}

async function generate() {
  if (state.running) return; const prompt = $('#prompt').value.trim(); if (!prompt) return;
  state.running = true; hideNotice(); state.tokens = []; $('#output').innerHTML = '<span class="cursor"></span>'; $('#tokenRibbon').innerHTML = '<span class="empty-trace">Waiting for first token…</span>'; $('#reasoningWrap').classList.add('hidden'); $('#reasoningText').textContent = '';
  $('#generateButton').disabled = true; $('#generationStatus').classList.add('running'); $('#generationStatus span').textContent = 'Running';
  let output = '', reasoning = '', lastTime = 0;
  try {
    const response = await fetch('/api/generate', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt, runtime_id:$('#runtimeSelect').value, temperature:Number($('#temperature').value), max_tokens:Number($('#maxTokens').value), top_p:.95})});
    if (!response.ok) { const body = await response.json(); throw new Error(body.error || response.statusText); }
    const reader = response.body.getReader(), decoder = new TextDecoder(); let buffer = '';
    while (true) {
      const {value,done} = await reader.read(); if (done) break; buffer += decoder.decode(value,{stream:true}); const lines = buffer.split('\n'); buffer = lines.pop();
      for (const line of lines) { if (!line.trim()) continue; const event = JSON.parse(line); if (event.type === 'token') {
        output += event.text || ''; reasoning += event.reasoning || ''; $('#output').textContent = output; $('#output').insertAdjacentHTML('beforeend','<span class="cursor"></span>'); $('#output').scrollTop = $('#output').scrollHeight;
        if (reasoning) { $('#reasoningWrap').classList.remove('hidden'); $('#reasoningText').textContent = reasoning; }
        renderProbabilities(event.probabilities,event.index); addTokenTrace(event.probabilities,event.text,event.index); updateSpeed(event.tokens_per_second); lastTime = event.elapsed_ms;
      } else if (event.type === 'finish') {
        const timing = event.timings || {}; const perToken = timing.predicted_per_token_ms || (event.tokens ? event.elapsed_ms/event.tokens : 0); $('#latency').textContent = perToken ? `${perToken.toFixed(1)} ms` : '—'; $('#latencyDetail').textContent = `${event.tokens || 0} tokens · ${((event.elapsed_ms||lastTime)/1000).toFixed(2)}s total`; $('#speedDelta').textContent = `Last run · ${event.tokens || 0} tokens`;
      } else if (event.type === 'error') throw new Error(event.message); }
    }
  } catch (error) { showNotice(error.message); if (!output) $('#output').innerHTML = `<span class="placeholder">${escapeHtml(error.message)}</span>`; }
  finally { state.running = false; $('#generateButton').disabled = false; $('#generationStatus').classList.remove('running'); $('#generationStatus span').textContent = 'Idle'; $$('#output .cursor').forEach(n=>n.remove()); refreshTelemetry(); }
}

async function refreshDiscovery() { try { renderDiscovery(await api('/api/discover')); await refreshTelemetry(); } catch (e) { showNotice(`Discovery failed: ${e.message}`); } }
async function refreshTelemetry() { try { renderTelemetry(await api('/api/telemetry')); } catch {} }
async function startRuntime() { const button=$('#startRuntime'); button.disabled=true; button.textContent='Starting…'; try { const result=await api('/api/runtime/start',{method:'POST',body:JSON.stringify({model_path:state.activeModel?.path})}); showNotice(`llama.cpp is ${result.state}. Loading a 27B model can take a minute; Bluso will reconnect automatically.`); let attempts=0; const poll=setInterval(async()=>{ attempts++; await refreshDiscovery(); if(state.discovery?.runtimes?.length||attempts>60)clearInterval(poll); },2000); } catch(e){showNotice(e.message);button.disabled=false;button.textContent='Start Saluki';} }
async function loadLogs() { try { const data=await api('/api/logs'); $('#logs').textContent=data.lines.join('\n')||'No runtime log output yet.'; $('#logs').scrollTop=$('#logs').scrollHeight; } catch(e){showNotice(e.message);} }

$$('.nav-item').forEach(button=>button.addEventListener('click',()=>{ $$('.nav-item').forEach(n=>n.classList.remove('active')); button.classList.add('active'); $$('.view').forEach(n=>n.classList.remove('active')); $(`#view-${button.dataset.view}`).classList.add('active'); if(button.dataset.view==='logs')loadLogs(); }));
$('#temperature').addEventListener('input',e=>$('#tempValue').textContent=Number(e.target.value).toFixed(2)); $('#maxTokens').addEventListener('input',e=>$('#tokensValue').textContent=e.target.value);
$('#generateButton').addEventListener('click',generate); $('#prompt').addEventListener('keydown',e=>{if((e.metaKey||e.ctrlKey)&&e.key==='Enter')generate();});
$('#refreshButton').addEventListener('click',refreshDiscovery); $('#startRuntime').addEventListener('click',startRuntime); $('#noticeClose').addEventListener('click',hideNotice); $('#reloadLogs').addEventListener('click',loadLogs);
$('#freezeProb').addEventListener('click',()=>{state.frozen=!state.frozen;$('#freezeProb').textContent=state.frozen?'▶':'Ⅱ';}); $('#runtimeSelect').addEventListener('change',refreshTelemetry);
renderContextMeter(0); $('#probList').innerHTML=Array.from({length:10},(_,i)=>`<li><span>${String(i+1).padStart(2,'0')}</span><code>—</code><div class="prob-bar"><i></i></div><b>0.00%</b></li>`).join('');
refreshDiscovery(); setInterval(()=>{ if(!state.running)refreshTelemetry(); },3000); setInterval(()=>{ if(!state.running&&!state.discovery?.runtimes?.length)refreshDiscovery(); },7000);
