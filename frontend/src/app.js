// 应用编排：上传 → 预览 → 仿真 → 报告，UI 状态管理。
import { Viewer, FEATURE_NAMES, tqiColor } from './viewer.js';

const $ = (id) => document.getElementById(id);
const state = {
  jobId: null,
  meta: null,
  layer: 0,
  mode: 'feature',
  polling: null,
  running: null,   // 'sim' | 'opt'
};

const viewer = new Viewer($('view'));

// ---------------------------------------------------------------- UI 基础
function toast(msg, ms = 5000) {
  const el = $('toast');
  el.textContent = msg;
  el.style.display = 'block';
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (el.style.display = 'none'), ms);
}

function fmtTime(s) {
  if (!s || s < 0) return '—';
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = Math.round(s % 60);
  return h > 0 ? `${h}h ${m}m` : m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}

// ---------------------------------------------------------------- 材料列表
async function loadMaterials() {
  try {
    const r = await fetch('/api/materials');
    const mats = await r.json();
    $('material').innerHTML = mats
      .map((m) => `<option value="${m.name}">${m.name}（喷嘴${m.nozzle}°C/热床${m.bed}°C）</option>`)
      .join('');
  } catch {
    toast('无法加载材料列表：请确认后端服务已启动');
  }
}

// ---------------------------------------------------------------- 上传
$('file').addEventListener('change', async (ev) => {
  const f = ev.target.files[0];
  if (!f) return;
  toast(`正在解析 ${f.name} …`, 60000);
  const fd = new FormData();
  fd.append('file', f);
  try {
    const r = await fetch('/api/upload', { method: 'POST', body: fd });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const { job_id, summary } = await r.json();
    state.jobId = job_id;
    state.meta = null;
    await loadPreview(job_id);
    showSummary(summary);
    $('simulate').disabled = false;
    toast(`解析完成：${summary.layers} 层 / ${summary.segments.toLocaleString()} 段`, 4000);
  } catch (e) {
    toast(`解析失败：${e.message}`);
  }
  ev.target.value = '';
});

// 预览：解析即渲染（特性配色），与仿真结果共用同一二进制格式
async function loadPreview(jobId) {
  const [metaR, binR] = await Promise.all([
    fetch(`/api/preview/${jobId}/meta`),
    fetch(`/api/preview/${jobId}/binary`),
  ]);
  if (!metaR.ok || !binR.ok) throw new Error('预览数据加载失败');
  const meta = await metaR.json();
  const buf = await binR.arrayBuffer();
  viewer.loadData(buf, meta);
  state.meta = meta;
  $('layer').max = meta.summary.layers - 1;
  $('layer').value = meta.summary.layers - 1;
  state.layer = meta.summary.layers - 1;
  $('layer').disabled = false;
  $('layernum').textContent = `层 ${meta.summary.layers - 1} / ${meta.summary.layers - 1}`;
  viewer.applyColors(state.mode);
  viewer.setLayerRange(state.layer, null);
}

// ---------------------------------------------------------------- 仿真
$('simulate').addEventListener('click', async () => {
  if (!state.jobId) return;
  const params = {
    material: $('material').value,
    voxel_mm: parseFloat($('voxel').value),
    bucket_s: 0.3,
  };
  const ch = $('chamber').value;
  if (ch !== '') params.chamber_temp = parseFloat(ch);
  try {
    const r = await fetch(`/api/job/${state.jobId}/simulate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
  } catch (e) {
    toast(`启动仿真失败：${e.message}`);
    return;
  }
  $('progress').classList.add('on');
  $('simulate').disabled = true;
  state.running = 'sim';
  state.polling = setInterval(pollJob, 700);
});

// ---------------------------------------------------------------- 速度优化
$('optimize').addEventListener('click', async () => {
  if (!state.jobId) return;
  const params = {
    material: $('material').value,
    rounds: 3,
  };
  const ch = $('chamber').value;
  if (ch !== '') params.chamber_temp = parseFloat(ch);
  try {
    const r = await fetch(`/api/job/${state.jobId}/optimize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
  } catch (e) {
    toast(`启动优化失败：${e.message}`);
    return;
  }
  $('progress').classList.add('on');
  $('optimize').disabled = true;
  $('simulate').disabled = true;
  state.running = 'opt';
  state.polling = setInterval(pollJob, 700);
});

async function pollJob() {
  if (!state.jobId) return;
  try {
    const r = await fetch(`/api/job/${state.jobId}`);
    const j = await r.json();
    const label = state.running === 'opt' ? '速度优化中' : '热仿真中';
    if (j.status === 'simulating' || j.status === 'optimizing') {
      $('pbar').firstElementChild.style.width = `${Math.round(j.progress * 100)}%`;
      $('ptext').textContent = `${label} ${(j.progress * 100).toFixed(1)}%`;
    } else if (j.status === 'done') {
      clearInterval(state.polling);
      $('progress').classList.remove('on');
      $('ptext').textContent = '';
      if (state.running === 'opt') {
        await loadOptimizeResult();
        $('optimize').disabled = false;
      } else {
        await loadResult();
        $('optimize').disabled = false;
        setMode('tqi');
      }
      $('simulate').disabled = false;
    } else if (j.status === 'error') {
      clearInterval(state.polling);
      $('progress').classList.remove('on');
      $('simulate').disabled = false;
      $('optimize').disabled = state.running === 'opt';
      toast(`计算出错：${(j.error || '').split('\n')[0]}`, 12000);
    }
  } catch { /* 轮询失败下次再试 */ }
}

async function loadOptimizeResult() {
  const r = await fetch(`/api/optimize/${state.jobId}/meta`);
  if (!r.ok) throw new Error('优化结果加载失败');
  const m = await r.json();
  renderOptimizeReport(m);
}

async function loadResult() {
  const [metaR, binR] = await Promise.all([
    fetch(`/api/result/${state.jobId}/meta`),
    fetch(`/api/result/${state.jobId}/binary`),
  ]);
  if (!metaR.ok || !binR.ok) throw new Error('结果加载失败');
  const meta = await metaR.json();
  const buf = await binR.arrayBuffer();
  viewer.loadData(buf, meta);
  state.meta = meta;
  $('layer').max = meta.summary.layers - 1;
  $('layer').value = meta.summary.layers - 1;
  state.layer = meta.summary.layers - 1;
  $('layernum').textContent = `层 ${state.layer} / ${meta.summary.layers - 1}`;
  viewer.applyColors(state.mode);
  viewer.setLayerRange(state.layer, null);
  renderReport(meta);
  $('report').classList.add('on');
  $('optimize').disabled = false;
}

// ---------------------------------------------------------------- 模式与层
function setMode(m) {
  state.mode = m;
  document.querySelectorAll('input[name=mode]').forEach((el) => (el.checked = el.value === m));
  viewer.applyColors(m);
  refreshLayerRange();
  renderLegend();
}

document.querySelectorAll('input[name=mode]').forEach((el) =>
  el.addEventListener('change', () => setMode(el.value))
);

// 滑块/仅当前层 只改 drawRange（便宜），图表高亮用 rAF 节流
function refreshLayerRange() {
  const only = $('onlycur').checked ? state.layer : null;
  viewer.setLayerRange(state.layer, only);
}

let chartRaf = 0;
$('layer').addEventListener('input', (ev) => {
  state.layer = parseInt(ev.target.value, 10);
  const max = parseInt($('layer').max, 10);
  $('layernum').textContent = `层 ${state.layer} / ${max}`;
  refreshLayerRange();
  if (!chartRaf) {
    chartRaf = requestAnimationFrame(() => {
      chartRaf = 0;
      if (state.meta && state.meta.layer_stats) drawLayerChart(state.meta);
    });
  }
});

$('onlycur').addEventListener('change', () => {
  refreshLayerRange();
  if (!chartRaf) {
    chartRaf = requestAnimationFrame(() => {
      chartRaf = 0;
      if (state.meta && state.meta.layer_stats) drawLayerChart(state.meta);
    });
  }
});

// ---------------------------------------------------------------- 图例
function renderLegend() {
  const lg = $('legend');
  if (state.mode === 'tqi') {
    let html = '';
    for (let i = 0; i < 26; i++) {
      const t = -100 + (i / 25) * 200;
      const [r, g, b] = tqiColor(t);
      html += `<i style="background:rgb(${r * 255 | 0},${g * 255 | 0},${b * 255 | 0})"></i>`;
    }
    lg.innerHTML = html;
    $('legendlab').textContent = 'TQI：−100 太冷（弱结合） ← 0 理想 → +100 太热（下垂）';
  } else {
    lg.innerHTML = '';
    $('legendlab').textContent =
      state.meta && state.meta.summary && state.meta.summary.has_type_comments === false
        ? '文件无特性标注，全部路径按统一色显示（TQI 仿真不受影响）'
        : '外墙橙 · 内墙黄 · 填充绿褐 · 支撑青 · 桥接粉';
  }
}

// ---------------------------------------------------------------- 摘要与报告
async function showSummary(s) {
  const el = $('report');
  el.classList.add('on');
  // 材料自动选中
  const matSel = $('material');
  if ([...matSel.options].some((o) => o.value === s.material)) matSel.value = s.material;
  $('optimize').disabled = true;
  el.innerHTML = `
    <h3>打印件信息</h3>
    <div class="kv">
      <span>材料（自动检测）</span><b>${s.material || '—'}</b>
      ${s.printer_model ? `<span>机型</span><b>${s.printer_model}${s.printer_variant ? ' · ' + s.printer_variant : ''}</b>` : ''}
      <span>尺寸</span><b>${(s.bbox_max[0]-s.bbox_min[0]).toFixed(0)}×${(s.bbox_max[1]-s.bbox_min[1]).toFixed(0)}×${(s.bbox_max[2]-s.bbox_min[2]).toFixed(1)} mm</b>
      <span>层数</span><b>${s.layers}</b>
      <span>路径段</span><b>${s.segments.toLocaleString()}</b>
      <span>切片耗时</span><b>${fmtTime(s.print_time_s)}</b>
      <span>耗料</span><b>${(s.extrusion_mm3 / 1000).toFixed(1)} cm³</b>
      <span>喷嘴/热床</span><b>${s.nozzle_temp}°C / ${s.bed_temp}°C</b>
      <span>切片器</span><b>${s.slicer || '—'}</b>
    </div>
    <h3>热仿真</h3>
    <div class="kv"><span>状态</span><b>待运行 —— 点击上方「开始热仿真」</b></div>`;
}

function renderReport(meta) {
  const s = meta.summary;
  const cfg = meta.config;
  // 统计（仅有效段）
  const stats = computeTqiStats(meta);
  const worst = meta.layer_stats
    .filter((l) => l.mean_tqi != null)
    .sort((a, b) => a.mean_tqi - b.mean_tqi)
    .slice(0, 5);

  const el = $('report');
  el.innerHTML = `
    <h3>打印件信息</h3>
    <div class="kv">
      <span>材料</span><b>${s.material}（仿真用 ${cfg.material}）</b>
      ${s.printer_model ? `<span>机型</span><b>${s.printer_model}</b>` : ''}
      <span>尺寸</span><b>${(s.bbox_max[0]-s.bbox_min[0]).toFixed(0)}×${(s.bbox_max[1]-s.bbox_min[1]).toFixed(0)}×${(s.bbox_max[2]-s.bbox_min[2]).toFixed(1)} mm</b>
      <span>层数 / 段数</span><b>${s.layers} / ${s.segments.toLocaleString()}</b>
      <span>切片耗时</span><b>${fmtTime(s.print_time_s)}</b>
    </div>
    <h3>TQI 总览（热质量指数）</h3>
    <div class="kv">
      <span>平均 TQI</span><b style="color:${tqiCss(stats.mean)}">${stats.mean.toFixed(1)}</b>
      <span>偏冷段（&lt;−50，弱结合）</span><b style="color:#6f9dff">${(stats.cold * 100).toFixed(1)}%</b>
      <span>偏热段（&gt;+50，下垂）</span><b style="color:#ff7a6e">${(stats.hot * 100).toFixed(1)}%</b>
      <span>统计样本</span><b>${stats.n.toLocaleString()} 段</b>
    </div>
    <h3>TQI 分布</h3>
    <canvas class="chart" id="hist" width="600" height="220"></canvas>
    <h3>逐层平均 TQI（点击跳层）</h3>
    <canvas class="chart" id="layerchart" width="600" height="220"></canvas>
    <h3>最差层 TOP 5</h3>
    ${worst.map((l) => `
      <div class="worstlayer" data-layer="${l.layer}">
        <span>第 ${l.layer} 层 · Z=${l.z.toFixed(2)}mm · 层时 ${fmtTime(l.t1 - l.t0)}</span>
        <b style="color:${tqiCss(l.mean_tqi)}">${l.mean_tqi.toFixed(0)}</b>
      </div>`).join('')}
    <h3>仿真设置</h3>
    <div class="kv">
      <span>体素 / 网格</span><b>${cfg.voxel_mm}mm · ${meta.grid_shape.join('×')}</b>
      <span>腔温 / 环境</span><b>${cfg.chamber_temp.toFixed(0)}°C / ${cfg.ambient_temp}°C</b>
      <span>仿真用时</span><b>${meta.runtime_s.toFixed(1)}s（${cfg.backend}）</b>
    </div>`;
  drawHist(stats);
  drawLayerChart(meta);
  el.querySelectorAll('.worstlayer').forEach((w) =>
    w.addEventListener('click', () => {
      const L = parseInt(w.dataset.layer, 10);
      $('layer').value = L;
      $('layer').dispatchEvent(new Event('input'));
    })
  );
}

function computeTqiStats(meta) {
  // 需要从二进制重算直方图 —— 直接读取 viewer 缓存
  const v = viewer;
  const n = v.count, u8 = v._u8, f32 = v._f32;
  const stride4 = 10;
  let sum = 0, cold = 0, hot = 0, cnt = 0;
  const bins = new Array(20).fill(0);
  for (let i = 0; i < n; i++) {
    if (u8[i * 40 + 37] !== 1) continue;
    const t = f32[i * stride4 + 6];
    sum += t; cnt++;
    if (t < -50) cold++;
    if (t > 50) hot++;
    const bi = Math.max(0, Math.min(19, Math.floor((t + 100) / 10)));
    bins[bi]++;
  }
  return { mean: cnt ? sum / cnt : 0, cold: cnt ? cold / cnt : 0, hot: cnt ? hot / cnt : 0, n: cnt, bins };
}

function tqiCss(t) {
  if (t == null) return '#8b96a5';
  const [r, g, b] = tqiColor(t);
  return `rgb(${r * 255 | 0},${g * 255 | 0},${b * 255 | 0})`;
}

function drawHist(stats) {
  const cv = $('hist');
  if (!cv) return;
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  const maxB = Math.max(...stats.bins, 1);
  const bw = W / 20;
  for (let i = 0; i < 20; i++) {
    const t = -100 + i * 10 + 5;
    const [r, g, b] = tqiColor(t);
    const h = (stats.bins[i] / maxB) * (H - 30);
    ctx.fillStyle = `rgb(${r * 255 | 0},${g * 255 | 0},${b * 255 | 0})`;
    ctx.fillRect(i * bw + 1, H - 18 - h, bw - 2, h);
  }
  ctx.fillStyle = '#8b96a5';
  ctx.font = '20px sans-serif';
  ctx.fillText('-100', 4, H - 2);
  ctx.fillText('0', W / 2 - 6, H - 2);
  ctx.fillText('+100', W - 52, H - 2);
  // 当前层标记
  const cur = currentLayerMean();
  if (cur != null) {
    const x = ((cur + 100) / 200) * W;
    ctx.strokeStyle = '#ffffff';
    ctx.beginPath();
    ctx.moveTo(x, 4); ctx.lineTo(x, H - 20);
    ctx.stroke();
  }
}

let layerChartMeta = null;
function drawLayerChart(meta) {
  const m = meta || layerChartMeta;
  if (!m) return;
  layerChartMeta = m;
  const cv = $('layerchart');
  if (!cv) return;
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  const rows = m.layer_stats.filter((l) => l.mean_tqi != null);
  if (!rows.length) return;
  const x0 = 8, x1 = W - 8, y0 = 8, y1 = H - 24;
  const X = (li) => x0 + ((li - rows[0].layer) / Math.max(rows[rows.length-1].layer - rows[0].layer, 1)) * (x1 - x0);
  const Y = (t) => y1 - ((t + 100) / 200) * (y1 - y0);
  // 参考线
  ctx.strokeStyle = '#3a4656';
  ctx.setLineDash([6, 6]);
  [-50, 0, 50].forEach((t) => {
    ctx.beginPath(); ctx.moveTo(x0, Y(t)); ctx.lineTo(x1, Y(t)); ctx.stroke();
  });
  ctx.setLineDash([]);
  // 平均 TQI 线
  ctx.strokeStyle = '#3fa7ff';
  ctx.lineWidth = 3;
  ctx.beginPath();
  rows.forEach((l, i) => {
    const x = X(l.layer), y = Y(l.mean_tqi);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
  // 最小 TQI 线（更暗）
  ctx.strokeStyle = '#33507a';
  ctx.lineWidth = 2;
  ctx.beginPath();
  rows.forEach((l, i) => {
    const x = X(l.layer), y = Y(l.min_tqi);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
  // 当前层竖线
  const cur = state.layer;
  ctx.strokeStyle = 'rgba(255,255,255,.5)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(X(cur), y0); ctx.lineTo(X(cur), y1);
  ctx.stroke();
  // 轴标签
  ctx.fillStyle = '#8b96a5';
  ctx.font = '20px sans-serif';
  ctx.fillText('+100', 4, y0 + 16);
  ctx.fillText('−100', 4, y1 - 4);
  ctx.fillText(`第${rows[0].layer}层`, x0, H - 4);
  ctx.fillText(`第${rows[rows.length - 1].layer}层`, x1 - 90, H - 4);
}

function currentLayerMean() {
  if (!state.meta || !state.meta.layer_stats) return null;
  const l = state.meta.layer_stats.find((x) => x.layer === state.layer);
  return l && l.mean_tqi != null ? l.mean_tqi : null;
}

// ---------------------------------------------------------------- 优化报告
function renderOptimizeReport(m) {
  const b = m.baseline, f = m.final;
  const dt = (b.est_time_s && f.est_time_s) ? (f.est_time_s - b.est_time_s) / b.est_time_s * 100 : null;
  const rows = m.rounds.map((r) => `
    <div class="worstlayer" style="cursor:default">
      <span>第 ${r.round} 轮</span>
      <b>TQI ${r.mean_tqi.toFixed(1)} · 冷 ${(r.cold_pct ?? 0).toFixed(0)}% · ${fmtTime(r.est_time_s)}</b>
    </div>`).join('');
  const el = $('report');
  el.insertAdjacentHTML('afterbegin', `
    <h3>速度优化结果（${m.material}）</h3>
    <div class="kv">
      <span>平均 TQI</span><b>${b.mean_tqi.toFixed(1)} → <span style="color:${tqiCss(f.mean_tqi)}">${f.mean_tqi.toFixed(1)}</span></b>
      <span>偏冷段占比</span><b>${(b.cold_pct ?? 0).toFixed(1)}% → ${(f.cold_pct ?? 0).toFixed(1)}%</b>
      <span>偏热段占比</span><b>${(b.hot_pct ?? 0).toFixed(1)}% → ${(f.hot_pct ?? 0).toFixed(1)}%</b>
      <span>预计时长</span><b>${fmtTime(b.est_time_s)} → ${fmtTime(f.est_time_s)}${dt != null ? `（${dt > 0 ? '+' : ''}${dt.toFixed(1)}%）` : ''}</b>
      <span>改写行数</span><b>${m.changed_lines.toLocaleString()}</b>
    </div>
    <h3>迭代过程</h3>${rows}
    <div style="margin:10px 0">
      <a class="btn primary" style="text-decoration:none;display:block;text-align:center"
         href="/api/optimize/${state.jobId}/download" download>下载优化后 G-code</a>
    </div>`);
}

// ---------------------------------------------------------------- BS 端点切换
async function loadBsMode() {
  try {
    const st = await (await fetch('/api/bsconfig')).json();
    if (st.mode) $('bsmode').value = st.mode;
  } catch { /* 服务未就绪 */ }
}

$('bsmode').addEventListener('change', async (ev) => {
  const mode = ev.target.value;
  if (!confirm(mode === 'local'
    ? '切换到本地引擎？需要完全关闭并重启 Bambu Studio 后生效。'
    : '切换回官方 Helio 云？需要完全关闭并重启 Bambu Studio 后生效（PAT 将自动还原为备份的官方令牌）。')) {
    await loadBsMode();
    return;
  }
  try {
    const r = await fetch('/api/bsconfig', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({mode})
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || r.statusText);
    if (j.warning) toast(j.warning, 8000);
    else toast(mode === 'local' ? '已切换到本地引擎——请重启 Bambu Studio'
                                : '已切换回官方 Helio 云——请重启 Bambu Studio', 8000);
  } catch (e) {
    toast('切换失败：' + e.message);
    await loadBsMode();
  }
});

// ---------------------------------------------------------------- 启动
loadMaterials();
renderLegend();
loadBsMode();

// 支持 /?job=<id> 直接载入已有任务（便于重开与调试）
(async function openJobFromUrl() {
  const id = new URLSearchParams(location.search).get('job');
  if (!id) return;
  try {
    const r = await fetch(`/api/job/${id}`);
    if (!r.ok) return;
    const j = await r.json();
    state.jobId = id;
    await loadPreview(id);
    showSummary(j.summary);
    $('simulate').disabled = false;
    if (j.status === 'done') {
      await loadResult();
      setMode('tqi');
      $('optimize').disabled = false;
      // 该任务若已优化过，直接恢复优化报告
      try {
        const om = await fetch(`/api/optimize/${id}/meta`);
        if (om.ok) renderOptimizeReport(await om.json());
      } catch { /* 无优化结果 */ }
    }
  } catch (e) {
    toast(`载入任务失败：${e.message}`);
  }
})();
