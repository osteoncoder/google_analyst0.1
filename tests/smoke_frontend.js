#!/usr/bin/env node
/*
 * smoke_frontend.js — headless smoke test for the dashboard frontend.
 *
 * Executes data.js + charts.js + ml_dashboard.js in a Node `vm` sandbox with
 * minimal DOM/Plotly/fetch stubs (no browser needed) and asserts that:
 *   - the KPI strip and charts 01-06 render with real, non-fabricated values
 *   - missing-data sections (chart 03 without dates, chart 06 without price)
 *     show explicit data-gap states instead of invented figures
 *   - sections 07-09 render from the real metrics.json payload
 *   - form submission performs a real inference call and renders the result
 *
 * Run:  node tests/smoke_frontend.js
 * Exits 0 on success, 1 on any failed assertion.
 */
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const ROOT = path.resolve(__dirname, '..');
const failures = [];
const fail = (msg) => { console.error('  FAIL: ' + msg); failures.push(msg); };
const ok = (msg) => console.log('  ok  : ' + msg);

/* ---------------- data payloads (real artifacts) ---------------- */
let appsJson, metricsJson;
try {
  appsJson = JSON.parse(fs.readFileSync(path.join(ROOT, 'data', 'apps.json'), 'utf8'));
  metricsJson = JSON.parse(fs.readFileSync(path.join(ROOT, 'ml', 'artifacts', 'metrics.json'), 'utf8'));
} catch (e) {
  console.error('Missing data/apps.json or ml/artifacts/metrics.json — run `python clean.py && python train_models.py` first.');
  process.exit(1);
}

/* ---------------- DOM / Plotly / fetch stubs ---------------- */
const elements = new Map();
class El {
  constructor(id) {
    this.id = id; this.innerHTML = ''; this.textContent = '';
    this.value = ''; this.style = {}; this.data = null; this.listeners = {};
    const self = this;
    this.classList = {
      _s: new Set(),
      add(c) { self.classList._s.add(c); },
      remove(c) { self.classList._s.delete(c); },
      toggle(c, force) { const on = force === undefined ? !self.classList._s.has(c) : !!force; on ? self.classList._s.add(c) : self.classList._s.delete(c); },
      contains(c) { return self.classList._s.has(c); },
    };
    this.parentElement = { innerHTML: '', _banners: '', insertAdjacentHTML(pos, html) { this.innerHTML += html; } };
  }
  addEventListener(ev, fn) { (this.listeners[ev] = this.listeners[ev] || []).push(fn); }
  insertAdjacentHTML(pos, html) { this.innerHTML += html; }
  focus() {}
}
const docListeners = {};
const plots = {};
const sandbox = {
  console,
  Date, JSON, Math, Promise, setTimeout, clearTimeout,
  parseInt, parseFloat, isFinite, Number, String, Object, Array, Map, Set, RegExp, Error,
  window: null,
  document: {
    getElementById(id) { if (!elements.has(id)) elements.set(id, new El(id)); return elements.get(id); },
    addEventListener(ev, fn) { (docListeners[ev] = docListeners[ev] || []).push(fn); },
    querySelectorAll() { return { forEach() {} }; },
  },
  fetch: async (url) => {
    if (url === 'data/apps.json') return { ok: true, json: async () => appsJson };
    if (url === 'api/metrics') return { ok: true, json: async () => metricsJson };
    if (url === 'api/predict/rating') {
      return { ok: true, json: async () => ({ predicted_rating: 4.2, model: 'Test', test_metrics: { mae: 0.3, rmse: 0.4, r2: 0.2 }, n_test: 10, assumptions: ['test assumption'], warning: 'test warning' }) };
    }
    if (url === 'api/predict/tier') {
      return { ok: true, json: async () => ({ predicted_tier: '1M-100M', probabilities: { 'Under 10K': 0.1, '10K-1M': 0.2, '1M-100M': 0.5, '100M+': 0.2 }, tier_order: ['Under 10K', '10K-1M', '1M-100M', '100M+'], model: 'Test', test_metrics: { accuracy: 0.7, macro_f1: 0.6, weighted_f1: 0.65 }, assumptions: ['test assumption'], warning: 'Probabilities are model estimates, not guarantees. test' }) };
    }
    return { ok: false, status: 404, json: async () => ({}) };
  },
  Plotly: {
    newPlot(el, traces, layout) { el.data = { traces, layout }; plots[el.id] = { traces, layout }; },
    Plots: { resize() {} },
  },
};
sandbox.window = sandbox;
sandbox.globalThis = sandbox;
sandbox.addEventListener = () => {}; // window.addEventListener (resize/scroll)
vm.createContext(sandbox);

/* ---------------- load the three frontend files in order ---------------- */
for (const f of ['data.js', 'charts.js', 'ml_dashboard.js']) {
  const code = fs.readFileSync(path.join(ROOT, f), 'utf8');
  try { vm.runInContext(code, sandbox, { filename: f }); }
  catch (e) { fail(`${f} threw at load: ${e.message}`); process.exit(1); }
}
ok('data.js + charts.js + ml_dashboard.js load without errors');

const tick = (ms = 100) => new Promise(r => setTimeout(r, ms));

(async () => {
  /* ---------------- fire DOMContentLoaded -> bootstrap ---------------- */
  (docListeners['DOMContentLoaded'] || []).forEach(fn => fn());
  await tick();

  /* ---------------- KPI + charts 01-06 ---------------- */
  const rows = appsJson.rows;
  const n = rows.length;

  const kpi = elements.get('kpiGrid');
  if (kpi && kpi.innerHTML.includes('Apps in dataset')) ok(`KPI strip rendered with real values (apps = ${n})`);
  else fail('KPI strip missing');

  const c1 = plots['chart1'];
  const cats1 = new Set(rows.map(r => r.category)).size;
  if (c1 && c1.traces.length === cats1) ok(`chart1: ${cats1} category traces (one per real category)`);
  else fail(`chart1 trace count ${c1 ? c1.traces.length : 'none'} != ${cats1}`);
  if (c1 && c1.traces.every(t => t.y.every(v => v >= 1 && v <= 5))) ok('chart1: all ratings within 1-5');

  const c2 = plots['chart2'];
  if (c2 && c2.traces[0].type === 'heatmap' && c2.traces[0].z.length === c2.traces[0].x.length && c2.traces[0].z.length >= 4)
    ok(`chart2: ${c2.traces[0].z.length}x${c2.traces[0].z.length} Pearson matrix over real columns`);
  else fail('chart2 missing/invalid');
  if (elements.get('chart2note') && elements.get('chart2note').textContent.includes('association, not causation'))
    ok('chart2: "correlation = association, not causation" note present');
  else fail('chart2 causation note missing');

  // chart3 is data-aware: real date column → stacked bars; no dates → gap card
  const hasDates = rows.some(r => r.last_updated);
  const c3el = elements.get('chart3');
  if (hasDates) {
    const c3 = plots['chart3'];
    const tot = c3 && c3.traces.reduce((s, t) => s + t.y.reduce((a, b) => a + b, 0), 0);
    if (c3 && c3.traces.length >= 1 && tot === rows.filter(r => r.last_updated).length)
      ok(`chart3 (dated data): stacked bars, counts sum to all ${tot} dated apps`);
    else fail(`chart3 dated data invalid (traces=${c3 ? c3.traces.length : 'none'}, total=${tot})`);
  } else {
    if (c3el && c3el.innerHTML.includes('data-gap')) ok('chart3: explicit data-gap state (no Last Updated column)');
    else fail('chart3 should show a data-gap card when dates are absent');
  }

  const c4 = plots['chart4'];
  const rated = rows.filter(r => typeof r.rating === 'number').length;
  if (c4 && c4.traces[0].y.reduce((a, b) => a + b, 0) === rated) ok(`chart4: histogram counts sum to ${rated} (every rated app)`);
  else fail('chart4 counts do not match rated apps');

  const c5 = plots['chart5'];
  if (c5 && c5.traces[0].x.every(v => v > 0)) ok('chart5: all category totals > 0 (zero-safe log axis)');
  else fail('chart5 contains non-positive totals on a log axis');
  if (elements.get('chart5note') && elements.get('chart5note').textContent.includes('unweighted mean')) ok('chart5: aggregation method stated in note');

  // chart6 is data-aware: real paid apps → Free/Paid + price panel; none → gap banner
  const hasPricing = rows.some(r => typeof r.price === 'number' && r.price > 0);
  const c6 = plots['chart6'];
  const parent6 = elements.get('chart6').parentElement;
  if (hasPricing) {
    if (c6 && c6.traces.length === 2 && plots['chart6b'] !== undefined &&
        !(parent6.innerHTML || '').includes('No Price column'))
      ok('chart6 (pricing data): Free/Paid counts + mean-listed-price panel, no "No Price" banner');
    else fail('chart6 pricing path invalid (expected 2 traces + price panel, no gap banner)');
  } else {
    if (c6 && plots['chart6b'] === undefined && (parent6.innerHTML || '').includes('No Price column'))
      ok('chart6 (no pricing): explicit "No Price column" banner (no invented revenue)');
    else fail('chart6 should show the data-gap banner without pricing data');
  }

  /* ---------------- ML sections 07-09 ---------------- */
  const rmc = elements.get('ratingModelCard');
  if (rmc && rmc.innerHTML.includes('rating regression')) ok('section 07: model card rendered from real metrics');
  else fail('section 07 model card missing');
  const cm = plots['chart_confusion'];
  if (cm && cm.traces[0].z.length === 4 && cm.traces[0].z.flat().length === 16) ok('section 08: 4x4 confusion matrix rendered');
  else fail('section 08 confusion matrix missing');
  const pM1 = elements.get('perfM1');
  if (pM1 && pM1.innerHTML.includes('Random Forest')) ok('section 09: M1 candidate table rendered');
  else fail('section 09 M1 table missing');
  const pM2a = elements.get('perfM2a'), pM2b = elements.get('perfM2b');
  if (pM2a && pM2b && pM2b.innerHTML.length > 50) ok('section 09: both M2 version tables rendered');
  else fail('section 09 M2 tables missing');
  const pSum = elements.get('perfSummary');
  if (pSum && pSum.innerHTML.includes('lower bounds') && pSum.innerHTML.includes('not observed revenue'))
    ok('section 09: limitation notes present (install bands, price ≠ revenue)');
  else fail('section 09 limitation notes missing');
  if (metricsJson.dataset.is_sample && pSum.innerHTML.includes('sample')) ok('section 09: sample-dataset warning surfaced');

  /* ---------------- chart3 with DATES (exercises the non-gap path) ---------------- */
  // The real sample has no dates, so also run renderChart3 on synthetic dated
  // rows to prove the stacked-bar path works when the full dataset is loaded.
  const datedRows = [
    { app: 'A', category: 'Game', rating: 4.2, reviews: 100, installs: 5000, size_mb: 10, price: 0, type: 'Free', last_updated: new Date(Date.UTC(2025, 0, 15)), sentiment: NaN },
    { app: 'B', category: 'Game', rating: 4.0, reviews: 200, installs: 8000, size_mb: 12, price: 0, type: 'Free', last_updated: new Date(Date.UTC(2025, 1, 10)), sentiment: NaN },
    { app: 'C', category: 'Tools', rating: 3.8, reviews: 50,  installs: 3000, size_mb: 8,  price: 0, type: 'Free', last_updated: new Date(Date.UTC(2025, 1, 20)), sentiment: NaN },
  ];
  // `DF` is a lexical binding in the vm context — drive it via an in-context fn.
  const driveChart3 = vm.runInContext(
    '(function(rows, real){ __realDF = real; DF = rows; renderChart3(); })', sandbox);
  const realDF = vm.runInContext('DF', sandbox);
  driveChart3(datedRows, realDF);
  const c3d = plots['chart3'];
  const totalStacked = c3d && c3d.traces.reduce((s, t) => s + t.y.reduce((a, b) => a + b, 0), 0);
  if (c3d && c3d.traces.length === 2 && c3d.traces.every(t => t.y.length === 2) && totalStacked === 3)
    ok('chart3 (dated path): 2 category traces, stacked counts sum to 3 apps');
  else fail(`chart3 dated path invalid (traces=${c3d ? c3d.traces.length : 'none'}, total=${totalStacked})`);
  driveChart3(realDF, null); // restore

  /* ---------------- form submission -> real inference path ---------------- */
  const g = (id) => sandbox.document.getElementById(id);
  const form = g('ratingForm');
  g('rfSize').value = '86';
  g('rfPrice').value = '0';
  g('rfReviews').value = '1200';
  g('rfCategory').value = rows[0].category;
  (form.listeners['submit'] || [])[0]({ preventDefault() {} });
  await tick();
  const rr = elements.get('ratingResult');
  if (rr && rr.innerHTML.includes('predicted rating')) ok('section 07: submit → inference result rendered');
  else fail('rating form submission did not render a result');

  const tform = g('tierForm');
  g('tfSize').value = '50';
  g('tfPrice').value = '';
  g('tfCategory').value = rows[0].category;
  (tform.listeners['submit'] || [])[0]({ preventDefault() {} });
  await tick();
  const tr = elements.get('tierResult');
  if (tr && tr.innerHTML.includes('prob-fill') && tr.innerHTML.includes('model estimates, not guarantees'))
    ok('section 08: submit → tier + probability bars + disclaimer rendered');
  else fail('tier form submission did not render result');

  console.log(failures.length ? `\nSMOKE TEST: ${failures.length} FAILURE(S)` : '\nSMOKE TEST: ALL CHECKS PASSED');
  process.exit(failures.length ? 1 : 0);
})();
