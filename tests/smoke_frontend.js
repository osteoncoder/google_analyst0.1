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
    this.dataset = {};
    this.attrs = {};
  }
  addEventListener(ev, fn) { (this.listeners[ev] = this.listeners[ev] || []).push(fn); }
  insertAdjacentHTML(pos, html) { this.innerHTML += html; }
  focus() {}
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  /* Real elements return a NodeList; the tests only ever iterate it. */
  querySelectorAll() { return { forEach() {} }; }
}
const docListeners = {};
const plots = {};
/* Minimal <script> injection support: ml_dashboard.js / data.js load the
   browser bundles this way (it is the only route a file:// page has). The
   stub reads the real file and executes it in the sandbox, like a browser. */
function makeScriptEl() {
  const el = { tag: 'script', onload: null, onerror: null, _src: '' };
  Object.defineProperty(el, 'src', { get: () => el._src, set: (v) => { el._src = v; } });
  return el;
}
const sandbox = {
  console,
  Date, JSON, Math, Promise, setTimeout, clearTimeout,
  parseInt, parseFloat, isFinite, Number, String, Object, Array, Map, Set, RegExp, Error,
  window: null,
  document: {
    getElementById(id) { if (!elements.has(id)) elements.set(id, new El(id)); return elements.get(id); },
    addEventListener(ev, fn) { (docListeners[ev] = docListeners[ev] || []).push(fn); },
    querySelectorAll() { return { forEach() {} }; },
    createElement(tag) { return makeScriptEl(); },
    head: {
      appendChild(el) {
        try {
          vm.runInContext(fs.readFileSync(path.join(ROOT, el.src), 'utf8'), sandbox, { filename: el.src });
          if (el.onload) el.onload();
        } catch (e) { if (el.onerror) el.onerror(e); }
      },
    },
  },
  fetch: async (url) => {
    if (url === 'data/apps.json') return { ok: true, json: async () => appsJson };
    // A live-backend answer, so the assertions below exercise app.py's route.
    // The static-host section further down switches the API off on purpose.
    if (url === 'api/health') return { ok: true, json: async () => ({ ok: true, models_loaded: true, m1_rating: true, m2_tier_without_reviews: true, trained_on_sample_dataset: true }) };
    if (url === 'api/metrics') return { ok: true, json: async () => metricsJson };
    // static mount serves the artifact file too (used by the degraded-hosting path)
    if (url === 'ml/artifacts/metrics.json') return { ok: true, json: async () => metricsJson };
    if (url === 'api/predict/rating') {
      return { ok: true, json: async () => ({ predicted_rating: 4.2, category_used: 'Education', category_changed: true, model: 'Test', test_metrics: { mae: 0.3, rmse: 0.4, r2: 0.2 }, n_test: 10, assumptions: ['test assumption'], warning: 'test warning' }) };
    }
    if (url === 'api/predict/rating-422') {
      return { ok: false, status: 422, json: async () => ({ detail: [{ loc: ['body', 'size_mb'], msg: 'Input should be greater than or equal to 0' }] }) };
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

/* ---------------- load the frontend files in order ---------------- */
for (const f of ['data.js', 'charts.js', 'ml_inference.js', 'ml_dashboard.js']) {
  const code = fs.readFileSync(path.join(ROOT, f), 'utf8');
  try { vm.runInContext(code, sandbox, { filename: f }); }
  catch (e) { fail(`${f} threw at load: ${e.message}`); process.exit(1); }
}
ok('data.js + charts.js + ml_inference.js + ml_dashboard.js load without errors');

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
  // A candidate that scores well can still be unshippable — the table must show
  // it struck out with its size, not quietly drop it.
  const pBudget = elements.get('perfBudget');
  if (pBudget && /shipping budget/.test(pBudget.innerHTML) && /<s>/.test(pM1.innerHTML))
    ok('section 09: oversized candidates struck out with the shipping-budget reason');
  else fail(`section 09 artifact-budget rejection not surfaced (budget=${pBudget ? pBudget.innerHTML.slice(0, 80) : 'none'})`);
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
  if (rr && rr.innerHTML.includes('Category used: <strong>Education</strong>') && rr.innerHTML.includes('normalized from your input'))
    ok('section 07: normalized category is shown to the user');
  else fail('section 07 should report the category actually used (and that it was normalized)');

  /* error path: FastAPI 422 must render as "field: reason", plus the sample note */
  const origFetch = sandbox.fetch;
  sandbox.fetch = async (url, opts) => url === 'api/predict/rating'
    ? { ok: false, status: 422, json: async () => ({ detail: [{ loc: ['body', 'size_mb'], msg: 'Input should be greater than or equal to 0' }] }) }
    : origFetch(url, opts);
  g('rfSize').value = '-5';
  (form.listeners['submit'] || [])[0]({ preventDefault() {} });
  await tick();
  const rrErr = elements.get('ratingResult').innerHTML;
  sandbox.fetch = origFetch;
  if (rrErr.includes('pred-error') && rrErr.includes('size_mb: Input should be greater than or equal to 0') && !rrErr.includes('[{'))
    ok('section 07: 422 rendered as readable "field: reason" (no raw JSON)');
  else fail(`422 message not rendered readably: ${rrErr.slice(0, 160)}`);
  if (metricsJson.dataset.is_sample && rrErr.includes('warn-note'))
    ok('section 07: sample caveat still shown on the error path');
  else fail('sample caveat missing from the rating error path');

  const tform = g('tierForm');
  g('tfSize').value = '50';
  g('tfPrice').value = '';
  g('tfCategory').value = rows[0].category;
  // The M2 form asks for the inputs a user can actually know (option 2):
  // content rating, minimum Android, and the ad / IAP / Editors' Choice flags.
  // The rating form's 422 case above makes runInference fall through to the
  // browser engine, which flips the recorded mode; reset it so this submit
  // exercises the API route (and its payload) rather than the bundle.
  vm.runInContext('INFERENCE.mode = "api";', sandbox);
  let tierPayload = null;
  const tierFetch = sandbox.fetch;
  sandbox.fetch = async (url, opts) => {
    if (url === 'api/predict/tier') tierPayload = JSON.parse(opts.body);
    return tierFetch(url, opts);
  };
  g('tfContentRating').value = 'Teen';
  g('tfAndroid').value = '8.0';
  g('tfAd').value = '1';
  g('tfIap').value = '0';
  g('tfEditors').value = '';
  (tform.listeners['submit'] || [])[0]({ preventDefault() {} });
  await tick();
  sandbox.fetch = tierFetch;
  const tr = elements.get('tierResult');
  if (tr && tr.innerHTML.includes('prob-fill') && tr.innerHTML.includes('model estimates, not guarantees'))
    ok('section 08: submit → tier + probability bars + disclaimer rendered');
  else fail('tier form submission did not render result');

  const payloadOk = tierPayload && tierPayload.content_rating === 'Teen'
    && tierPayload.min_android === 8 && tierPayload.ad_supported === 1
    && tierPayload.in_app_purchases === 0 && tierPayload.editors_choice === null;
  if (payloadOk) ok('section 08: content rating / min Android / ad / IAP sent; '
    + 'a blank Editors\' Choice stays null (imputed, not guessed)');
  else fail(`tier form payload wrong: ${JSON.stringify(tierPayload)}`);

  /* ---------------- hosting-mode resilience (static / dead backend) ---------------- */
  // Metrics fall back to the static ml/artifacts/metrics.json when api/* is unreachable,
  // so sections 08-09 show the real measured numbers even without the backend.
  const staticOnly = async (url, opts) => /^(\/)?api\//.test(String(url))
    ? Promise.reject(new TypeError('Failed to fetch'))
    : origFetch(url, opts);
  sandbox.fetch = staticOnly;
  const viaStatic = await vm.runInContext('loadMetrics()', sandbox);
  if (viaStatic && viaStatic.url === 'ml/artifacts/metrics.json' && viaStatic.data.models)
    ok('hosting: metrics fall back to the static snapshot when the API is unreachable');
  else fail('metrics static fallback did not resolve ml/artifacts/metrics.json');
  const hint = vm.runInContext('backendHint(new TypeError("Failed to fetch"))', sandbox);
  if (hint.includes('python app.py') && hint.includes('live preview'))
    ok('hosting: failed prediction explains how to start the backend');
  else fail('backendHint should tell the user how to start the backend');
  const unavailableHtml = vm.runInContext(
    '(()=>{const d=document.getElementById("__unavailableProbe"); unavailable(d, "note", new Error("HTTP 503")); return d.innerHTML;})()', sandbox);
  if (unavailableHtml.includes('HTTP 503') && unavailableHtml.includes('retry-btn'))
    ok('hosting: unavailable panel shows the real error and a retry button');
  else fail('unavailable panel should include the error detail and a retry button');

  /* ------- regression: a page with NO working fetch must not show 11 rows -------
     This is the VS Code "Run Active File" / double-click case: fetch() is blocked
     on file://, so the dataset has to come from the <script> bundle instead of
     silently collapsing to the embedded 11-row sample. */
  sandbox.fetch = async () => { throw new TypeError('Failed to fetch'); };
  const offline = await vm.runInContext('loadApps()', sandbox);
  if (offline.mode === 'bundle' && offline.rows.length === rows.length)
    ok(`file:// path: dataset loaded from data/apps_bundle.js (${offline.rows.length.toLocaleString()} rows, not 11)`);
  else fail(`file:// path: expected ${rows.length} bundle rows, got ${offline.rows.length} (${offline.mode})`);

  /* ------- regression: predictions must work with no backend at all ------- */
  const mode = await vm.runInContext('detectInference()', sandbox);
  if (mode === 'browser') ok('hosting: inference falls back to the in-browser exported pipeline');
  else fail(`detectInference() => ${mode} (expected "browser" when no API answers)`);

  g('rfSize').value = '86';
  g('rfPrice').value = '0';
  g('rfReviews').value = '1200';
  g('rfCategory').value = rows[0].category;
  (form.listeners['submit'] || [])[0]({ preventDefault() {} });
  await tick();
  const rrLocal = elements.get('ratingResult').innerHTML;
  if (rrLocal.includes('predicted rating') && rrLocal.includes('in your browser'))
    ok('hosting: rating form predicts with no backend and says the browser engine did it');
  else fail(`browser-engine rating result missing: ${rrLocal.slice(0, 200)}`);

  g('tfSize').value = '50';
  g('tfPrice').value = '';
  g('tfCategory').value = rows[0].category;
  (tform.listeners['submit'] || [])[0]({ preventDefault() {} });
  await tick();
  const trLocal = elements.get('tierResult').innerHTML;
  if (trLocal.includes('prob-fill') && trLocal.includes('in your browser'))
    ok('hosting: tier form predicts with no backend and discloses the engine');
  else fail(`browser-engine tier result missing: ${trLocal.slice(0, 200)}`);

  sandbox.fetch = origFetch;

  /* ---------------- off-canvas nav (hamburger) ---------------- */
  const sidebarEl = g('sidebar'), toggleEl = g('navToggle'), backdropEl = g('navBackdrop');
  const clickToggle = () => (toggleEl.listeners['click'] || [])[0]({});
  const pressEscape = () => (docListeners['keydown'] || []).forEach(fn => fn({ key: 'Escape' }));
  clickToggle();
  if (sidebarEl.classList.contains('open') && toggleEl.getAttribute('aria-expanded') === 'true'
      && backdropEl.hidden === false)
    ok(`nav: toggle opens the drawer (aria-expanded=true, backdrop shown)`);
  else fail(`nav: drawer did not open (open=${sidebarEl.classList.contains('open')}, aria=${toggleEl.getAttribute('aria-expanded')})`);
  (backdropEl.listeners['click'] || [])[0]({});
  if (!sidebarEl.classList.contains('open') && backdropEl.hidden === true)
    ok('nav: clicking the backdrop closes the drawer');
  else fail('nav: backdrop click did not close the drawer');
  clickToggle();
  pressEscape();
  if (!sidebarEl.classList.contains('open') && toggleEl.getAttribute('aria-expanded') === 'false')
    ok('nav: Escape closes the drawer and resets aria-expanded');
  else fail('nav: Escape did not close the drawer');

  /* ---------------- lazy chart rendering ---------------- */
  // Charts must be plotted at most once each, and the timings must be
  // recorded (that is what ?bench=1 prints).
  const plottedOnce = vm.runInContext(`(() => {
    const before = Object.keys(APEX_CHART_TIMES).length;
    const times0 = Object.assign({}, APEX_CHART_TIMES);
    plotChart('chart1'); plotChart('chart1'); plotChart('chart4');
    return { before, after: Object.keys(APEX_CHART_TIMES).length,
             unchanged: JSON.stringify(times0) === JSON.stringify(APEX_CHART_TIMES) };
  })()`, sandbox);
  if (plottedOnce.before >= 6 && plottedOnce.unchanged)
    ok(`charts: each chart is plotted at most once (${plottedOnce.before} already rendered, re-plot is a no-op)`);
  else fail(`charts: re-plotting a chart re-ran the renderer (${JSON.stringify(plottedOnce)})`);

  // With an IntersectionObserver present nothing is plotted until it fires.
  const lazy = vm.runInContext(`(() => {
    const seen = [];
    globalThis.IntersectionObserver = class {
      constructor(cb){ this.cb = cb; globalThis.__io = this; }
      observe(el){ seen.push(el.id); }
      unobserve(){}
    };
    let plottedNames = 0;
    const realPlot = globalThis.Plotly.newPlot;
    globalThis.Plotly.newPlot = function(){ plottedNames++; return realPlot.apply(this, arguments); };
    renderCharts();
    const observedImmediately = seen.slice();
    const plottedBeforeScroll = plottedNames;
    globalThis.__io.cb([{ isIntersecting: true, target: { id: 'chart2' } }]);
    globalThis.Plotly.newPlot = realPlot;
    delete globalThis.IntersectionObserver;
    return { observedImmediately, plottedBeforeScroll };
  })()`, sandbox);
  if (lazy.observedImmediately.length === 6 && lazy.plottedBeforeScroll === 0)
    ok('charts: with IntersectionObserver, nothing renders before it fires (6 observed, 0 plotted)');
  else fail(`charts: lazy render not wired up (${JSON.stringify(lazy)})`);
  await tick();
  if (plots['chart2'] && plots['chart2'].traces.length)
    ok('charts: a chart revealed by the observer is rendered on the next tick');
  else fail('charts: observed chart was never rendered');

  console.log(failures.length ? `\nSMOKE TEST: ${failures.length} FAILURE(S)` : '\nSMOKE TEST: ALL CHECKS PASSED');
  process.exit(failures.length ? 1 : 0);
})();
