/* ================================================================
   ml_dashboard.js — ML sections 07-09 + app bootstrap

   Sections 07/08 perform REAL inference by calling the FastAPI
   backend (app.py), which loads the saved scikit-learn pipelines
   once at startup. No retraining per request. If the backend or
   the artifacts are unavailable, a clear "unavailable" state is
   shown — never random or demo predictions.
   ================================================================ */

let METRICS = null;
let METRICS_FROM_API = false;   // true = live backend; false = static snapshot / unknown

/* Endpoints the read-only metrics may come from, in order of preference:
   1. the FastAPI route (live backend, also proves inference is available)
   2. the same file over the static mount — app.py serves the repo root, and a
      plain static host (e.g. a file viewer) serves it too, so sections 08/09
      can still show the REAL measured metrics with no backend running.
   Predictions always require the live API; this is read-only data only. */
const METRICS_ENDPOINTS = ['api/metrics', '/api/metrics', 'ml/artifacts/metrics.json'];

async function fetchJSON(url, opts){
  const res = await fetch(url, opts);
  if(!res.ok){
    let msg = 'HTTP ' + res.status;
    try{
      const j = await res.json();
      if(j && j.detail){
        if(typeof j.detail === 'string'){
          msg = j.detail;
        }else if(Array.isArray(j.detail)){
          // FastAPI validation errors: render "field: reason" instead of raw JSON
          msg = j.detail.map(d=>{
            const field = (d.loc || []).filter(x=>x !== 'body').join('.') || 'input';
            return `${field}: ${d.msg || 'invalid value'}`;
          }).join(' · ');
        }else{
          msg = JSON.stringify(j.detail);
        }
      }
    }catch(e){ /* keep default */ }
    throw new Error(msg);
  }
  return res.json();
}

/* Fetch the first endpoint that answers. Returns {data, url}. */
async function fetchFirst(urls){
  let lastErr = null;
  for(const u of urls){
    try{
      return { data: await fetchJSON(u), url: u };
    }catch(err){
      lastErr = err;
    }
  }
  throw lastErr || new Error('no endpoint reachable');
}

/* Backend startup is not always instant (and a page can be opened mid-restart),
   so a failed metrics fetch is retried once before the UI gives up. */
async function loadMetrics(){
  let lastErr = null;
  for(let attempt = 0; attempt < 2; attempt++){
    try{
      const hit = await fetchFirst(METRICS_ENDPOINTS);
      return { ...hit, viaApi: hit.url.startsWith('api/') || hit.url.startsWith('/api/') };
    }catch(err){
      lastErr = err;
      if(attempt === 0) await new Promise(r=>setTimeout(r, 1200));
    }
  }
  throw lastErr || new Error('metrics unreachable');
}

/* Actionable hint for a failed inference call. */
function backendHint(err){
  const m = String((err && err.message) || err || '');
  if(/Failed to fetch|NetworkError|Load failed|network/i.test(m)){
    return '<p class="tiny">The model service could not be reached from this page. '
      + 'Predictions need the FastAPI backend: run <code>python app.py</code> and open the served page '
      + '(<code>http://localhost:8000</code>, or the live preview of port 8000). '
      + 'Everything else on this page (charts, metrics tables) works without it.</p>';
  }
  if(/HTTP 503/.test(m)){
    return '<p class="tiny">The backend is running but its model artifacts are missing. '
      + 'Run <code>python clean.py &amp;&amp; python train_models.py</code>, then retry.</p>';
  }
  return '';
}

function wireRetry(el, handler){
  const btn = el.querySelector ? el.querySelector('.retry-btn') : null;
  if(btn && btn.addEventListener) btn.addEventListener('click', handler);
}

function unavailable(el, note, err){
  const detail = err ? ` <span class="tiny">(${String(err.message || err)})</span>` : '';
  el.innerHTML = `
    <div class="unavailable">
      <h3>Model service not available</h3>
      <p>These sections call a small FastAPI backend that loads the saved
      scikit-learn pipelines (no retraining per request). Start it with:</p>
      <pre>pip install -r requirements.txt
python clean.py &amp;&amp; python train_models.py
python app.py</pre>
      <p>Until then no predictions are shown — this project deliberately never
      returns random or demo values when models are absent.</p>
      ${note ? `<p class="tiny">${note}${detail}</p>` : ''}
      <p><button type="button" class="retry-btn">Retry</button></p>
    </div>`;
  wireRetry(el, ()=>renderML());
}

/* Banner when the read-only metrics came from a static file instead of the API:
   the numbers are still the real measured ones, but inference is not available. */
function showMlNotice(){
  const el = document.getElementById('mlNotice');
  if(!el) return;
  if(METRICS && !METRICS_FROM_API){
    el.innerHTML = `<p class="warn-note">⚠ Sections 08–09 below are reading the measured metrics snapshot
    <code>ml/artifacts/metrics.json</code> directly (the live API did not answer). The numbers are the real
    results of the reproducible training run, but the prediction forms need the backend —
    run <code>python app.py</code> and open the page it serves.</p>`;
  }else{
    el.innerHTML = '';
  }
}

/* Honest, dataset-driven caveat: never claim "11-row" when the model was
   trained on the 40k stratified sample (or any other subset). */
function sampleNote(){
  const d = (METRICS && METRICS.dataset) || null;
  if(!d || !d.is_sample) return '';
  const rows = d.rows_cleaned ? Number(d.rows_cleaned).toLocaleString() : 'sample';
  const full = d.full_dataset_rows ? ` of the ${Number(d.full_dataset_rows).toLocaleString()}-row dataset` : '';
  const mechanical = !d.full_dataset_rows && Number(d.rows_cleaned) < 100;
  return `<p class="warn-note">⚠ Trained on a ${rows}-row <strong>sample</strong>${full} — `
    + (mechanical
        ? 'these numbers verify the pipeline mechanically only.'
        : 'results are representative but not full-scale; rare install tiers are deliberately over-sampled '
          + '(see <code>data/playstore_sample.meta.json</code>). Run <code>python fetch_dataset.py</code> '
          + 'and re-run the three commands above for full-scale numbers.')
    + '</p>';
}

function fillCategorySelect(sel){
  const cats = [...new Set(DF.map(d=>d.category))].sort();
  sel.innerHTML = cats.map(c=>`<option value="${c}">${c}</option>`).join('')
    + '<option value="__custom">Other / custom…</option>';
}

function wireCategoryPair(sel, custom){
  fillCategorySelect(sel);
  sel.addEventListener('change', ()=>{
    custom.classList.toggle('hidden', sel.value !== '__custom');
    if(sel.value === '__custom') custom.focus();
  });
}

function catValue(sel, custom){
  return (sel.value === '__custom' ? custom.value : sel.value).trim();
}

/* ---------------- number inputs: empty => null ---------------- */
function numOrNull(input){
  const v = input.value.trim();
  if(v === '') return null;
  const n = Number(v);
  return isFinite(n) ? n : null;
}

/* Counts (reviews) must be integers — rounds rather than sending a fraction
   that the API would reject; mirrors the API's own rounding of counts. */
function intOrNull(input){
  const n = numOrNull(input);
  return n === null ? null : Math.round(n);
}

/* ================================================================
   SECTION 07 — Rating Predictor (M1)
   ================================================================ */
function initRatingForm(){
  const sel = document.getElementById('rfCategory');
  const custom = document.getElementById('rfCategoryCustom');
  const form = document.getElementById('ratingForm');
  const result = document.getElementById('ratingResult');
  wireCategoryPair(sel, custom);

  form.addEventListener('submit', async (e)=>{
    e.preventDefault();
    const cat = catValue(sel, custom);
    if(!cat){
      result.innerHTML = '<div class="pred-error">Please choose or type a category.</div>';
      return;
    }
    result.innerHTML = '<p class="tiny">Predicting…</p>';
    try{
      const out = await fetchJSON('api/predict/rating', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({
          category: cat,
          size_mb: numOrNull(document.getElementById('rfSize')),
          price: numOrNull(document.getElementById('rfPrice')),
          reviews: intOrNull(document.getElementById('rfReviews')),
        }),
      });
      const tm = out.test_metrics || {};
      const catNote = out.category_used
        ? `<p class="tiny">Category used: <strong>${out.category_used}</strong>${out.category_changed ? ' (normalized from your input the same way the training data was)' : ''}.</p>`
        : '';
      result.innerHTML = `
        <div class="pred-big">${out.predicted_rating.toFixed(2)}<span> / 5 predicted rating</span></div>
        <ul class="assump">${out.assumptions.map(a=>`<li>${a}</li>`).join('')}</ul>
        ${catNote}
        <p class="tiny"><strong>${out.model || 'model'}</strong> · held-out test:
        MAE ${tm.mae?.toFixed(3)} · RMSE ${tm.rmse?.toFixed(3)} · R² ${tm.r2?.toFixed(3)}
        (n_test = ${out.n_test ?? '—'}). ${out.warning}</p>`;
    }catch(err){
      result.innerHTML = `<div class="pred-error">Prediction failed: ${err.message}</div>`
        + backendHint(err) + sampleNote();
    }
  });
}

function renderRatingModelCard(){
  const el = document.getElementById('ratingModelCard');
  const m = METRICS.models.m1_rating;
  const tm = m.test_metrics || {};
  el.innerHTML = `
    <h4>Model M1 — rating regression</h4>
    <dl class="kv">
      <dt>Selected model</dt><dd>${m.model} (chosen by ${m.selection_metric})</dd>
      <dt>Target</dt><dd>${m.target} (excluded from inputs)</dd>
      <dt>Inputs</dt><dd>${m.features.join(', ')}</dd>
      <dt>Split</dt><dd>${m.n_train} train / ${m.n_val} val / ${m.n_test} test (grouped by app name)</dd>
      <dt>Held-out test</dt><dd>MAE ${tm.mae?.toFixed(3)} · RMSE ${tm.rmse?.toFixed(3)} · R² ${tm.r2?.toFixed(3)}</dd>
    </dl>
    <p class="tiny">Limitations: cross-sectional snapshot, so this is <em>not</em> a pre-launch or
    future rating predictor; installs are excluded from inputs; weak R² is reported honestly in
    section 09.</p>`;
}

/* ================================================================
   SECTION 08 — Install-Tier Classifier (M2, without Reviews)
   ================================================================ */
function initTierForm(){
  const sel = document.getElementById('tfCategory');
  const custom = document.getElementById('tfCategoryCustom');
  const form = document.getElementById('tierForm');
  const result = document.getElementById('tierResult');
  wireCategoryPair(sel, custom);

  form.addEventListener('submit', async (e)=>{
    e.preventDefault();
    const cat = catValue(sel, custom);
    if(!cat){
      result.innerHTML = '<div class="pred-error">Please choose or type a category.</div>';
      return;
    }
    result.innerHTML = '<p class="tiny">Predicting…</p>';
    try{
      const out = await fetchJSON('api/predict/tier', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({
          category: cat,
          size_mb: numOrNull(document.getElementById('tfSize')),
          price: numOrNull(document.getElementById('tfPrice')),
        }),
      });
      const order = out.tier_order || Object.keys(out.probabilities);
      const bars = order.map(t=>{
        const p = out.probabilities[t] ?? 0;
        return `<div class="prob-row">
          <span class="prob-name">${t}</span>
          <span class="prob-track"><span class="prob-fill" style="width:${(100*p).toFixed(1)}%"></span></span>
          <span class="prob-val">${(100*p).toFixed(1)}%</span>
        </div>`;
      }).join('');
      const tm = out.test_metrics || {};
      const catNote = out.category_used
        ? `<p class="tiny">Category used: <strong>${out.category_used}</strong>${out.category_changed ? ' (normalized from your input the same way the training data was)' : ''}.</p>`
        : '';
      result.innerHTML = `
        <div class="tier-badge">${out.predicted_tier}</div>
        <div class="prob-bars">${bars}</div>
        ${catNote}
        <p class="tiny"><strong>${out.model || 'model'}</strong> (without Reviews) · held-out test:
        accuracy ${tm.accuracy?.toFixed(3)} · macro-F1 ${tm['macro_f1']?.toFixed(3)}
        · weighted-F1 ${tm['weighted_f1']?.toFixed(3)}. ${out.warning}</p>`;
    }catch(err){
      result.innerHTML = `<div class="pred-error">Prediction failed: ${err.message}</div>`
        + backendHint(err) + sampleNote();
    }
  });
}

function renderConfusionMatrix(){
  const el = document.getElementById('chart_confusion');
  const v = METRICS.models.m2_tier.without_reviews;
  const cm = v.test_metrics && v.test_metrics.confusion;
  if(!cm || !cm.matrix){
    el.innerHTML = '<p class="tiny">Confusion matrix not available.</p>';
    return;
  }
  const labels = cm.labels;
  const ann = [];
  cm.matrix.forEach((row,i)=>row.forEach((val,j)=>{
    ann.push({x:labels[j], y:labels[i], text:String(val), showarrow:false,
              font:{color:'#e9e4f7', size:11, family:'JetBrains Mono'}});
  }));
  Plotly.newPlot(el, [{
    type:'heatmap', z:cm.matrix, x:labels, y:labels,
    colorscale:[[0,'#181432'],[1,'#a855f7']],
    hovertemplate:'true %{y} → predicted %{x}<br>%{z} apps<extra></extra>',
    colorbar:{tickfont:{color:'#8f86ac'}, len:0.8, thickness:14, outlinewidth:0},
    xgap:2, ygap:2,
  }], {
    paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
    font:{family:'Inter, sans-serif', color:'#b1a8cf', size:11},
    margin:{t:16,l:96,r:24,b:96},
    xaxis:{side:'top', tickangle:0, tickfont:{color:'#8f86ac'}, title:{text:'Predicted tier', font:{color:'#8f86ac', size:11}}},
    yaxis:{autorange:'reversed', tickfont:{color:'#8f86ac'}, title:{text:'True tier', font:{color:'#8f86ac', size:11}}},
    annotations:ann,
  }, CONFIG);
  const note = document.getElementById('confusionNote');
  if(note){
    const n = cm.matrix.reduce((s,r)=>s+r.reduce((a,b)=>a+b,0),0);
    note.textContent = `Held-out test set (n = ${n}), final model evaluated once. `
      + `Row = true tier, column = predicted tier. Class counts in full data: `
      + (v.class_counts_full_data ? Object.entries(v.class_counts_full_data).map(([k,vv])=>`${k} ${vv}`).join(' · ') : '—') + '.';
  }
}

/* ================================================================
   SECTION 09 — Model Performance
   ================================================================ */
function tableHTML(headers, rows, selectedRow){
  const thead = '<tr>' + headers.map((h,i)=>`<th class="${i===0?'':'num'}">${h}</th>`).join('') + '</tr>';
  const tbody = rows.map((r,i)=>`<tr class="${i===selectedRow?'selected':''}">`
    + r.map((c,j)=>`<td class="${j===0?'':'num'}">${c ?? '—'}</td>`).join('') + '</tr>').join('');
  return `<table class="metric-table"><thead>${thead}</thead><tbody>${tbody}</tbody></table>`;
}

function renderPerfM1(){
  const el = document.getElementById('perfM1');
  const m = METRICS.models.m1_rating;
  const rows = Object.entries(m.candidates).map(([name, r])=>{
    if(!r.val) return [name, 'FAILED', r.error, '', '', ''];
    return [name,
      r.val.mae?.toFixed(3), r.val.rmse?.toFixed(3), r.val.r2?.toFixed(3),
      r.test.mae?.toFixed(3), r.test.rmse?.toFixed(3), r.test.r2?.toFixed(3)];
  });
  el.innerHTML = tableHTML(
    ['Model','MAE (val)','RMSE (val)','R² (val)','MAE (test)','RMSE (test)','R² (test)'],
    rows, rows.findIndex(r=>r[0]===m.model));
}

function renderPerfM2(version){
  const el = document.getElementById(version === 'with_reviews' ? 'perfM2a' : 'perfM2b');
  const m = METRICS.models.m2_tier[version];
  const rows = Object.entries(m.candidates).map(([name, r])=>{
    if(!r.val) return [name, 'FAILED', r.error, '', '', ''];
    return [name, r.val.accuracy?.toFixed(3), r.val['macro_f1']?.toFixed(3),
            r.test.accuracy?.toFixed(3), r.test['macro_f1']?.toFixed(3), r.test['weighted_f1']?.toFixed(3)];
  });
  el.innerHTML = tableHTML(
    ['Model','Acc (val)','Macro-F1 (val)','Acc (test)','Macro-F1 (test)','Weighted-F1 (test)'],
    rows, rows.findIndex(r=>r[0]===m.model));
}

function renderPerClass(){
  const el = document.getElementById('perfPerClass');
  const m = METRICS.models.m2_tier.without_reviews;
  const pc = (m.test_metrics && m.test_metrics.per_class) || {};
  const rows = m.labels.map(l=>{
    const o = pc[l] || {};
    return [l, o.precision?.toFixed(3), o.recall?.toFixed(3), o.f1?.toFixed(3), o.support ?? 0];
  });
  el.innerHTML = tableHTML(['Class (test support)','Precision','Recall','F1','Support'], rows, -1);
}

function renderImportance(){
  const el = document.getElementById('chart_importance');
  const m = METRICS.models.m2_tier.without_reviews;
  const imps = (m.rf_feature_importances || []).slice(0,12).reverse();
  if(!imps.length){
    el.innerHTML = '<p class="tiny">Random Forest importances not available (candidate did not train).</p>';
    return;
  }
  Plotly.newPlot(el, [{
    x:imps.map(p=>p[1]), y:imps.map(p=>p[0]),
    type:'bar', orientation:'h',
    marker:{color:imps.map((_,i)=>PURPLE_SCALE[(i%3===0)?3:(i%2?1:0)]), line:{width:0}},
    text:imps.map(p=>p[1].toFixed(3)), textposition:'outside',
    textfont:{color:'#8f86ac', family:'JetBrains Mono', size:10},
    hovertemplate:'%{y}<br>importance %{x:.4f}<extra></extra>',
  }], {
    paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
    font:{family:'Inter, sans-serif', color:'#b1a8cf', size:11},
    margin:{t:36,l:170,r:44,b:36},
    xaxis:{gridcolor:'rgba(168,85,247,0.08)', zerolinecolor:'rgba(168,85,247,0.15)',
           linecolor:'rgba(168,85,247,0.2)', tickfont:{color:'#8f86ac'},
           title:{text:'Feature importance (Random Forest candidate)', font:{color:'#8f86ac', size:11}}},
    yaxis:{gridcolor:'rgba(168,85,247,0.08)', linecolor:'rgba(168,85,247,0.2)', tickfont:{color:'#b1a8cf', size:11}},
    annotations:[{x:0.02, y:1.12, xref:'paper', yref:'paper', showarrow:false,
      text:m.importances_from || 'Random Forest', font:{color:'#8f86ac', family:'JetBrains Mono', size:10}}],
  }, CONFIG);
}

function renderCompare(){
  const el = document.getElementById('perfCompare');
  const a = METRICS.models.m2_tier.with_reviews;
  const b = METRICS.models.m2_tier.without_reviews;
  const ta = a.test_metrics || {}, tb = b.test_metrics || {};
  const dAcc = (ta.accuracy??0) - (tb.accuracy??0);
  const dMac = (ta['macro_f1']??0) - (tb['macro_f1']??0);
  el.innerHTML = `
    ${tableHTML(['Version','Accuracy (test)','Macro-F1 (test)','Weighted-F1 (test)'],
      [
        ['A — with Reviews', ta.accuracy?.toFixed(3), ta['macro_f1']?.toFixed(3), ta['weighted_f1']?.toFixed(3)],
        ['B — without Reviews (primary)', tb.accuracy?.toFixed(3), tb['macro_f1']?.toFixed(3), tb['weighted_f1']?.toFixed(3)],
      ], -1)}
    <p class="tiny">Δ (A − B): accuracy ${dAcc>=0?'+':''}${dAcc.toFixed(3)}, macro-F1 ${dMac>=0?'+':''}${dMac.toFixed(3)}.
    Reviews is a strong proxy for installs (apps with more installs accumulate more reviews), so
    including it inflates fit — version B is the honest primary model. An accuracy drop alone does
    <em>not</em> prove temporal leakage, and removing Reviews does not make every remaining feature
    leakage-free (e.g. category and size can still carry popularity information).</p>`;
}

function renderSummary(){
  const el = document.getElementById('perfSummary');
  const d = METRICS.dataset || {};
  const a = METRICS.models.m2_tier.without_reviews;
  const m1 = METRICS.models.m1_rating;
  el.innerHTML = `
    <dl class="kv">
      <dt>Dataset</dt><dd>${d.source || '—'} (${d.rows_cleaned ?? '—'} cleaned rows${d.is_sample ? ', SAMPLE' : ''}) · md5 ${d.md5 || '—'}${d.sample_note ? `<br><span class="tiny">${d.sample_note}</span>` : ''}</dd>
      <dt>Split</dt><dd>80/20 train/test before any preprocessing; GroupShuffleSplit on app name (same app never on both sides); 75/25 train/val for selection; test used exactly once</dd>
      <dt>Seed</dt><dd>${METRICS.seed}</dd>
      <dt>M1 inputs</dt><dd>${m1.features.join(', ')} → target ${m1.target}</dd>
      <dt>M2 inputs (B)</dt><dd>${a.features.join(', ')} → 4 install bands</dd>
      <dt>Tier bounds</dt><dd>${(a.tier_bounds||[]).map(([lo,hi,n])=>`${n}: [${lo.toLocaleString()}, ${hi===null?'∞':hi.toLocaleString()})`).join('; ')}</dd>
      <dt>Class counts (full data)</dt><dd>${Object.entries(a.class_counts_full_data||{}).map(([k,v])=>`${k} ${v}`).join(' · ')}</dd>
    </dl>
    <ul class="limit-list">
      <li>Installs are reported download-band <strong>lower bounds</strong>, not exact downloads.</li>
      <li>Listed price is a price tag, <strong>not observed revenue</strong>; no revenue is estimated anywhere.</li>
      <li>Neither model is a pre-launch or future-growth predictor: training data is a cross-sectional store snapshot and evaluation is in-sample-time.</li>
      <li>Unknown categories at prediction time are encoded as “not seen in training”.</li>
      ${d.is_sample ? `<li>Current numbers are from a ${d.rows_cleaned ? Number(d.rows_cleaned).toLocaleString() : ''}-row sample${d.full_dataset_rows ? ` of the ${Number(d.full_dataset_rows).toLocaleString()}-row dataset` : ''} — representative, not full-scale.</li>` : ''}
    </ul>`;
}

function renderML(){
  initRatingForm();
  initTierForm();
  (async ()=>{
    try{
      const hit = await loadMetrics();
      METRICS = hit.data;
      METRICS_FROM_API = hit.viaApi;
    }catch(err){
      METRICS = null;
      METRICS_FROM_API = false;
      showMlNotice();
      const note = 'Backend reachable but no artifacts yet? Run: python clean.py && python train_models.py';
      ['ratingResult','tierResult'].forEach(id=>unavailable(document.getElementById(id), note, err));
      ['ratingModelCard','confusionCard','perfContent'].forEach(id=>{
        const e = document.getElementById(id);
        if(e) unavailable(e, note, err);
      });
      return;
    }
    showMlNotice();
    renderRatingModelCard();
    renderConfusionMatrix();
    renderPerfM1();
    renderPerfM2('with_reviews');
    renderPerfM2('without_reviews');
    renderPerClass();
    renderImportance();
    renderCompare();
    renderSummary();
  })();
}

/* ================================================================
   BOOTSTRAP
   ================================================================ */
function initNav(){
  const links = document.querySelectorAll('.nav-list a');
  window.addEventListener('scroll', ()=>{
    let current = 'overview';
    document.querySelectorAll('section.viz-section').forEach(sec=>{
      if(window.scrollY >= sec.offsetTop - 160) current = sec.id;
    });
    links.forEach(a=>{
      a.classList.toggle('active', a.getAttribute('href') === '#'+current);
    });
  }, {passive:true});
}

function bootstrap(){
  initNav();
  loadApps().then(({rows, source, is_sample})=>{
    DF = rows;
    APP_SOURCE = source;
    APP_IS_SAMPLE = is_sample;
    document.getElementById('rowCount').textContent = rows.length.toLocaleString();
    document.getElementById('runTime').textContent = source;
    const fs = document.getElementById('footSource');
    if(fs) fs.textContent = rows.length.toLocaleString() + ' cleaned rows · ' + source;
    renderKPIs();
    renderCharts();
    renderML();
  });
}

document.addEventListener('DOMContentLoaded', bootstrap);
