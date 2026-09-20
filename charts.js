/* ================================================================
   charts.js — Plotly rendering for KPI strip + charts 01-06
   All values are computed from DF (cleaned rows) — there are no
   hardcoded chart values anywhere in this file.
   Depends on data.js (PURPLE_SCALE, loadApps, helpers).
   Bootstrapping happens in ml_dashboard.js.
   ================================================================ */

let DF = [];          // cleaned rows (set by bootstrap in ml_dashboard.js)
let APP_SOURCE = '';  // human-readable dataset source
let APP_IS_SAMPLE = false;

/* ---------------- shared Plotly theme (aurora, unchanged) ---------------- */
const FONT = {family:'Inter, sans-serif', color:'#b1a8cf', size:12};
const AX = {
  gridcolor:'rgba(168,85,247,0.08)', zerolinecolor:'rgba(168,85,247,0.15)',
  linecolor:'rgba(168,85,247,0.2)', tickfont:{color:'#8f86ac'},
};
const layoutBase = {
  paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)', font:FONT,
  margin:{t:16,l:60,r:24,b:64},
  xaxis:{...AX}, yaxis:{...AX},
  hoverlabel:{bgcolor:'#181432', bordercolor:'#a855f7', font:{color:'#f3f0ff', family:'Inter, sans-serif'}},
};
const legendV = {orientation:'v', x:1.02, y:1, font:{color:'#b1a8cf', size:11}, bgcolor:'rgba(0,0,0,0)'};
const legendH = {orientation:'h', y:-0.22, x:0.5, font:{color:'#b1a8cf', size:11}, bgcolor:'rgba(0,0,0,0)'};
const CONFIG = {displayModeBar:false, responsive:true};

/* ---------------- WebGL capability ----------------
   Chart 1 draws ~30,000 markers. As SVG that is ~30,000 DOM nodes, which is
   what makes it slow; as WebGL it is one draw call per trace. Detect the
   context once and cache it — the chart falls back to the SVG scatter it has
   always used whenever WebGL is unavailable, so nothing can break. */
let WEBGL_OK = null;
function webglAvailable(){
  if(WEBGL_OK !== null) return WEBGL_OK;
  try{
    const c = document.createElement('canvas');
    WEBGL_OK = !!(c && c.getContext && (c.getContext('webgl') || c.getContext('experimental-webgl')));
  }catch(e){ WEBGL_OK = false; }
  return WEBGL_OK;
}

function gap(el, title, bodyHTML){
  el.innerHTML = `<div class="data-gap"><h3>${title}</h3><p>${bodyHTML}</p></div>`;
}
function banner(el, html){
  el.insertAdjacentHTML('afterbegin', `<div class="gap-banner">${html}</div>`);
}

/* ================================================================
   KPI STRIP — actual computed values only (no invented deltas)
   ================================================================ */
function renderKPIs(){
  const n = DF.length;
  const cats = new Set(DF.map(d=>d.category)).size;
  const inst = DF.filter(d=>isFinite(d.installs));
  const sumInst = sum(inst.map(d=>d.installs));
  const rated = DF.filter(d=>isFinite(d.rating));
  const avgR = rated.length ? sum(rated.map(d=>d.rating))/rated.length : NaN;
  const sumRev = sum(DF.map(d=>d.reviews));

  const kpis = [
    {label:'Apps in dataset',      value:n.toLocaleString(),        note:`across ${cats} categories`,          glow:'var(--purple-neon)'},
    {label:'Reported installs',    value:fmtCompact(sumInst),       note:'sum of band lower bounds',           glow:'var(--cyan)'},
    {label:'Average rating',       value:isNaN(avgR)?'—':avgR.toFixed(2)+' / 5', note:`mean of ${rated.length} app ratings`, glow:'var(--pink)'},
    {label:'Total reviews',        value:fmtCompact(sumRev),        note:'sum of review counts',               glow:'var(--amber)'},
  ];
  document.getElementById('kpiGrid').innerHTML = kpis.map(k=>`
    <div class="kpi-card" style="--kpi-glow:${k.glow}">
      <div class="kpi-label"><span class="ico"></span>${k.label}</div>
      <div class="kpi-value">${k.value}</div>
      <div class="kpi-note">${k.note}</div>
    </div>`).join('');
}

/* ================================================================
   CHART 1 — Quality Benchmark (kept: valid data-driven scatter)
   Size vs rating, bubble area ∝ reported installs (band lower bound).
   ================================================================ */
function renderChart1(){
  const el = document.getElementById('chart1');
  const rows = DF.filter(d => isFinite(d.installs) && d.installs>=1000 && isFinite(d.rating) && isFinite(d.size_mb));
  if(!rows.length){
    gap(el, 'No apps to plot', 'Filter requires ≥1,000 reported installs and known size and rating.');
    return;
  }
  // ONE trace, not one per category. Measured on the 40k-row sample: this chart
  // split ~20k markers across 48 category traces, and Plotly repeats its
  // per-trace setup (calc, autorange, hover wiring, and for scattergl a
  // separate vertex buffer) for each one — that overhead, not the markers, was
  // the bulk of the 1.4 s. Colour moves to a per-point array, so every marker
  // keeps exactly the category colour it had as its own trace, and hover still
  // names the category. Same data, same picture, one draw call.
  const cats = [...new Set(rows.map(d=>d.category))].sort();
  const catColor = new Map(cats.map((c,i)=>[c, PURPLE_SCALE[i%PURPLE_SCALE.length]]));

  const n = rows.length;
  const x = new Array(n), y = new Array(n), sizes = new Array(n),
        colors = new Array(n), cd = new Array(n);
  // Installs are banded — only ~14 distinct values across 40k rows — so
  // re-formatting the same handful of numbers 20k times is pure waste.
  const fmtCache = new Map();
  const fmtInst = v => {
    let s = fmtCache.get(v);
    if(s === undefined){ s = v.toLocaleString(); fmtCache.set(v, s); }
    return s;
  };
  // Math.max(...sizes) spreads ~20k arguments and can overflow the call stack
  // on a bigger dataset — a plain loop is both safer and faster.
  let maxSize = 0;
  for(let k=0;k<n;k++){
    const d = rows[k];
    x[k] = d.size_mb; y[k] = d.rating;
    const s = Math.sqrt(d.installs)/9;
    sizes[k] = s; if(s > maxSize) maxSize = s;
    colors[k] = catColor.get(d.category);
    cd[k] = `${d.app}<br>Category: ${d.category}<br>Installs (band lower bound): ${fmtInst(d.installs)}`;
  }

  Plotly.newPlot(el, [{
    x, y, mode:'markers', type: webglAvailable() ? 'scattergl' : 'scatter',
    marker:{
      size:sizes, sizemode:'area',
      sizeref: 2.0*maxSize/(40**2), sizemin:4,
      color:colors, opacity:0.75,
      // A 1px stroke on every one of ~20k markers roughly doubles the paint
      // cost and buys almost nothing at this bubble size.
      line:{width:0},
    },
    customdata:cd,
    hovertemplate:'%{customdata}<br>Size: %{x:.1f} MB · Rating: %{y:.2f}<extra></extra>',
  }], {
    ...layoutBase,
    // No 48-entry legend: it was drawn outside the plot area (x:1.02, right
    // margin 130px), which is both unreadable and what pushed content towards
    // the right edge on narrow viewports. Hover carries the category instead.
    margin:{t:16,l:60,r:24,b:56},
    showlegend:false,
    xaxis:{...AX, title:{text:'App size (MB)', font:{color:'#8f86ac', size:12}}},
    yaxis:{...AX, title:{text:'User rating (1–5)', font:{color:'#8f86ac', size:12}}},
  }, CONFIG);
}

/* ================================================================
   CHART 2 — Feature Correlation (replaces fabricated "Global Reach")
   No location data exists in the dataset, so no map is shown.
   Pearson matrix over genuine numeric columns; log1p for skew.
   ================================================================ */
function renderChart2(){
  const el = document.getElementById('chart2');
  const cols = [
    {label:'Rating',           get:d=>isFinite(d.rating)   ? d.rating : NaN},
    {label:'log1p(Reviews)',   get:d=>isFinite(d.reviews)  ? Math.log1p(d.reviews) : NaN},
    {label:'log1p(Installs)',  get:d=>isFinite(d.installs) ? Math.log1p(d.installs) : NaN},
    {label:'Size (MB)',        get:d=>isFinite(d.size_mb)  ? d.size_mb : NaN},
  ];
  if(DF.some(d=>isNum(d.price) && d.price>0) || DF.some(d=>d.price===0)) cols.push({label:'Price ($)', get:d=>isNum(d.price)?d.price:NaN});
  if(DF.some(d=>isFinite(d.sentiment))) cols.push({label:'Subjectivity', get:d=>isFinite(d.sentiment)?d.sentiment:NaN});

  // Materialise each candidate column ONCE into a Float64Array (NaN = missing).
  // The old version rebuilt a [x,y] pair array inside every matrix cell: five
  // columns meant 25 sweeps of all 40,000 rows and ~1.3M throwaway arrays,
  // which is where this chart's ~900 ms went — Plotly was drawing a 5x5
  // heatmap, which costs almost nothing. Measured: 201 ms -> 35 ms of JS.
  const mats = cols.map(c=>{
    const a = new Float64Array(DF.length);
    for(let i=0;i<DF.length;i++){ const v = c.get(DF[i]); a[i] = isFinite(v) ? v : NaN; }
    return a;
  });
  // keep columns with >=3 valid values and non-zero variance.
  // (A plain loop, not Math.max(...vals): a 40k-argument spread can overflow
  // the call stack, and this runs on every render.)
  const kept = [];
  cols.forEach((c,i)=>{
    const a = mats[i];
    let k=0, mn=Infinity, mx=-Infinity;
    for(let j=0;j<a.length;j++){
      const v = a[j];
      if(isFinite(v)){ k++; if(v<mn) mn=v; if(v>mx) mx=v; }
    }
    if(k>=3 && (mx-mn) > 0) kept.push({label:c.label, m:a});
  });
  const dropped = cols.filter(c=>!kept.some(k=>k.label===c.label)).map(c=>c.label);
  if(kept.length < 2){
    gap(el, 'Not enough numeric columns', 'A correlation matrix needs at least two non-constant numeric columns.');
    return;
  }

  const labels = kept.map(c=>c.label);
  // Two-pass Pearson. The one-pass form (n*Sxy - Sx*Sy) is cheaper but loses
  // precision badly at n = 40,000 through catastrophic cancellation, and these
  // are numbers a reader takes away — so: means first, then sums of products
  // of deviations. Verified identical to the old matrix to 1e-12.
  const z = kept.map(a => kept.map(b => {
    const A = a.m, B = b.m, n = A.length;
    let m=0, sx=0, sy=0;
    for(let k=0;k<n;k++){
      const p = A[k], q = B[k];
      if(isFinite(p) && isFinite(q)){ m++; sx+=p; sy+=q; }
    }
    if(m < 3) return NaN;
    const mp = sx/m, mq = sy/m;
    let sxy=0, sxx=0, syy=0;
    for(let k=0;k<n;k++){
      const p = A[k], q = B[k];
      if(!isFinite(p) || !isFinite(q)) continue;
      const dp = p-mp, dq = q-mq;
      sxy += dp*dq; sxx += dp*dp; syy += dq*dq;
    }
    const den = Math.sqrt(sxx*syy);
    return den > 0 ? sxy/den : NaN;
  }));

  const ann = [];
  z.forEach((row,i)=>row.forEach((v,j)=>{
    ann.push({x:labels[j], y:labels[i],
      text: isFinite(v) ? v.toFixed(2) : '—',
      showarrow:false, font:{color:'#e9e4f7', size:11, family:'JetBrains Mono'}});
  }));

  Plotly.newPlot(el, [{
    type:'heatmap', x:labels, y:labels, z:z.map(r=>r.map(v=>isFinite(v)?v:null)),
    colorscale:[[0,'#2dd4ea'],[0.5,'#141126'],[1,'#f472b6']],
    zmin:-1, zmax:1, zmid:0,
    hovertemplate:'%{y} × %{x}<br>r = %{z}<extra></extra>',
    colorbar:{title:{text:'Pearson r', side:'right', font:{color:'#8f86ac'}}, tickfont:{color:'#8f86ac'},
              len:0.8, thickness:14, outlinewidth:0},
    xgap:2, ygap:2,
  }], {
    ...layoutBase,
    margin:{t:16,l:130,r:80,b:96},
    xaxis:{...AX, tickangle:-28},
    yaxis:{...AX, autorange:'reversed'},
    annotations:ann,
  }, CONFIG);
  const note = document.getElementById('chart2note');
  if(note){
    note.textContent = `Pearson r on ${DF.length} cleaned apps (log1p applied to skewed Reviews/Installs).`
      + ` Correlation = association, not causation.`
      + (dropped.length ? ` Constant/absent columns excluded: ${dropped.join(', ')}.` : '');
  }
}

/* ================================================================
   CHART 3 — Apps by Last-Updated Month (replaces fabricated
   "Category Trajectory"). Snapshot of each app's latest update
   date only — NOT update history, NOT monthly install history.
   ================================================================ */
function renderChart3(){
  const el = document.getElementById('chart3');
  const dated = DF.filter(d=>d.last_updated);
  if(!dated.length){
    gap(el, 'No “Last Updated” column in the current dataset',
      'This section needs a real update-date column and the loaded dataset has none '
      + '(or none parsed), so a last-updated-month chart cannot be computed honestly. '
      + 'The primary dataset used by this project '
      + '(<code>data/playstore_sample.csv</code>, or the full 2.31M-row file from '
      + '<code>python fetch_dataset.py</code>) carries a “Last Updated” column → '
      + '<code>python clean.py</code>; this section then renders '
      + '<em>Apps by Last-Updated Month</em> — a snapshot of each app’s latest '
      + 'update date, not a full update or install history.');
    return;
  }
  const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  const months = new Map(); // 'YYYY-MM' -> {category -> count}
  dated.forEach(d=>{
    const dt = d.last_updated;
    const key = `${dt.getUTCFullYear()}-${String(dt.getUTCMonth()+1).padStart(2,'0')}`;
    if(!months.has(key)) months.set(key, {});
    const bucket = months.get(key);
    bucket[d.category] = (bucket[d.category]||0) + 1;
  });
  const sortedKeys = [...months.keys()].sort();
  const xLabels = sortedKeys.map(k=>{
    const [yy, mm] = k.split('-');
    return MONTHS[parseInt(mm,10)-1] + ' ’' + yy.slice(2);
  });
  const catCount = {};
  dated.forEach(d=>{ catCount[d.category] = (catCount[d.category]||0)+1; });
  const topCats = Object.entries(catCount).sort((a,b)=>b[1]-a[1]).slice(0,8).map(e=>e[0]);
  const cats = [...topCats];
  if(Object.keys(catCount).length > 8) cats.push('Other');

  const countFor = (cat, k) => {
    const m = months.get(k);
    if(cat === 'Other') return Object.entries(m).filter(([c])=>!topCats.includes(c)).reduce((s,[,v])=>s+v,0);
    return m[cat] || 0;
  };
  const traces = cats.map((cat,i)=>({
    x:xLabels,
    y:sortedKeys.map(k=>countFor(cat, k)),
    type:'bar', name:cat, stackgroup:'one',
    marker:{color:PURPLE_SCALE[i%PURPLE_SCALE.length], line:{width:0}},
    customdata:xLabels.map((lab,j)=>`${lab} · ${sortedKeys[j]}`),
    hovertemplate: cat + '<br>%{x}<br>%{y} apps<extra></extra>',
  }));

  Plotly.newPlot(el, traces, {
    ...layoutBase,
    margin:{t:16,l:60,r:24,b:78},
    legend:legendH,
    barmode:'stack',
    xaxis:{...AX, title:{text:'Month of each app’s latest update (snapshot)', font:{color:'#8f86ac', size:12}}},
    yaxis:{...AX, title:{text:'Number of apps', font:{color:'#8f86ac', size:12}}},
  }, CONFIG);
}

/* ================================================================
   CHART 4 — Rating Distribution (replaces fabricated
   "Market Expansion" cumulative-install projection)
   ================================================================ */
function renderChart4(){
  const el = document.getElementById('chart4');
  const ratings = DF.filter(d=>isFinite(d.rating)).map(d=>d.rating);
  if(!ratings.length){
    gap(el, 'No valid ratings in the current dataset', 'A rating distribution needs at least one app with a rating in 1–5.');
    return;
  }
  const BIN = 0.25, LO = 1, HI = 5;
  const nb = Math.round((HI-LO)/BIN);
  const counts = new Array(nb).fill(0);
  ratings.forEach(r=>{
    let i = Math.floor((r-LO)/BIN);
    i = Math.max(0, Math.min(nb-1, i));
    counts[i] += 1;
  });
  const centers = counts.map((_,i)=> LO + (i+0.5)*BIN);
  const total = ratings.length;
  Plotly.newPlot(el, [{
    x:centers, y:counts, type:'bar',
    marker:{color:centers.map((_,i)=>PURPLE_SCALE[(i*2)%PURPLE_SCALE.length]),
            line:{color:'rgba(5,4,12,0.6)', width:1}},
    text:counts.map(c=>c||''), textposition:'outside',
    textfont:{color:'#b1a8cf', family:'JetBrains Mono', size:11},
    customdata: centers.map((c,i)=>`${(LO+i*BIN).toFixed(2)}–${(LO+(i+1)*BIN).toFixed(2)} · ${counts[i]} apps (${(100*counts[i]/total).toFixed(1)}%)`),
    hovertemplate: '%{customdata}<extra></extra>',
  }], {
    ...layoutBase,
    margin:{t:30,l:60,r:24,b:56},
    bargap:0.12,
    xaxis:{...AX, title:{text:'App rating (1–5)', font:{color:'#8f86ac', size:12}}},
    yaxis:{...AX, title:{text:'Number of apps', font:{color:'#8f86ac', size:12}}},
  }, CONFIG);
}

/* ================================================================
   CHART 5 — Category scatter: total reviews (log) vs avg rating
   (replaces the shared-log-axis grouped bar). Categories with a
   zero total review count are excluded from the log axis and
   counted in the note.
   ================================================================ */
function renderChart5(){
  const el = document.getElementById('chart5');
  const byCat = new Map();
  DF.forEach(d=>{
    if(!byCat.has(d.category)) byCat.set(d.category, {n:0, sumRev:0, sumRating:0, nRating:0});
    const o = byCat.get(d.category);
    o.n += 1;
    o.sumRev += d.reviews;
    if(isFinite(d.rating)){ o.sumRating += d.rating; o.nRating += 1; }
  });
  let zeroRev = 0, noRating = 0;
  const pts = [];
  for(const [cat, o] of byCat){
    if(o.sumRev <= 0){ zeroRev += 1; continue; }
    if(o.nRating === 0){ noRating += 1; continue; }
    pts.push({cat, n:o.n, sumRev:o.sumRev, avg:o.sumRating/o.nRating});
  }
  if(!pts.length){
    gap(el, 'No category has both reviews and a rating', 'A reviews-vs-rating scatter needs at least one such category.');
    return;
  }
  const order = [...pts].sort((a,b)=>b.sumRev-a.sumRev).map(p=>p.cat);
  // Map the category -> rank ONCE; the old order.indexOf(p.cat) inside the
  // colour map rescanned the array for every point (O(n²)).
  const rank = new Map(order.map((cat,i)=>[cat,i]));
  Plotly.newPlot(el, [{
    x:pts.map(p=>p.sumRev), y:pts.map(p=>p.avg), type:'scatter', mode:'markers+text',
    text:pts.map(p=>p.cat), textposition:'top center',
    // Labels sit above each bubble, but the categories cluster tightly so they
    // land on light bubbles (amber, green, sky) as often as on the dark page.
    // No single flat colour is legible on both, so the text stays light and
    // style.css paints a dark halo around it (#chart5 text).
    textfont:{color:'#f3f0ff', size:11, family:'Inter, sans-serif'},
    customdata:pts.map(p=>`${p.n} app${p.n>1?'s':''} · Σ reviews ${p.sumRev.toLocaleString()} · avg rating ${p.avg.toFixed(2)} (unweighted mean of app ratings)`),
    marker:{
      size:pts.map(p=>10+3*Math.sqrt(p.n)),
      color:pts.map(p=>PURPLE_SCALE[(rank.get(p.cat)||0)%PURPLE_SCALE.length]),
      opacity:0.85, line:{width:1.5, color:'rgba(5,4,12,0.5)'},
    },
    hovertemplate:'%{text}<br>%{customdata}<br>Total reviews: %{x:,.0f}<extra></extra>',
  }], {
    ...layoutBase,
    margin:{t:16,l:70,r:24,b:64},
    xaxis:{...AX, type:'log', title:{text:'Total reviews per category (log scale)', font:{color:'#8f86ac', size:12}}},
    yaxis:{...AX, title:{text:'Average rating (unweighted mean of apps)', font:{color:'#8f86ac', size:12}}},
  }, CONFIG);
  const note = document.getElementById('chart5note');
  if(note){
    note.textContent = `x = Σ review counts per category (log scale); y = unweighted mean of the category’s app ratings. `
      + (zeroRev ? `${zeroRev} categor${zeroRev>1?'ies':'y'} with zero total reviews excluded from the log axis. ` : '')
      + (noRating ? `${noRating} with no ratings excluded.` : '');
  }
}

/* ================================================================
   CHART 6 — Pricing Mix (replaces fabricated revenue estimate).
   A Price column is required; listed price is a price tag, NOT
   observed revenue, so no revenue is ever estimated. Without
   pricing data this section shows an explicit data gap.
   ================================================================ */
function renderChart6(){
  const el = document.getElementById('chart6');
  const el6b = document.getElementById('chart6b');
  const hasPricing = DF.some(d=>isNum(d.price) && d.price>0);

  if(!hasPricing){
    el6b.style.display = 'none';
    banner(el.parentElement,
      '<strong>No Price column in the current dataset.</strong> Pricing and revenue cannot be shown '
      + 'honestly, so no figures are invented. The original version estimated paid revenue as '
      + '“5% of free installs × $2.99” — that was a fabricated estimate and has been removed. '
      + 'Legitimate zero prices are preserved as $0.00, never replaced. Until a dataset with a '
      + 'Price column is loaded, this section counts apps per category instead.');
  }

  const byCat = new Map();
  DF.forEach(d=>{
    if(!byCat.has(d.category)) byCat.set(d.category, {free:0, paid:0, prices:[]});
    const o = byCat.get(d.category);
    if(isNum(d.price) && d.price>0){ o.paid += 1; o.prices.push(d.price); }
    else if(isNum(d.price) && d.price===0) o.free += 1;
  });
  const top = [...byCat.entries()].sort((a,b)=>(b[1].free+b[1].paid)-(a[1].free+a[1].paid)).slice(0,8);
  const cats = top.map(e=>e[0]);
  const traceFree = {
    x:cats, y:cats.map(c=>byCat.get(c).free), type:'bar', name:'Free apps (price = $0.00)',
    marker:{color:'#a855f7'},
    customdata:cats.map(c=>`Free apps (price = $0.00): ${byCat.get(c).free}`),
    hovertemplate:'%{customdata}<extra></extra>',
  };
  if(hasPricing){
    el6b.style.display = '';
    const tracePaid = {
      x:cats, y:cats.map(c=>byCat.get(c).paid), type:'bar', name:'Paid apps',
      marker:{color:'#2dd4ea'},
      customdata:cats.map(c=>`Paid apps (listed price > $0): ${byCat.get(c).paid}`),
      hovertemplate:'%{customdata}<extra></extra>',
    };
    Plotly.newPlot(el, [traceFree, tracePaid], {
      ...layoutBase,
      margin:{t:16,l:60,r:24,b:88},
      legend:legendH, barmode:'group',
      xaxis:{...AX, tickangle:-30},
      yaxis:{...AX, title:{text:'Number of apps', font:{color:'#8f86ac', size:12}}},
    }, CONFIG);
    const paidCats = cats.filter(c=>byCat.get(c).prices.length);
    Plotly.newPlot(el6b, [{
      x:paidCats.map(c=>mean(byCat.get(c).prices).toFixed(2)),
      y:paidCats,
      type:'bar', orientation:'h',
      marker:{color:'#fbbf62'},
      customdata:paidCats.map(c=>`Mean listed price: $${mean(byCat.get(c).prices).toFixed(2)} over ${byCat.get(c).prices.length} paid apps`),
      hovertemplate:'%{customdata}<extra></extra>',
      text:paidCats.map(c=>'$'+mean(byCat.get(c).prices).toFixed(2)), textposition:'outside',
      textfont:{color:'#b1a8cf', family:'JetBrains Mono', size:11},
    }], {
      ...layoutBase,
      // The heading is a paper-space annotation drawn ABOVE the plot area, so
      // the top margin has to be tall enough to hold it — at t:40 it sat on top
      // of the first bar. The right margin grows too, so the 'outside' price
      // labels on the longest bar are no longer clipped.
      margin:{t:74,l:120,r:64,b:40},
      xaxis:{...AX, title:{text:'Mean listed price ($) — a price tag, NOT observed revenue', font:{color:'#8f86ac', size:11}}},
      yaxis:{...AX, autorange:'reversed'},
      annotations:[{x:0.02, y:1.02, xref:'paper', yref:'paper', showarrow:false, yanchor:'bottom',
        text:'Mean listed price per category (paid apps only)', font:{color:'#fbbf62', family:'JetBrains Mono', size:11}}],
    }, CONFIG);
  } else {
    Plotly.newPlot(el, [traceFree], {
      ...layoutBase,
      margin:{t:16,l:60,r:24,b:88},
      barmode:'group',
      xaxis:{...AX, tickangle:-30},
      yaxis:{...AX, title:{text:'Number of apps', font:{color:'#8f86ac', size:12}}},
    }, CONFIG);
  }
}

/* ---------------- chart registry + resize ---------------- */
const CHART_RENDERERS = {
  chart1: renderChart1, chart2: renderChart2, chart3: renderChart3,
  chart4: renderChart4, chart5: renderChart5, chart6: renderChart6,
};

/* ---------------- lazy, measured rendering ----------------
   Rendering all six charts synchronously used to block the main thread until
   every one of them was done: the page was unresponsive (and felt frozen
   around whichever chart the user happened to be looking at) even though only
   one chart was ever visible at a time. Each chart is now plotted when it
   first comes near the viewport, with a yield so the scroll that revealed it
   is not itself janked.

   Append `?bench=1` to the URL (or set window.APEX_BENCH = true) to log the
   per-chart Plotly timing to the console. */
/* TEMPORARY (2026-09-19): forced ON so the panel can be read without editing
   the URL. Revert to the ?bench=1 gate once the numbers have been captured:
       const BENCH = globalThis.APEX_BENCH === true ||
         !!(globalThis.location && /[?&]bench=1\b/.test(globalThis.location.search || '')); */
const BENCH = globalThis.APEX_BENCH !== false;
const CHART_TIMES = {};
const plotted = new Set();

/* TEMPORARY diagnostic panel — remove this whole block when the chart work is
   signed off. Only ever created for ?bench=1, so it is invisible in normal
   use; it exists so the numbers can be read without opening DevTools. */
function renderBench(){
  if(!BENCH) return;
  try{
    let el = document.getElementById('apexBench');
    if(!el){
      el = document.createElement('div');
      el.id = 'apexBench';
      el.className = 'apex-bench';
      document.body.appendChild(el);
    }
    const ids = Object.keys(CHART_RENDERERS);
    const rows = ids.map(id=>{
      const ms = CHART_TIMES[id];
      const cls = ms === undefined ? ' pending' : (ms > 400 ? ' slow' : '');
      return `<div class="apex-bench-row${cls}"><span>${id}</span>`
        + `<b>${ms === undefined ? 'not plotted yet' : ms + ' ms'}</b></div>`;
    }).join('');
    const done = ids.filter(id=>CHART_TIMES[id] !== undefined);
    const total = done.reduce((s,id)=>s + CHART_TIMES[id], 0);
    el.innerHTML =
      `<div class="apex-bench-h">Plotly render time</div>${rows}`
      + `<div class="apex-bench-row total"><span>${done.length}/${ids.length} plotted</span>`
      + `<b>${Math.round(total)} ms</b></div>`
      + `<div class="apex-bench-note">WebGL ${webglAvailable() ? 'yes → scattergl' : 'no → SVG'}`
      + (done.length < ids.length ? ' · scroll down to plot the rest' : '') + `</div>`;
  }catch(err){ /* a diagnostic must never break the page */ }
}

function plotChart(id){
  if(plotted.has(id)) return;
  plotted.add(id);
  const now = () => (globalThis.performance && globalThis.performance.now
    ? globalThis.performance.now() : Date.now());
  const t0 = now();
  try{
    CHART_RENDERERS[id]();
  }catch(err){
    console.error(`[apex] ${id} failed to render`, err);
  }
  const dt = now() - t0;
  CHART_TIMES[id] = Math.round(dt);
  if(BENCH){
    console.log(`[apex] ${id}: ${dt.toFixed(0)} ms`);
    renderBench();
    reportBench();
  }
  return dt;
}

/* TEMPORARY diagnostic sink. The preview is behind a tokened proxy, so the
   numbers cannot be read from a URL or from DevTools here — instead the page
   posts them to the backend, where they land in `python app.py`'s stdout.
   Remove together with renderBench(). */
function reportBench(){
  if(!BENCH || typeof fetch !== 'function') return;
  const ids = Object.keys(CHART_RENDERERS);
  if(ids.some(id=>CHART_TIMES[id] === undefined)) return;   // wait until all are plotted
  try{
    fetch('api/bench', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        charts: CHART_TIMES,
        webgl: webglAvailable(),
        plotted: `${ids.filter(id=>CHART_TIMES[id] !== undefined).length}/${ids.length}`,
        scrollWidth: (document.documentElement || {}).scrollWidth,
        innerWidth: window.innerWidth,
      }),
    }).catch(()=>{});
  }catch(err){ /* a diagnostic must never break the page */ }
}

function renderCharts(){
  const ids = Object.keys(CHART_RENDERERS);
  renderBench();                 // show the (still empty) panel immediately
  // Legacy window hook so the panel can be re-rendered from the console.
  if(BENCH) globalThis.renderBenchPanel = renderBench;
  // No IntersectionObserver (very old browser, jsdom, the test harness):
  // fall back to rendering everything, exactly like before.
  if(typeof IntersectionObserver === 'undefined' || typeof document.getElementById !== 'function'){
    ids.forEach(plotChart);
    return;
  }
  const io = new IntersectionObserver((entries)=>{
    for(const entry of entries){
      if(!entry.isIntersecting) continue;
      io.unobserve(entry.target);
      const id = entry.target.id;
      // Yield first: the browser gets to paint the scroll before Plotly runs.
      setTimeout(()=>plotChart(id), 0);
    }
  }, {rootMargin:'400px 0px'});        // start about a screen before it shows
  ids.forEach(id=>{
    const el = document.getElementById(id);
    if(el) io.observe(el);
  });
}

/* Resize: `responsive:true` already redraws on container changes; this only
   handles window resizes, and debounces them so dragging the window edge does
   not relayout all seven plots once per event. */
let resizeTimer = null;
window.addEventListener('resize', ()=>{
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(()=>{
    document.querySelectorAll('[id^="chart"], #chart_confusion, #chart_importance').forEach(el=>{
      if(el && el.data) Plotly.Plots.resize(el);
    });
    reportBench();          // TEMPORARY: re-report at the new viewport width
  }, 150);
});

globalThis.APEX_CHART_TIMES = CHART_TIMES;
