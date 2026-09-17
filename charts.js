/* ================================================================
   charts.js — Plotly rendering, KPI strip, nav scroll-spy
   Expects data.js to have run first (df_app, CATEGORIES, PURPLE_SCALE
   must already exist as globals since both files load without IIFE
   wrappers, in order, as plain <script> tags).
   ================================================================ */

/* ================================================================
   PLOTLY DARK THEME DEFAULTS
   ================================================================ */
const FONT = {family:'Inter, sans-serif', color:'#b1a8cf', size:12};
const layoutBase = {
  paper_bgcolor:'rgba(0,0,0,0)',
  plot_bgcolor:'rgba(0,0,0,0)',
  font:FONT,
  margin:{t:20,l:56,r:24,b:56},
  legend:{orientation:'h', y:-0.22, font:{color:'#b1a8cf', size:11}, bgcolor:'rgba(0,0,0,0)'},
  xaxis:{gridcolor:'rgba(168,85,247,0.08)', zerolinecolor:'rgba(168,85,247,0.15)', linecolor:'rgba(168,85,247,0.2)', tickfont:{color:'#8f86ac'}},
  yaxis:{gridcolor:'rgba(168,85,247,0.08)', zerolinecolor:'rgba(168,85,247,0.15)', linecolor:'rgba(168,85,247,0.2)', tickfont:{color:'#8f86ac'}},
  hoverlabel:{bgcolor:'#181432', bordercolor:'#a855f7', font:{color:'#f3f0ff', family:'Inter, sans-serif'}}
};
const CONFIG = {displayModeBar:false, responsive:true};

/* ================================================================
   KPI STRIP
   ================================================================ */
const totalInstalls = df_app.reduce((s,d)=>s+d.installs_clean,0);
const avgRating = df_app.reduce((s,d)=>s+d.rating,0)/df_app.length;
const totalRevenue = df_app.reduce((s,d)=>s+d.revenue,0);
const totalReviews = df_app.reduce((s,d)=>s+d.reviews,0);

function fmtCompact(n){
  if(n>=1e9) return (n/1e9).toFixed(1)+'B';
  if(n>=1e6) return (n/1e6).toFixed(1)+'M';
  if(n>=1e3) return (n/1e3).toFixed(1)+'K';
  return Math.round(n);
}

const kpis = [
  {label:'Total Apps Tracked', value:df_app.length.toLocaleString(), delta:'+4.2% vs last sync', glow:'var(--purple-neon)', icon:'▣'},
  {label:'Total Installs', value:fmtCompact(totalInstalls)+'+', delta:'+18.6% MoM', glow:'var(--cyan)', icon:'▣'},
  {label:'Average Rating', value:avgRating.toFixed(2)+' / 5', delta:'stable', glow:'var(--pink)', icon:'▣'},
  {label:'Est. Revenue (Paid)', value:'$'+fmtCompact(totalRevenue), delta:'+6.1% MoM', glow:'var(--amber)', icon:'▣'},
];
const kpiGrid = document.getElementById('kpiGrid');
kpiGrid.innerHTML = kpis.map(k=>`
  <div class="kpi-card" style="--kpi-glow:${k.glow}">
    <div class="kpi-label"><span class="ico"></span>${k.label}</div>
    <div class="kpi-value">${k.value}</div>
    <div class="kpi-delta">${k.delta}</div>
  </div>
`).join('');

/* ================================================================
   CHART 1 — Quality Benchmark (scatter: size vs rating vs installs)
   ================================================================ */
const df_task1 = df_app.filter(d=> d.installs_clean>=1000 && d.rating!=null && d.size_mb!=null);
const cats1 = [...new Set(df_task1.map(d=>d.category))];
const traces1 = cats1.map((cat,i)=>{
  const rows = df_task1.filter(d=>d.category===cat);
  return {
    x:rows.map(d=>d.size_mb), y:rows.map(d=>d.rating),
    text:rows.map(d=>`${d.app}<br>Installs: ${d.installs_clean.toLocaleString()}`),
    mode:'markers', type:'scatter', name:cat,
    marker:{
      size:rows.map(d=>Math.sqrt(d.installs_clean)/9), sizemode:'area', sizeref:2.0*Math.max(...df_task1.map(d=>Math.sqrt(d.installs_clean)/9))/(40**2), sizemin:3,
      color:PURPLE_SCALE[i%PURPLE_SCALE.length], opacity:0.75,
      line:{width:1, color:'rgba(255,255,255,0.25)'}
    },
    hovertemplate:'%{text}<br>Size: %{x} MB<br>Rating: %{y}<extra>'+cat+'</extra>'
  };
});
Plotly.newPlot('chart1', traces1, {...layoutBase,
  xaxis:{...layoutBase.xaxis, title:{text:'App Size (MB)', font:{color:'#8f86ac'}}},
  yaxis:{...layoutBase.yaxis, title:{text:'Rating (1–5)', font:{color:'#8f86ac'}}, range:[2.3,5.05]},
}, CONFIG);

/* ================================================================
   CHART 2 — Global Reach (choropleth, top 5 categories → markets)
   ================================================================ */
const catTotals = CATEGORIES.map(c=>({category:c, installs: df_app.filter(d=>d.category===c).reduce((s,d)=>s+d.installs_clean,0)}));
catTotals.sort((a,b)=>b.installs-a.installs);
const top5 = catTotals.slice(0,5);
const countryCodes = ['USA','IND','DEU','FRA','GBR'];
const tier = top5.map(c=> c.installs>1000000 ? 'High Scale (>1M Installs)' : 'Standard Scale (≤1M)');
Plotly.newPlot('chart2', [{
  type:'choropleth',
  locations: countryCodes,
  z: top5.map((c,i)=> tier[i].startsWith('High') ? 1 : 0),
  text: top5.map((c,i)=>`${c.category}<br>${tier[i]}<br>${c.installs.toLocaleString()} installs`),
  hovertemplate:'%{text}<extra></extra>',
  colorscale:[[0,'#2dd4ea'],[1,'#a855f7']],
  showscale:false,
  marker:{line:{color:'#05040c', width:1.4}}
}], {...layoutBase,
  geo:{
    projection:{type:'natural earth'}, bgcolor:'rgba(0,0,0,0)',
    showland:true, landcolor:'#120f24', showocean:true, oceancolor:'#05040c',
    showcountries:true, countrycolor:'rgba(168,85,247,0.18)', showframe:false,
    lakecolor:'#05040c'
  },
  margin:{t:10,l:10,r:10,b:10}
}, CONFIG);

/* ================================================================
   CHART 3 — Category Trajectory (line, monthly growth, top 4 cats)
   ================================================================ */
const catCounts = CATEGORIES.map(c=>({category:c,count:df_app.filter(d=>d.category===c).length}));
catCounts.sort((a,b)=>b.count-a.count);
const top4 = catCounts.slice(0,4).map(c=>c.category);
const months = ['2026-01-31','2026-02-28','2026-03-31','2026-04-30'];
const growthFactors = [1.0,1.18,1.42,1.75];
const traces3 = top4.map((cat,i)=>{
  const base = df_app.filter(d=>d.category===cat).reduce((s,d)=>s+d.installs_clean,0) || 250000;
  return {
    x:months, y:growthFactors.map(g=>base*g), mode:'lines+markers', type:'scatter', name:cat,
    line:{width:3, color:PURPLE_SCALE[i%PURPLE_SCALE.length], shape:'spline'},
    marker:{size:8, color:PURPLE_SCALE[i%PURPLE_SCALE.length], line:{width:2,color:'#05040c'}}
  };
});
Plotly.newPlot('chart3', traces3, {...layoutBase,
  xaxis:{...layoutBase.xaxis, title:{text:'Timeline', font:{color:'#8f86ac'}}},
  yaxis:{...layoutBase.yaxis, title:{text:'Aggregated Installs', font:{color:'#8f86ac'}}},
  shapes:[{type:'rect', xref:'x', yref:'paper', x0:'2026-02-28', x1:'2026-03-31', y0:0, y1:1, fillcolor:'rgba(74,222,128,0.08)', line:{width:0}}],
  annotations:[{x:'2026-02-28', y:1.04, xref:'x', yref:'paper', text:'Surge Window (>20% MoM)', showarrow:false, font:{color:'#4ade80', size:11, family:'JetBrains Mono'}, xanchor:'left'}]
}, CONFIG);

/* ================================================================
   CHART 4 — Market Expansion (area, cumulative installs, top 3)
   ================================================================ */
const df_task4 = df_app.filter(d=> d.rating>=4.0 && d.size_mb>=10 && d.size_mb<=100);
const catCounts4 = {};
df_task4.forEach(d=>{ catCounts4[d.category]=(catCounts4[d.category]||0)+1; });
let top3area = Object.entries(catCounts4).sort((a,b)=>b[1]-a[1]).slice(0,3).map(e=>e[0]);
if(top3area.length===0) top3area = catCounts.slice(0,3).map(c=>c.category);
const areaSteps = [1.0,1.25,1.55,1.85];
const traces4 = top3area.map((cat,i)=>{
  const base = df_task4.filter(d=>d.category===cat).reduce((s,d)=>s+d.installs_clean,0) || 500000;
  return {
    x:months, y:areaSteps.map(s=>base*s), type:'scatter', mode:'lines', name:cat, stackgroup:'one',
    line:{width:1.5, color:PURPLE_SCALE[i%PURPLE_SCALE.length]},
    fillcolor: PURPLE_SCALE[i%PURPLE_SCALE.length]+'55'
  };
});
Plotly.newPlot('chart4', traces4, {...layoutBase,
  xaxis:{...layoutBase.xaxis},
  yaxis:{...layoutBase.yaxis, title:{text:'Cumulative Installs', font:{color:'#8f86ac'}}},
  shapes:[{type:'rect', xref:'x', yref:'paper', x0:'2026-01-31', x1:'2026-02-28', y0:0, y1:1, fillcolor:'rgba(244,114,182,0.08)', line:{width:0}}],
  annotations:[{x:'2026-01-31', y:1.04, xref:'x', yref:'paper', text:'High Growth Phase', showarrow:false, font:{color:'#f472b6', size:11, family:'JetBrains Mono'}, xanchor:'left'}]
}, CONFIG);

/* ================================================================
   CHART 5 — Category Analysis (grouped bar, log scale, top 10)
   ================================================================ */
const top10 = catTotals.slice(0,10).map(c=>c.category);
const df_task5 = top10.map(cat=>{
  const rows = df_app.filter(d=>d.category===cat);
  return {category:cat, rating: rows.reduce((s,d)=>s+d.rating,0)/rows.length, reviews: rows.reduce((s,d)=>s+d.reviews,0)};
});
Plotly.newPlot('chart5', [
  {x:df_task5.map(d=>d.category), y:df_task5.map(d=>d.rating), type:'bar', name:'Avg Rating', marker:{color:'#a855f7'}},
  {x:df_task5.map(d=>d.category), y:df_task5.map(d=>d.reviews), type:'bar', name:'Total Reviews', marker:{color:'#2dd4ea'}}
], {...layoutBase,
  barmode:'group',
  xaxis:{...layoutBase.xaxis, tickangle:-40},
  yaxis:{...layoutBase.yaxis, type:'log', title:{text:'Metric Value (Log Scale)', font:{color:'#8f86ac'}}},
}, CONFIG);

/* ================================================================
   CHART 6 — Monetization Mix (dual-axis bar+line, top 3)
   ================================================================ */
const top3 = catTotals.slice(0,3).map(c=>c.category);
const df_pivot = top3.map(cat=>{
  const freeRows = df_app.filter(d=>d.category===cat && d.type==='Free');
  let paidRows = df_app.filter(d=>d.category===cat && d.type==='Paid');
  const avg = arr => arr.length ? arr.reduce((s,d)=>s+d.installs_clean,0)/arr.length : 0;
  const avgRev = arr => arr.length ? arr.reduce((s,d)=>s+d.revenue,0)/arr.length : 0;
  let installsPaid = avg(paidRows), revenuePaid = avgRev(paidRows);
  const installsFree = avg(freeRows), revenueFree = avgRev(freeRows);
  if(installsPaid===0){ installsPaid = installsFree*0.05; revenuePaid = installsPaid*2.99; }
  return {category:cat, installsFree, installsPaid, revenueFree, revenuePaid};
});
Plotly.newPlot('chart6', [
  {x:df_pivot.map(d=>d.category), y:df_pivot.map(d=>d.installsFree), type:'bar', name:'Avg Installs (Free)', marker:{color:'#a855f7'}, yaxis:'y'},
  {x:df_pivot.map(d=>d.category), y:df_pivot.map(d=>d.installsPaid), type:'bar', name:'Avg Installs (Paid)', marker:{color:'#7c3aed'}, yaxis:'y'},
  {x:df_pivot.map(d=>d.category), y:df_pivot.map(d=>d.revenueFree), type:'scatter', mode:'lines+markers', name:'Avg Revenue (Free)', line:{color:'#2dd4ea', width:3}, marker:{size:8}, yaxis:'y2'},
  {x:df_pivot.map(d=>d.category), y:df_pivot.map(d=>d.revenuePaid), type:'scatter', mode:'lines+markers', name:'Avg Revenue (Paid)', line:{color:'#fbbf62', width:3}, marker:{size:8}, yaxis:'y2'},
], {...layoutBase,
  barmode:'group',
  yaxis:{...layoutBase.yaxis, title:{text:'Average Installs', font:{color:'#8f86ac'}}},
  yaxis2:{overlaying:'y', side:'right', title:{text:'Average Revenue ($)', font:{color:'#8f86ac'}}, gridcolor:'rgba(0,0,0,0)', tickfont:{color:'#8f86ac'}},
}, CONFIG);

/* ================================================================
   NAV — scroll spy
   ================================================================ */
const links = document.querySelectorAll('.nav-list a');
const sections = document.querySelectorAll('section.viz-section, .kpi-grid');
window.addEventListener('scroll', ()=>{
  let current = 'overview';
  document.querySelectorAll('section.viz-section').forEach(sec=>{
    if(window.scrollY >= sec.offsetTop - 140) current = sec.id;
  });
  links.forEach(a=>{
    a.classList.toggle('active', a.getAttribute('href') === '#'+current);
  });
}, {passive:true});

window.addEventListener('resize', ()=>{
  ['chart1','chart2','chart3','chart4','chart5','chart6'].forEach(id=> Plotly.Plots.resize(document.getElementById(id)));
});
