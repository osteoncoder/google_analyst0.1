/* ================================================================
   data.js — dataset loading + cleaning mirror

   Source of truth: data/apps.json (produced by `python clean.py`).
   The primary dataset is the MIT-licensed gauthamp10 Google-Playstore
   scrape (2,312,944 apps, June 2021); the committed default is a
   deterministic 40,000-row stratified sample of it (see
   data/playstore_sample.meta.json), and the full run substitutes the
   complete file via `python fetch_dataset.py`.
   If apps.json cannot be fetched (e.g. the page is opened directly
   with file://, where browsers block fetch()), the dashboard loads
   data/apps_bundle.js — the same payload as a plain <script>, which
   IS allowed from file://. EMBEDDED_SAMPLE (11 rows) is only the last
   resort when both are unreachable, and is labelled as such.

   The parsing functions mirror clean.py rule-by-rule so the JS
   fallback and the Python pipeline can never disagree:
     Installs  -> reported download-band LOWER BOUND (int), not exact
     Reviews   -> int, blank => 0
     Rating    -> kept only if 1..5
     Size      -> "86M" / "72K" / "Varies with device"
     Price     -> "" => 0 (free, preserved), "$1.99" => 1.99
     Category  -> trim + title case + one documented typo fix
   ================================================================ */

const PURPLE_SCALE = ['#a855f7','#7c3aed','#2dd4ea','#f472b6','#fbbf62','#818cf8','#4ade80','#fb7185','#38bdf8','#e879f9'];

/* The only documented spelling fix (typo present in the sample data). */
const CATEGORY_TYPO_MAP = { 'Commication': 'Communication' };

/* ---- EMBEDDED SAMPLE — the project's original 11 rows, unchanged ---- */
const EMBEDDED_SAMPLE = [
  {App:"Subway Surfers",     Category:"Game",         Rating:4.5, Reviews:1000000, Size:"86M", Installs:1000000000, Sentiment_Subjectivity:0.60},
  {App:"Candy Crush Saga",   Category:"Game",         Rating:4.4, Reviews:800000,  Size:"72M", Installs:500000000,  Sentiment_Subjectivity:0.55},
  {App:"My Beauty Salon",    Category:"Beauty",       Rating:4.2, Reviews:1200,    Size:"25M", Installs:100000,     Sentiment_Subjectivity:0.65},
  {App:"Business Tycoon",    Category:"business",     Rating:3.8, Reviews:600,     Size:"15M", Installs:60000,      Sentiment_Subjectivity:0.52},
  {App:"Manga Reader",       Category:"commics",      Rating:4.0, Reviews:2000,    Size:"10M", Installs:150000,     Sentiment_Subjectivity:0.58},
  {App:"Chat Messenger",     Category:"commication",  Rating:4.1, Reviews:15000,   Size:"20M", Installs:500000,     Sentiment_Subjectivity:0.61},
  {App:"Dating Match",       Category:"Dating",       Rating:3.9, Reviews:800,     Size:"30M", Installs:80000,      Sentiment_Subjectivity:0.54},
  {App:"Fun Stream",         Category:"Entertainment",Rating:4.3, Reviews:25000,   Size:"45M", Installs:1000000,    Sentiment_Subjectivity:0.70},
  {App:"Social Connect",     Category:"social",       Rating:4.2, Reviews:18000,   Size:"35M", Installs:500000,     Sentiment_Subjectivity:0.63},
  {App:"City Fest Guide",    Category:"event",        Rating:4.6, Reviews:550,     Size:"12M", Installs:55000,      Sentiment_Subjectivity:0.68},
  {App:"Super Speed Racer",  Category:"Game",         Rating:4.7, Reviews:3000,    Size:"50M", Installs:200000,     Sentiment_Subjectivity:0.51},
];

/* ---------------- parsing (mirrors clean.py) ---------------- */

function isNum(v){ return typeof v === 'number' && isFinite(v); }

function parseInstalls(v){
  if(v===null || v===undefined) return NaN;
  if(isNum(v)) return v>=0 ? v : NaN;
  let s = String(v).trim().replace(/,/g,'').replace(/\s+/g,'');
  if(s==='' || s.toLowerCase()==='nan') return NaN;
  if(s.endsWith('+')) s = s.slice(0,-1);
  if(!/^\d+$/.test(s)) return NaN;
  return parseInt(s,10);
}

function parseReviews(v){
  if(v===null || v===undefined) return 0;
  if(isNum(v)) return v>0 ? Math.floor(v) : 0;
  const s = String(v).trim().replace(/,/g,'');
  if(s==='' || s.toLowerCase()==='nan') return 0;
  const n = parseFloat(s);
  return isFinite(n) && n>=0 ? Math.floor(n) : 0;
}

function parsePrice(v){
  if(v===null || v===undefined) return NaN;          // missing price (no column) = unknown
  if(isNum(v)) return v;                              // 0 stays 0 (free, legitimate)
  let s = String(v).trim();
  if(s==='' || s.toLowerCase()==='nan') return 0;     // blank = free in Play exports
  s = s.replace(/\$/g,'').replace(/,/g,'');
  const n = parseFloat(s);
  return isFinite(n) ? n : NaN;
}

function parseSizeMB(v){
  if(v===null || v===undefined) return NaN;
  if(isNum(v)) return v>0 ? v : NaN;
  const s = String(v).trim().toUpperCase().replace(/,/g,'');
  if(s==='' || s==='NAN' || s==='NONE' || s==='VARIES WITH DEVICE') return NaN;
  const m = s.match(/^(\d+(?:\.\d+)?)([KM]?)$/);
  if(!m) return NaN;
  let val = parseFloat(m[1]);
  if(m[2]==='K') val /= 1024;
  return val;
}

function parseRating(v){
  if(v===null || v===undefined) return NaN;
  const n = isNum(v) ? v : parseFloat(v);
  return (isFinite(n) && n>=1 && n<=5) ? n : NaN;
}

/* Mirrors clean.py rule 8: "Jan 15, 2024" / "January 15, 2024" / ISO.
   All returned dates are UTC-midnight Date objects so consumers can safely
   read getUTC* fields (charts.js does) without local-timezone day drift. */
const MONTH_IDX = {
  jan:0, january:0, feb:1, february:1, mar:2, march:2, apr:3, april:3,
  may:4, jun:5, june:5, jul:6, july:6, aug:7, august:7, sep:8, sept:8, september:8,
  oct:9, october:9, nov:10, november:10, dec:11, december:11,
};

function parseDate(v){
  if(v===null || v===undefined) return null;
  const s = String(v).trim();
  if(s==='' || s.toLowerCase()==='nan') return null;
  const t = Date.parse(s);                 // handles "2024-01-15" (ISO, UTC midnight)
  if(!isNaN(t)) return new Date(t);
  const m = s.match(/^([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})$/);   // "Jan 15, 2024" / "January 15, 2024"
  if(m){
    const mi = MONTH_IDX[m[1].toLowerCase()];
    if(mi!==undefined) return new Date(Date.UTC(parseInt(m[3],10), mi, parseInt(m[2],10)));
  }
  return null;
}

function parseSentiment(v){
  const n = isNum(v) ? v : parseFloat(v);
  return (isFinite(n) && n>=0 && n<=1) ? n : NaN;
}

function titleCase(s){ return s.replace(/\w\S*/g, t => t.charAt(0).toUpperCase() + t.slice(1).toLowerCase()); }

/* Mirrors clean.py rule 9: underscores/hyphens → spaces, title case, typo fix.
   Blank-after-normalization ("___", "-") is missing, not a category named " ". */
function normCategory(v){
  if(v===null || v===undefined) return 'Unknown';
  const t = titleCase(String(v).trim().replace(/[_\-]+/g, ' ')).trim();
  if(!t) return 'Unknown';
  return CATEGORY_TYPO_MAP[t] || t;
}

/* Idempotent: works on raw export rows AND on already-cleaned apps.json rows. */
function cleanRow(r){
  const price = parsePrice(r.Price !== undefined ? r.Price : r.price);
  return {
    app: String(r.App !== undefined ? r.App : (r.app || '')).trim(),
    category: normCategory(r.Category !== undefined ? r.Category : r.category),
    rating: parseRating(r.Rating !== undefined ? r.Rating : r.rating),
    reviews: parseReviews(r.Reviews !== undefined ? r.Reviews : r.reviews),
    installs: parseInstalls(r.Installs !== undefined ? r.Installs : r.installs),
    size_mb: parseSizeMB(r.Size !== undefined ? r.Size : r.size_mb),
    price,
    type: isNum(price) ? (price > 0 ? 'Paid' : 'Free') : 'Unknown',
    last_updated: parseDate(r['Last Updated'] !== undefined ? r['Last Updated'] : r.last_updated),
    sentiment: parseSentiment(r.Sentiment_Subjectivity !== undefined ? r.Sentiment_Subjectivity : r.sentiment_subjectivity),
  };
}

/* ---------------- dataset loading ----------------
   Three routes, tried in order:
     1. fetch('data/apps.json')            served over http(s): app.py, Live Server, GitHub Pages
     2. <script src="data/apps_bundle.js"> the SAME payload as a script — the only
                                           route that works on a file:// page, where
                                           browsers block fetch() with an opaque origin
     3. EMBEDDED_SAMPLE                    11 rows, last resort, always labelled as such

   Route 2 is why the dashboard no longer silently collapses to 11 rows when the
   page is opened straight off the disk (VS Code "Run Active File", double-click).
   -------------------------------------------------------------------------- */

const APPS_JSON_URL = 'data/apps.json';
const APPS_BUNDLE_URL = 'data/apps_bundle.js';

/* Classic <script> injection: allowed from file:// (fetch/XHR is not). */
function loadScriptTag(src){
  return new Promise((resolve, reject)=>{
    const s = document.createElement('script');
    s.src = src;
    s.onload = ()=>resolve();
    s.onerror = ()=>reject(new Error('could not load ' + src));
    document.head.appendChild(s);
  });
}

/* Expand the columnar bundle back into row objects (inverse of
   browser_export.build_dataset_bundle). Values are byte-identical to
   apps.json — tests/browser_inference.test.js asserts that. */
function expandBundle(p){
  const cols = p.columns || {};
  const keys = Object.keys(cols);
  const n = p.row_count || ((cols.app && cols.app.data) || []).length;
  const rows = new Array(n);
  for(let i=0;i<n;i++){
    const r = {};
    for(let k=0;k<keys.length;k++){
      const key = keys[k];
      const c = cols[key];
      r[key] = c.dict ? c.dict[c.data[i]] : c.data[i];
    }
    rows[i] = r;
  }
  return rows;
}

async function loadAppsFromBundle(){
  await loadScriptTag(APPS_BUNDLE_URL);
  const payload = window.APEX_APPS_PAYLOAD;
  if(!payload || !payload.columns) throw new Error(APPS_BUNDLE_URL + ' loaded but empty');
  const rows = expandBundle(payload);
  if(rows.length === 0) throw new Error('no rows in ' + APPS_BUNDLE_URL);
  return {rows, payload};
}

function isFilePage(){
  return typeof location !== 'undefined' && location.protocol === 'file:';
}

/* Shared label logic for both routes. */
function sourceLabel(rows, payload){
  const sampling = payload.sampling || null;
  const label = sampling && sampling.full_rows
    ? `${Number(sampling.sample_rows || rows.length).toLocaleString()}-row sample of the ` +
      `${Number(sampling.full_rows).toLocaleString()}-row dataset`
    : (payload.is_sample ? 'SAMPLE' : '');
  return {sampling, label};
}

async function loadApps(){
  let fetchErr = null;

  // file:// pages cannot fetch: skip straight to the script bundle.
  if(!isFilePage()){
    try{
      const res = await fetch(APPS_JSON_URL, {cache:'no-store'});
      if(!res.ok) throw new Error('HTTP ' + res.status);
      const payload = await res.json();
      const rows = (payload.rows || []).map(cleanRow);
      if(rows.length === 0) throw new Error('no rows in apps.json');
      const {sampling, label} = sourceLabel(rows, payload);
      return {
        rows,
        report: payload.report || {},
        sampling,
        source: (payload.source || APPS_JSON_URL) + (label ? ` (${label})` : ''),
        is_sample: !!payload.is_sample,
        mode: 'fetch',
      };
    }catch(err){
      fetchErr = err;
    }
  }

  try{
    const {rows, payload} = await loadAppsFromBundle();
    const cleaned = rows.map(cleanRow);
    const {sampling, label} = sourceLabel(cleaned, payload);
    return {
      rows: cleaned,
      report: payload.report || {},
      sampling,
      source: (payload.source || APPS_JSON_URL) + ' via ' + APPS_BUNDLE_URL + (label ? ` (${label})` : ''),
      is_sample: !!payload.is_sample,
      mode: 'bundle',
    };
  }catch(bundleErr){
    // Genuinely nothing to show — say so instead of pretending 11 rows is the data.
    return {
      rows: EMBEDDED_SAMPLE.map(cleanRow),
      report: {},
      source: 'EMBEDDED 11-ROW SAMPLE — ' + APPS_JSON_URL + ' and ' + APPS_BUNDLE_URL +
              ' were both unreachable' + (isFilePage() ? ' (file:// pages cannot fetch)' : ''),
      is_sample: true,
      mode: 'embedded',
      degraded: true,
      errors: [fetchErr, bundleErr].filter(Boolean).map(e=>String(e.message || e)),
    };
  }
}

/* ---------------- small math/format helpers ---------------- */

function fmtCompact(n){
  if(!isFinite(n)) return '—';
  if(n>=1e9) return (n/1e9).toFixed(1)+'B';
  if(n>=1e6) return (n/1e6).toFixed(1)+'M';
  if(n>=1e3) return (n/1e3).toFixed(1)+'K';
  return String(Math.round(n));
}
function sum(arr){ return arr.reduce((s,x)=>s+x,0); }
function mean(arr){ return arr.length ? sum(arr)/arr.length : NaN; }
function pearson(pairs){
  const n = pairs.length;
  if(n < 3) return NaN;
  const mx = sum(pairs.map(p=>p[0]))/n, my = sum(pairs.map(p=>p[1]))/n;
  let num=0, dx=0, dy=0;
  for(const [x,y] of pairs){ const a=x-mx, b=y-my; num+=a*b; dx+=a*a; dy+=b*b; }
  if(dx===0 || dy===0) return NaN;
  return num/Math.sqrt(dx*dy);
}
