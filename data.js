/* ================================================================
   data.js — dataset + ETL cleaning
   Mirrors load_and_clean_dataset() from the Python notebook:
   - normalizes category casing (title case)
   - parses "86M" / "512K" style size strings to numeric MB
   - keeps installs/reviews/rating as-is (already numeric here)
   Replace RAW_ROWS with a fetch()/PapaParse call against your real
   CSV/XLSX export whenever you have the full dataset.
   ================================================================ */

const PURPLE_SCALE = ['#a855f7','#7c3aed','#2dd4ea','#f472b6','#fbbf62','#818cf8','#4ade80','#fb7185','#38bdf8','#e879f9'];

/* ---- RAW ROWS — exactly as supplied ---- */
const RAW_ROWS = [
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

/* ---- ETL: mirrors load_and_clean_dataset() from the notebook ---- */
function convertSize(v){
  if(v==null) return NaN;
  v = String(v).toUpperCase().trim();
  if(v==='VARIES WITH DEVICE') return NaN;
  if(v.endsWith('M')) return parseFloat(v.slice(0,-1));
  if(v.endsWith('K')) return parseFloat(v.slice(0,-1))/1024;
  return parseFloat(v);
}
function titleCase(s){ return s.replace(/\w\S*/g, t=> t.charAt(0).toUpperCase()+t.slice(1).toLowerCase()); }

const df_app = RAW_ROWS.map((r,i)=>({
  id:i,
  app:r.App,
  category:titleCase(r.Category.trim()),      // normalizes "business"→"Business", "commics"→"Commics", "commication"→"Commication"
  rating:r.Rating,
  reviews:r.Reviews,
  installs_clean:r.Installs,
  size_mb:convertSize(r.Size),
  price_clean:0,
  type:'Free',
  revenue:0,
  sentiment_subjectivity:r.Sentiment_Subjectivity
}));

const CATEGORIES = [...new Set(df_app.map(d=>d.category))];

document.getElementById('rowCount').textContent = df_app.length.toLocaleString();
document.getElementById('runTime').textContent = new Date().toLocaleDateString('en-US',{month:'short',day:'numeric',year:'numeric'});
