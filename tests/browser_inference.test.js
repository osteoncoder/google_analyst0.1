#!/usr/bin/env node
/*
 * browser_inference.test.js — the browser engine must agree with sklearn.
 *
 * ml_inference.js exists so the prediction forms work without the FastAPI
 * backend (GitHub Pages, VS Code Live Server, a file:// page). That is only
 * honest if it reproduces the pipeline's arithmetic, so every case in
 * ml/artifacts/browser/parity_cases.json — inputs plus the sklearn
 * pipeline's own output — is replayed here and required to match.
 *
 * Run:  node tests/browser_inference.test.js
 * Exits 0 on success, 1 on any mismatch.
 */
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const ROOT = path.resolve(__dirname, '..');
const MODELS_BUNDLE = path.join(ROOT, 'ml', 'artifacts', 'browser', 'models.js');
const PARITY = path.join(ROOT, 'ml', 'artifacts', 'browser', 'parity_cases.json');
const APPS_BUNDLE = path.join(ROOT, 'data', 'apps_bundle.js');

const failures = [];
const fail = (msg) => { console.error('  FAIL: ' + msg); failures.push(msg); };
const ok = (msg) => console.log('  ok  : ' + msg);

for (const f of [MODELS_BUNDLE, PARITY]) {
  if (!fs.existsSync(f)) {
    console.error(`Missing ${path.relative(ROOT, f)} — run \`python browser_export.py\` first.`);
    process.exit(1);
  }
}

/* ---------------- load the bundle + engine into one context ---------------- */
const sandbox = { console, Math, JSON, Number, String, Object, Array, isFinite, module: undefined };
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(ROOT, 'ml_inference.js'), 'utf8'), sandbox, { filename: 'ml_inference.js' });
vm.runInContext(fs.readFileSync(MODELS_BUNDLE, 'utf8'), sandbox, { filename: 'models.js' });

if (!sandbox.APEX_BROWSER_MODELS || !sandbox.APEX_BROWSER_MODELS.models) {
  console.error('APEX_BROWSER_MODELS missing after loading the bundle');
  process.exit(1);
}

const { ApexInference, APEX_BROWSER_MODELS } = sandbox;
const parity = JSON.parse(fs.readFileSync(PARITY, 'utf8'));

console.log(`browser_inference: ${Object.keys(APEX_BROWSER_MODELS.models).length} model(s), ` +
  `exported ${APEX_BROWSER_MODELS.generated_at}`);

/* ---------------- 1. parity with scikit-learn ---------------- */
const TOL = 1e-9;   // both sides are float64; only association order differs

for (const [task, block] of Object.entries(parity.tasks)) {
  const model = APEX_BROWSER_MODELS.models[task];
  if (!model) { fail(`${task}: in parity fixtures but missing from models.js`); continue; }

  let worst = 0;
  let mismatch = null;
  const n = block.cases.length;

  for (let i = 0; i < n; i++) {
    const inputs = Object.assign({}, block.cases[i]);
    // The pipeline consumes engineered columns, exactly like app.py builds them.
    const row = {
      category: inputs.category,
      content_rating: inputs.content_rating === undefined ? null : inputs.content_rating,
      size_mb: inputs.size_mb === null || inputs.size_mb === undefined ? null : inputs.size_mb,
      price: inputs.price === null || inputs.price === undefined ? null : inputs.price,
      price_is_positive: (inputs.price || 0) > 0 ? 1 : 0,
    };
    // clean.py rule 13's inputs: omitted (null) => the imputer's median, which
    // is the behaviour this test pins — so leave the key off entirely.
    for (const k of ['app_age_days', 'days_since_update', 'developer_app_count',
                     'min_android', 'ad_supported', 'in_app_purchases', 'editors_choice']) {
      if (inputs[k] !== undefined && inputs[k] !== null) row[k] = inputs[k];
    }
    if ((model.features || []).indexOf('reviews_log') !== -1) {
      row.reviews_log = Math.log1p(inputs.reviews || 0);
    }

    const expected = block.expected[i];
    if (typeof expected === 'number') {
      const got = ApexInference.predictRegression(model, row);
      const d = Math.abs(got - expected);
      if (d > worst) worst = d;
      if (d > TOL && !mismatch) mismatch = `case ${i}: js=${got} sklearn=${expected}`;
    } else {
      const got = ApexInference.classDistribution(model, row);
      const keys = Object.keys(expected.probabilities);
      for (let k = 0; k < keys.length; k++) {
        const d = Math.abs(got[k] - expected.probabilities[keys[k]]);
        if (d > worst) worst = d;
        if (d > TOL && !mismatch) mismatch = `case ${i} ${keys[k]}: js=${got[k]} sklearn=${expected.probabilities[keys[k]]}`;
      }
      const label = ApexInference.tier(model, row).predicted_tier;
      if (label !== expected.tier && !mismatch) mismatch = `case ${i}: tier js=${label} sklearn=${expected.tier}`;
    }
  }

  if (mismatch) fail(`${task}: ${mismatch}`);
  else ok(`${task}: ${n} cases match sklearn (max abs error ${worst.toExponential(2)})`);
}

/* ---------------- 2. behaviour that mirrors app.py ---------------- */
const m1 = APEX_BROWSER_MODELS.models.m1_rating;
if (m1) {
  const base = { category: 'Education', size_mb: 86, price: 0, price_is_positive: 0, reviews_log: Math.log1p(1200) };
  const sized = ApexInference.predictRegression(m1, base);
  const noSize = ApexInference.predictRegression(m1, Object.assign({}, base, { size_mb: null }));
  if (isFinite(sized) && isFinite(noSize)) ok('m1: omitted size falls back to the training median (no NaN)');
  else fail('m1: NaN leaked into a prediction when size was omitted');

  const unknown = ApexInference.predictRegression(m1, Object.assign({}, base, { category: 'Not A Real Category' }));
  const zeroVec = ApexInference.featureVector(m1, Object.assign({}, base, { category: 'Not A Real Category' }));
  // The one-hot block is [category..., content_rating...] — slice just the part
  // this assertion is about.
  const numLen = m1.prep.num.columns.length;
  const catLen = m1.prep.cat.categories[0].length;
  const catPart = zeroVec.slice(numLen, numLen + catLen);
  if (catPart.every((v) => v === 0) && isFinite(unknown)) {
    ok('m1: unseen category -> all-zero one-hot (handle_unknown="ignore"), prediction still finite');
  } else fail('m1: unseen category should encode as all zeros');

  // clean.py rule 13: two categorical columns. Omitted -> the exported
  // training-set mode; unseen -> all zeros. Both must hold for content_rating.
  const ratingBlock = (row) =>
    ApexInference.featureVector(m1, row).slice(numLen + catLen);
  const defaultRating = ratingBlock(Object.assign({}, base, { content_rating: null }));
  const expectedDefault = m1.prep.cat.defaults[1];
  const defaultIdx = m1.prep.cat.categories[1].indexOf(expectedDefault);
  if (defaultIdx >= 0 && defaultRating[defaultIdx] === 1 &&
      defaultRating.reduce((a, b) => a + b, 0) === 1) {
    ok(`content_rating omitted -> training mode "${expectedDefault}" (not an unknown category)`);
  } else fail('content_rating omitted should fall back to the exported training mode');
  if (ratingBlock(Object.assign({}, base, { content_rating: 'Unseen Rating 21+' }))
        .every((v) => v === 0)) {
    ok('content_rating unseen -> all-zero one-hot (handle_unknown="ignore")');
  } else fail('unseen content rating should encode as all zeros');

  // Must agree with clean.py: clean_category('  books_&_reference ') === 'Books & Reference'
  const messy = ApexInference.normalizeCategory('  books_&_reference ');
  if (messy === 'Books & Reference') ok('normalizeCategory: underscores/spaces/casing -> "Books & Reference"');
  else fail(`normalizeCategory('  books_&_reference ') => ${JSON.stringify(messy)}`);
  const messy2 = ApexInference.normalizeCategory('  books_and-reference ');
  if (messy2 === 'Books And Reference') ok('normalizeCategory: "_and_" is a word, not "&"');
  else fail(`normalizeCategory('  books_and-reference ') => ${JSON.stringify(messy2)}`);
  if (ApexInference.normalizeCategory('EDUCATION') === 'Education') ok('normalizeCategory: casing normalized');
  else fail('normalizeCategory should title-case');
  if (ApexInference.normalizeCategory('___') === null) ok('normalizeCategory: blank-after-normalization is missing');
  else fail('normalizeCategory("___") should be null');
  if (ApexInference.normalizeCategory('commication') === 'Communication') ok('normalizeCategory: documented typo fixed');
  else fail('normalizeCategory should apply the Commication -> Communication typo fix');
}

const m2 = APEX_BROWSER_MODELS.models.m2_tier_without_reviews;
if (m2) {
  const p = ApexInference.tier(m2, { category: 'Game', size_mb: 86, price: 0, price_is_positive: 0 });
  const total = Object.values(p.probabilities).reduce((a, b) => a + b, 0);
  if (Math.abs(total - 1) < 1e-9) ok('m2: tier probabilities sum to 1');
  else fail(`m2: probabilities sum to ${total}`);
  if (p.probabilities[p.predicted_tier] === Math.max.apply(null, Object.values(p.probabilities))) {
    ok('m2: predicted tier is the argmax class');
  } else fail('m2: predicted tier is not the argmax');
}

/* ---------------- 3. the dataset bundle (file:// path) ---------------- */
if (fs.existsSync(APPS_BUNDLE)) {
  const dataCtx = { console };
  dataCtx.globalThis = dataCtx;
  vm.createContext(dataCtx);
  vm.runInContext(fs.readFileSync(APPS_BUNDLE, 'utf8'), dataCtx, { filename: 'apps_bundle.js' });
  const payload = dataCtx.APEX_APPS_PAYLOAD;
  const appsJson = JSON.parse(fs.readFileSync(path.join(ROOT, 'data', 'apps.json'), 'utf8'));

  if (payload && payload.columns) {
    const cols = payload.columns;
    const keys = Object.keys(cols);
    const n = payload.row_count;
    // Reference expansion (mirrors expandBundle() in data.js).
    const rows = [];
    for (let i = 0; i < n; i++) {
      const r = {};
      for (const k of keys) r[k] = cols[k].dict ? cols[k].dict[cols[k].data[i]] : cols[k].data[i];
      rows.push(r);
    }
    if (rows.length === appsJson.rows.length) ok(`apps_bundle.js: ${rows.length.toLocaleString()} rows expand`);
    else fail(`apps_bundle.js: ${rows.length} rows vs apps.json ${appsJson.rows.length}`);

    let diffs = 0;
    for (let i = 0; i < Math.min(n, 500); i++) {
      if (JSON.stringify(rows[i]) !== JSON.stringify(appsJson.rows[i])) diffs++;
    }
    if (diffs === 0) ok('apps_bundle.js: expanded rows are identical to data/apps.json (500-row spot check)');
    else fail(`apps_bundle.js: ${diffs}/500 rows differ from apps.json`);

    if (!/<\/script/i.test(fs.readFileSync(APPS_BUNDLE, 'utf8'))) {
      ok('apps_bundle.js: no raw </script> sequence to break out of the tag');
    } else fail('apps_bundle.js: contains a </script> sequence');
  } else fail('apps_bundle.js: missing columns');
} else {
  console.log('  --  : data/apps_bundle.js not built yet (run `python browser_export.py --data`)');
}

console.log(failures.length
  ? `\nBROWSER INFERENCE: ${failures.length} FAILURE(S)`
  : '\nBROWSER INFERENCE: ALL CHECKS PASSED');
process.exit(failures.length ? 1 : 0);
