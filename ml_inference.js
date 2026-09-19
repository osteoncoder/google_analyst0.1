/* ================================================================
   ml_inference.js — evaluate the *saved* scikit-learn pipelines in JS

   WHY
   ---
   The prediction forms used to work only while `python app.py` was
   running: they POST to /api/predict/*, which a static host (GitHub
   Pages) and a file:// page simply do not have. This file removes that
   dependency by reproducing the pipeline's arithmetic in the browser:

       impute (training medians) -> StandardScaler -> one-hot(category)
       -> walk the exported trees

   It is NOT a re-implementation of scikit-learn and it is NOT an
   approximation. `browser_export.py` flattens the *fitted* artifacts
   (ml/artifacts/<task>/pipeline.joblib) into ml/artifacts/browser/models.js,
   and this file walks exactly those numbers. If the bundle is missing
   or holds an unsupported estimator, it refuses to guess — the caller
   keeps using the API and shows the honest "unavailable" state.

   Correctness is enforced by tests/browser_inference.test.js, which
   replays ml/artifacts/browser/parity_cases.json (inputs + the sklearn
   pipeline's own outputs) and requires an exact match.
   ================================================================ */

(function (global) {
  'use strict';

  /* Mirrors clean.py rule 9 / data.js normCategory — a hand-typed category
     is normalized the same way the training data was, so "education" and
     "Education" cannot silently produce different predictions. */
  const CATEGORY_TYPO_MAP = { Commication: 'Communication' };

  function titleCase(s) {
    return s.replace(/\w\S*/g, (t) => t.charAt(0).toUpperCase() + t.slice(1).toLowerCase());
  }

  function normalizeCategory(v) {
    if (v === null || v === undefined) return null;
    let s = String(v).trim();
    if (s === '' || s.toLowerCase() === 'nan' || s.toLowerCase() === 'none') return null;
    s = s.replace(/[_\-]+/g, ' ');
    if (s.trim() === '') return null;
    const t = titleCase(s).trim();
    return CATEGORY_TYPO_MAP[t] || t || null;
  }

  function isFiniteNumber(v) {
    return typeof v === 'number' && isFinite(v);
  }

  /* impute -> scale -> one-hot, in the exact column order the fitted
     ColumnTransformer emits (numeric block first, then the one-hot block). */
  function featureVector(model, inputs) {
    const prep = model.prep;
    const num = prep.num;
    const out = new Array(num.columns.length + prep.cat.categories.length);

    for (let i = 0; i < num.columns.length; i++) {
      const key = num.columns[i];
      let v = inputs ? inputs[key] : undefined;
      if (v === null || v === undefined || v === '' || !isFiniteNumber(Number(v))) {
        v = num.medians[i];                       // SimpleImputer(strategy='median')
      }
      const scale = num.scale[i];
      out[i] = (Number(v) - num.mean[i]) / (scale === 0 ? 1 : scale);
    }

    const cats = prep.cat.categories;
    const raw = inputs ? inputs[prep.cat.column] : undefined;
    const normalized = normalizeCategory(
      raw === undefined || raw === null || raw === '' ? prep.cat.missing_fill : raw
    );
    const idx = normalized === null ? -1 : cats.indexOf(normalized);
    // handle_unknown='ignore': an unseen category becomes all zeros.
    const offset = num.columns.length;
    for (let j = 0; j < cats.length; j++) out[offset + j] = (j === idx ? 1 : 0);
    return out;
  }

  /* Walk one exported tree. l[i] === -1 marks a leaf. */
  function traverse(tree, x) {
    let node = 0;
    const l = tree.l, r = tree.r, f = tree.f, t = tree.t;
    while (l[node] !== -1) {
      node = (x[f[node]] <= t[node]) ? l[node] : r[node];
    }
    return node;
  }

  function softmax(z) {
    const m = Math.max.apply(null, z);
    const e = z.map((v) => Math.exp(v - m));
    const s = e.reduce((a, b) => a + b, 0) || 1;
    return e.map((v) => v / s);
  }

  function argmax(arr) {
    let best = 0;
    for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
    return best;                                   // ties keep the first max (numpy argmax)
  }

  /* ---------------- regression ---------------- */

  function predictRegression(model, inputs) {
    if (model.family === 'GradientBoostingClassifier') {
      throw new Error('GradientBoostingClassifier is not supported by the browser engine');
    }
    const x = featureVector(model, inputs);
    if (model.family === 'GradientBoostingRegressor') {
      // sklearn: raw = init.constant_ + learning_rate * sum(tree leaf values)
      let raw = model.init;
      for (let i = 0; i < model.trees.length; i++) {
        raw += model.learning_rate * model.trees[i].v[traverse(model.trees[i], x)];
      }
      return raw;
    }
    if (model.family === 'DecisionTreeRegressor') {
      return model.tree.v[traverse(model.tree, x)][0];
    }
    throw new Error('unsupported regression estimator: ' + model.family);
  }

  /* ---------------- classification ---------------- */

  function classDistribution(model, inputs) {
    const x = featureVector(model, inputs);
    const classes = model.classes;

    if (model.family && model.family.indexOf('DecisionTree') === 0) {
      // sklearn stores the leaf class distribution (already normalized).
      const v = model.tree.v[traverse(model.tree, x)];
      const total = v.reduce((a, b) => a + b, 0) || 1;
      return v.map((c) => c / total);
    }
    if (model.family === 'LogisticRegression') {
      // multinomial: z_k = intercept_k + coef_k . x, then softmax
      const z = model.coef.map((row, k) => {
        let s = model.intercept[k] || 0;
        for (let j = 0; j < row.length; j++) s += row[j] * x[j];
        return s;
      });
      return softmax(z);
    }
    if (model.family === 'GradientBoostingClassifier') {
      throw new Error('GradientBoostingClassifier is not supported by the browser engine');
    }
    throw new Error('unsupported classifier: ' + model.family);
  }

  /* Public, UI-shaped helpers ---------------------------------------------
     These mirror the FastAPI responses field-for-field so ml_dashboard.js
     can render either source without knowing which one answered. */

  function rating(model, inputs) {
    return { raw_prediction: predictRegression(model, inputs) };
  }

  function tier(model, inputs) {
    const probs = classDistribution(model, inputs);
    const labels = model.classes || [];
    const probabilities = {};
    labels.forEach((c, i) => { probabilities[c] = probs[i]; });
    return { predicted_tier: labels[argmax(probs)], probabilities: probabilities };
  }

  const API = {
    version: 1,
    normalizeCategory,
    featureVector,
    traverse,
    rating,
    tier,
    predictRegression,
    classDistribution,
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = API;
  global.ApexInference = API;
})(typeof window !== 'undefined' ? window : globalThis);
