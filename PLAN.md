# Implementation Plan

Written so work can resume cleanly if the agent session terminates. Every phase
is self-contained, lists the files it touches, and says how to verify it.

---

## 0. Current state (verified 2026-09-19)

- Branch: `arena/01a0ba27-google-analyst0-1`. PR **#3** is open against `main`
  (14 files, +3773/−94) and **not merged** — so GitHub Pages and a `main` ZIP
  do not yet have any of this work.
- All changes so far are pushed and the branch is in sync with the remote.
- `python app.py` serves the dashboard with the FastAPI backend; the page also
  works with **no backend at all** (GitHub Pages, Live Server, `file://`) via
  `ml_inference.js` + the committed bundles. Both engines were verified to
  return **identical** predictions.

### Baseline metrics (committed 40,000-row sample, seed 42)

| Model | Test metric | Value |
|---|---|---|
| M1 rating regression (Gradient Boosting) | MAE / RMSE / R² | 0.5027 / 0.6615 / **+0.0506** |
| M2-A install tier, with Reviews (Logistic Regression) | acc / macro-F1 | 0.9069 / 0.8275 |
| M2-B install tier, without Reviews (Decision Tree) | acc / macro-F1 | 0.6729 / **0.3067** |
| M2 majority-class baseline | acc / macro-F1 | 0.7242 / 0.2100 |

Note: M2-B's accuracy is **below** the majority baseline. That is documented
honestly in the README; it is the main model-quality weakness.

### Open decision carried over (not yet made)

Adding engineered features — `app_age_days`, `days_since_update`,
`developer_app_count`, `min_android`, `ad_supported`, `in_app_purchases`,
`editors_choice`, `content_rating` — was **measured in a scratch copy** and
lifts M2-B substantially:

| Variant | acc | macro-F1 | artifact size |
|---|---|---|---|
| committed baseline (Decision Tree) | 0.6729 | 0.3067 | 1.0 MB |
| + features, Random Forest auto-selected | 0.7692 | 0.5492 | **443 MB** |
| + features, no `developer_app_count` | 0.7631 | 0.5344 | ~443 MB |
| + features, Gradient Boosting forced | 0.7682 | 0.5440 | **1.0 MB** |

- The lift is **real, not leakage**: removing the one feature computed across
  the whole dataset (`developer_app_count`, which can straddle the app-name
  grouped split) costs only ~0.015 macro-F1.
- The auto-selected Random Forest is a **443 MB** artifact — unshippable.
  Gradient Boosting gets 99% of the lift at 1.0 MB.
- **Decision needed:** adopt the features with Gradient Boosting (recommended),
  adopt with a depth-capped Random Forest, or leave models as they are.
- If adopted, three things must change for it to be shippable:
  1. `clean.py::_consumed_columns` must whitelist the 8 extra raw columns —
     `load_raw()` only loads whitelisted columns (a memory optimisation for the
     666 MB file), so they are otherwise silently all-NaN. This was the bug
     that produced empty features on the first attempt.
  2. `browser_export.py` / `ml_inference.js` need the extra numeric columns and
     a second categorical column, plus `GradientBoostingClassifier` support
     (the engine currently supports regression GBR, Decision Tree and Logistic
     Regression only).
  3. The prediction forms must either gain ~7 inputs or impute the new
     features to training medians/modes and state that as an assumption.

---

## Phase 1 — Make the models faster at low compute cost

**Constraint: must not break the current execution pathway.** Specifically,
every change below must preserve:

- the `ml/artifacts/<task>/{pipeline.joblib,meta.json}` contract that
  `app.py` loads at startup;
- the `metrics.json` schema consumed by dashboard sections 08–09;
- the Pydantic request models in `app.py` (`RatingIn`, `TierIn`);
- `browser_export.py` → `ml_inference.js` parity (tests must still pass);
- the documented run commands (`clean.py` → `train_models.py` → `app.py`).

### Measured on this machine (40k sample, 2 cores, n_train = 31,926)

| Estimator | Fit time | acc | macro-F1 |
|---|---|---|---|
| `GradientBoostingClassifier(200, depth 3)` | 23.5 s | 0.7260 | 0.2481 |
| **`HistGradientBoostingClassifier(200)`** | **1.7 s** | 0.7265 | 0.2478 |
| `HistGradientBoostingClassifier(200, early_stopping)` | 1.7 s | 0.7265 | 0.2478 |
| `RandomForestClassifier(300)`, `n_jobs=1` | 9.2 s | 0.6917 | 0.3081 |
| `RandomForestClassifier(300)`, `n_jobs=-1` | **4.8 s** | 0.6917 | 0.3081 |

Two headline findings:

1. **`HistGradientBoosting` is ~14× faster than `GradientBoosting` with metrics
   within noise** (acc +0.0005, macro-F1 −0.0003). It bins features and handles
   NaN natively. `requirements.txt` already pins `scikit-learn>=1.3`, which is
   the minimum for it — no dependency change needed.
2. **`n_jobs=-1` on Random Forest is 1.9× faster here and bit-identical** —
   results are unchanged because `random_state` is fixed. It scales with core
   count, so the win is larger on normal hardware.

### Actions, in priority order

1. **Add `HistGradientBoostingClassifier` / `Regressor` as candidates** and let
   the existing selection rule pick them.
   - *Risk:* `browser_export.py` does not support HistGB, so if it wins, the
     in-browser engine loses that model. Mitigation, in order of preference:
     add HistGB support to `ml_inference.js` (it is an additive ensemble of
     binned trees — exportable, but a bigger job than the current trees), or
     restrict the browser-exported model to a supported family and document it.
   - *Gate:* do not merge until `tests/browser_inference.test.js` passes.
2. **Set `n_jobs=-1`** on every Random Forest candidate. Zero metric change.
3. **Enable `early_stopping` on HistGB** with the existing train/val split —
   free speed on the full run.
4. **Cap tree complexity** where unbounded: `min_samples_leaf` / `max_depth`
   on the tree and forest candidates. This is what prevents the 443 MB
   artifact and speeds both fit and predict.
5. **Cut the design-matrix memory footprint** for the 2.3M-row run:
   `OneHotEncoder(sparse_output=True)` where the estimator accepts sparse
   input, and/or downcast numerics to `float32`. Today the dense matrix is
   ~940 MB (2.31M × ~51 × 8 bytes).
6. **Cheap dimensionality control:** `OneHotEncoder(min_frequency=…)` folds
   rare categories instead of giving each its own column.
7. **Optional, for Track B only:** select among candidates on a subsample, then
   refit the winner on the full data. Document it if used.

### Expected effect on the full 2.31M-row run

The previous estimate was **1.5–4 h** dominated by 3× RF-300 and 3× GB-200, all
single-threaded. With HistGB replacing GB (~14×) and `n_jobs=-1` on RF (~2× on
this box, more on a real laptop), the same work should land in roughly
**15–45 minutes**. This is a projection, not a measurement — it must be
re-measured on real hardware before being written down as fact.

### Verification

- `node tests/browser_inference.test.js` — JS engine still matches sklearn.
- `node tests/smoke_frontend.js` — frontend unchanged.
- `python train_models.py` on the 40k sample: metrics move only within noise
  unless a deliberate change is made; record before/after.
- `curl localhost:8000/api/health` → `models_loaded: true`.

---

## Phase 2 — Optimise and professionalise the website

### 2.1 Hamburger menu for navigation

Today `.shell` is `grid-template-columns:250px 1fr`. At `≤760px` the sidebar
becomes `position:relative` and stacks **above** the content, pushing the
dashboard down a full screen; between 760–1080px it still eats 250px.

- Add a `<button class="nav-toggle">` (hamburger) in the topbar, `aria-expanded`
  / `aria-controls`, hidden on desktop.
- Sidebar becomes off-canvas below ~1080px: `transform:translateX(-100%)`,
  `.sidebar.open{transform:none}`, with an overlay backdrop.
- JS in `ml_dashboard.js`: toggle, close on nav-link click, close on `Escape`,
  close on backdrop click. Keep the existing scroll-spy `initNav()` working.
- Files: `index.html`, `style.css`, `ml_dashboard.js`.
- Verify: keyboard-only navigation works; no layout shift on desktop; existing
  `initNav` scroll behaviour unaffected.

### 2.2 Fix elements overflowing the right edge

Identified causes:

1. **CSS grid min-size (most likely primary cause).** A `1fr` track's
   automatic minimum is `auto`, so wide content (Plotly containers, wide
   metric tables) expands the track instead of shrinking. Fix: `main{min-width:0}`.
2. **Plotly legends placed outside the plot area** — `legendV = {orientation:'v',
   x:1.02, y:1}` with `margin:{r:130}` on chart 1. At narrow widths the legend
   is pushed off-screen. Fix: switch to an in-plot / horizontal legend below a
   breakpoint and reduce the right margin responsively.
3. `main{padding:36px 44px; max-width:1400px}` — no upper bound on the grid
   child itself.
4. `body{overflow-x:hidden}` currently **masks** the overflow rather than
   fixing it; real offenders stay clipped.

Fix in `style.css`; verify by loading at 1440/1080/768/375 px and confirming
`document.documentElement.scrollWidth <= window.innerWidth`.

### 2.3 Speed up chart 1 (bubble) and chart 5 (category scatter)

Measure first, then fix. Add a temporary `performance.now()` around each
`Plotly.newPlot` and record per-chart times before changing anything.

**Chart 1 — App Quality Benchmark.** Builds **48 separate scatter traces**
(one per category) over ~30k markers, each with a 1px stroke, and groups with
48 `rows.filter()` passes plus `Math.max(...sizes)` spread. Suspects, in order:
- 48 SVG traces with per-marker strokes — the dominant cost.
- Plotly legend with 48 entries.

**Chart 5 — Category Analysis.** A *single* trace with only ~48 points, but
rendered `mode:'markers+text'` with 48 always-visible text labels on a
**log** axis. Suspects: SVG text layout on a log axis, and `order.indexOf()`
inside a `map` (O(n²), trivial at n=48 but worth fixing while touching it).

Shared suspects for both:
- **All six charts render synchronously** in `renderCharts()` during
  bootstrap, so one heavy chart blocks the main thread and the whole page
  (including chart 5's region) feels frozen.
- `CONFIG = {responsive:true}` attaches a resize observer to every plot, so any
  window resize redraws all six — very expensive with chart 1 present.

Actions:
1. Render charts **lazily** via `IntersectionObserver` — plot each chart when
   it first scrolls into view. Biggest single win for perceived speed.
2. Yield between renders (e.g. `await new Promise(r => setTimeout(r, 0))`) so
   the first paint is never blocked for seconds.
3. Chart 1: group categories in a **single pass** (one `Map`, not 48 filters);
   collapse 48 traces into **one** trace with a categorical colour array, or
   use `scattergl` (WebGL) for the point count; drop per-marker strokes.
4. Chart 5: precompute the colour index in a `Map` (kill the `indexOf` scan);
   consider showing labels on hover only, or `textposition` that avoids
   overlap.
5. Keep `responsive:true` but debounce resize handling.

Verification: measure per-chart `newPlot` duration before/after; confirm first
contentful paint no longer waits on all six charts; re-run
`node tests/smoke_frontend.js` (it asserts trace counts and totals, so
collapsing 48 traces into 1 will require updating that assertion deliberately
and consciously).

### 2.4 Remove / collapse unnecessary explanations

The pages carry long `section-desc` paragraphs and several caveat banners.
Trim them, but **do not delete the data-integrity caveats** — "installs are
band lower bounds", "price is a tag, not revenue", "correlation is not
causation", "trained on a sample" are a deliberate strength of this project and
are asserted by the test suite.

Approach: keep the one-line claim visible; move the justification into a
collapsible `<details>` ("Why?"). Applies to `index.html` section descriptions,
`footer-note`, and the ML section notes.

---

## Phase 3 — Professionalise the README

Restructure without losing the honesty that makes it credible:

- Add a table of contents.
- Lead with a 3–4 line summary: what it is, the dataset, the two models.
- Keep the two run tracks (sample / full 2.3M) — these were rebuilt recently
  and tested against real errors.
- Move viva-facing material out into `VIVA_PREP.txt` (Phase 4) so the README
  reads as documentation rather than exam notes.
- Tighten prose; keep the trap callouts (project root, `0.0.0.0` vs
  `localhost`, `--sample 4000` typo).
- Add the Phase 1 speed findings once implemented.

---

## Phase 4 — Viva preparation file

Create **`VIVA_PREP.txt`** (plain text, downloadable, no Markdown so it opens
anywhere):

1. **One-page project summary** — goal, dataset, pipeline, stack.
2. **Pipeline stages** with the number that matters at each stage
   (2,312,944 → 40,000 stratified sample → 40,000 cleaned → 22,418 rated →
   31,926 train / 7,967 test).
3. **Numbers to quote**, each with its caveat (see Phase 0 table).
4. **Anticipated questions and answers** — draft list:
   - Why a 40,000-row sample instead of the full 2.31M? (git size; deterministic
     stratified selection; full run is one command away.)
   - Why is install count a band lower bound, not a real download count?
   - Why exclude Reviews from M2-B when it obviously helps? (target proxy —
     leakage; the with/without comparison is shown in section 09.)
   - Why is M2-B's accuracy below the majority baseline? (honest limitation;
     what the enriched features do about it.)
   - Why a grouped split on app name?
   - What is the difference between listed price and revenue? (no revenue is
     estimated anywhere.)
   - Why is R² only ~0.05 on M1? (ratings are concentrated near 4.2; the mean
     baseline gets MAE 0.52.)
   - How is reproducibility guaranteed? (seed 42, deterministic sample by
     content hash, metrics record the source MD5.)
   - How do predictions work without a backend? (exported pipelines evaluated
     in-browser, parity-tested against sklearn.)
   - Ethics / licensing / scraping provenance (MIT, June-2021 scrape).
   - Limitations and what you would do next.
5. **Common challenge questions** — "is this just a dashboard?", "how do you
   know the numbers are real?", "what would you change with more time?"

Also add a "download" affordance so it is genuinely downloadable rather than
only viewable.

---

## Sequencing and dependencies

```
Phase 1 (models)      ──┐
Phase 2 (website)     ──┼── independent of each other; any order
Phase 3 (README)      ──┘   (but 3 should come after 1 so the numbers are final)
Phase 4 (viva file)   ── last: must quote the FINAL metrics
```

- The **open decision** in §0 (adopt enriched features?) should be settled
  before or during Phase 1, since it changes the metrics that Phases 3 and 4
  quote.
- Phase 2's chart refactor touches `tests/smoke_frontend.js` assertions
  (48 traces → 1). Update those assertions deliberately.

## How to resume this plan

1. `git checkout arena/01a0ba27-google-analyst0-1 && git pull`
2. Confirm state: `node tests/browser_inference.test.js` and
   `node tests/smoke_frontend.js` must both print `ALL CHECKS PASSED`.
3. Start from the **Phase 0 open decision**, then take phases in the order
   above. Each phase is independently shippable — commit and push at the end
   of each one rather than batching.
4. Scratch experiments live in `/tmp` (`/tmp/feat`, `/tmp/feat3`) and are
   **not** persisted — they hold the enriched-feature patches and the measured
   metrics JSON. If those numbers are wanted again, re-derive them; do not
   assume `/tmp` survived.
5. Merge PR #3 into `main` only when told to — that is what republishes GitHub
   Pages.
