# Implementation Plan

Written so work can resume cleanly if the agent session terminates. Every phase
is self-contained, lists the files it touches, and says how to verify it.

---

## 0. Current state (verified 2026-09-19)

- Branch: `arena/01a0ba27-google-analyst0-1`. PR **#3** is open against `main`
  (14 files, +3773/−94) and **not merged** — so a `main` ZIP does not yet
  have any of this work. (2026-09-20: the user repointed GitHub Pages to
  deploy from *this branch*, so the live site does currently show it; merging
  PR #3 remains what publishes it to `main`.)
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

### Decision: adopt the enriched features (settled 2026-09-19)

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
- **DECIDED: adopt the enriched features**, shipped as **Gradient Boosting**
  (acc 0.7682 / macro-F1 0.5440, 1.0 MB) rather than the auto-selected
  Random Forest (0.7692 / 0.5492, 443 MB). GB gives 99% of the lift at 1/443
  the size.
- Mechanism for that choice: rather than hard-coding the estimator, add an
  **artifact-size budget to model selection** (Step 2.3). RF-300 unbounded is
  rejected as unshippable; GB then wins on merit by validation macro-F1
  (0.530 vs LogReg 0.482 and DT 0.459). This generalises — it prevents the
  next 443 MB surprise instead of special-casing today's.
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

## Phase 1 — Adopt enriched features + make training faster  ✅ DONE

**Status: complete and verified on 2026-09-19.** Every step below was
implemented and every item in §1.5 passes. Measured outcome (committed 40,000-row
sample, seed 42, 2-core sandbox):

| Model | Before | After | Predicted in §1.3 |
|---|---|---|---|
| M2-B accuracy | 0.6729 (below the 0.7242 baseline) | **0.7682** (above it) | 0.7682 ✓ |
| M2-B macro-F1 | 0.3067 | **0.5440** | 0.5440 ✓ |
| M2-B selected model | Decision Tree | **Gradient Boosting**, 0.99 MB | GB ✓ |
| M1 R² | +0.0506 | **+0.0711** (MAE 0.494, RMSE 0.654) | +0.0711 ✓ |
| M2-A macro-F1 / accuracy | 0.8275 / 0.9069 | **0.8385 / 0.9077** | 0.8385 ✓ |
| Full training run | — | **2 min 34 s** (was ~4 min before `n_jobs=-1`) | — |

Rejected by the 10 MB artifact budget, scores recorded and shown struck out in
section 09: Random Forest at **412 MB** (M1), **176 MB** (M2-A) and **334 MB**
(M2-B). RF's M2-B validation macro-F1 (0.535) was marginally above GB's (0.530)
— the budget is what makes GB the selection, and it is visible on the page.

Implementation notes that a fresh session needs:

- `clean.py` rule 13 derives the eight features; `_consumed_columns()` must
  list the raw columns or `load_raw()` never reads them (silent all-NaN bug —
  it cost two iterations). The features are written to `apps_cleaned.csv` but
  deliberately **not** to `apps.json`, so the browser payload is unchanged.
- `train_models.py` drops any derived column that is entirely NaN (old Kaggle
  export, 11-row mechanical sample), so those datasets still train.
- `prep.cat` in the browser bundle is now **multi-column**
  (`{columns, categories:[[...],[...]], defaults:[...]}`). `ml_inference.js`
  still accepts the legacy single-column shape, so a stale bundle cannot throw.
- `GradientBoostingClassifier` inference: `score_k = log(class_prior_k) +
  lr · Σ_stages leaf_k`, then softmax — verified against sklearn to 1e-16, and
  the parity suite matches to 5e-11.
- Categorical defaults (`categorical_defaults` in `meta.json`, `prep.cat.defaults`
  in the bundle) are what an *omitted* field means: the training-set mode.
  `app.py` and `ml_inference.js` both read them, so the engines cannot diverge.


**Goal:** ship the measured M2-B improvement (acc 0.6729 → 0.7682, macro-F1
0.3067 → 0.5440, now above the 0.7242 majority baseline) and cut training cost,
without breaking anything that works today.

### 1.0 Non-negotiable constraints

Every step must preserve these, or the change does not ship:

1. `ml/artifacts/<task>/{pipeline.joblib,meta.json}` — `app.py` loads these at
   startup; the contract is fixed.
2. `metrics.json` schema — dashboard sections 08–09 read it directly.
3. Pydantic models `RatingIn` / `TierIn` in `app.py` — existing requests must
   keep working (new fields must be optional).
4. **Browser-engine parity** — `ml_inference.js` must reproduce the served
   pipeline exactly, proven by `tests/browser_inference.test.js`. The API and
   the in-browser engine must never disagree.
5. The documented run flow: `clean.py` → `train_models.py` → `app.py`.
6. `node tests/smoke_frontend.js` stays green.

### 1.1 Why Gradient Boosting and not the 14×-faster HistGB

`HistGradientBoosting` measured ~14× faster with metrics inside noise, but it
has **no public accessor for its binned trees** — verified: no `estimators_`,
no tree objects on the public API. It therefore cannot be exported to
`ml_inference.js`, so adopting it would silently kill the no-backend
prediction path (GitHub Pages, Live Server, `file://`).

`GradientBoostingClassifier` **is** exportable and small:

- `estimators_` shape `(n_estimators, n_classes)` = `(200, 4)` → 800 trees
- ~**11,880 nodes total** (measured 594 nodes for 10×4, scaled ×20)
- `init_.class_prior_` gives the softmax starting point; `loss == 'log_loss'`
  (multinomial), so inference is: start at `log(class_prior)`, add
  `learning_rate × Σ leaf values` per class, then softmax.

So: **exact Gradient Boosting everywhere**, browser parity intact.
HistGB is retained only as an explicitly-flagged `--fast` option for the
optional Track B full run, documented as incompatible with the browser bundle.

### 1.2 Step-by-step

#### Step 1 — `clean.py`: carry the 8 new columns through

Three separate edits are required; missing any one silently yields NaNs.

1. `_consumed_columns()` — whitelist the raw columns, otherwise `load_raw()`
   never loads them (it uses `usecols` to keep the 666 MB file in RAM).
   This was the bug that produced all-NaN features on the first attempt:
   ```python
   ("Released",), ("Scraped Time",), ("Developer Id",), ("Content Rating",),
   ("Minimum Android",), ("Ad Supported",), ("In App Purchases",), ("Editors Choice",),
   ```
2. Derive the features inside `clean()`, after `df["type"]` is set (so the
   derived columns inherit the corrupted-row and duplicate filters applied
   afterwards):
   | Column | Rule |
   |---|---|
   | `app_age_days` | (`Scraped Time` − `Released`) in days |
   | `days_since_update` | (`Scraped Time` − `Last Updated`) in days |
   | `developer_app_count` | rows per `Developer Id` |
   | `min_android` | leading number of `"5.0 and up"` → `5.0` |
   | `ad_supported`, `in_app_purchases`, `editors_choice` | `True`/`False` → `1.0`/`0.0`, else NaN |
   | `content_rating` | trimmed string (categorical) |
3. Add them to the `out = df[[...]]` whitelist near the end of `clean()`.

Sanity check (measured on the sample): `app_age_days` mean 1062 d (38,673
non-null), `days_since_update` mean 546 d, `developer_app_count` mean 2.87,
`min_android` mean 4.36, `ad_supported` 51.6% true, `in_app_purchases` 10.6%,
`editors_choice` 0.27%, `content_rating` = Everyone 34,510 / Teen 3,707 /
Mature 17+ 1,111 / Everyone 10+ 671.

#### Step 2 — `train_models.py`: features, selection budget, speed

1. Extend `BASE_NUM`:
   ```python
   BASE_NUM = ["size_mb", "price", "price_is_positive", "app_age_days",
               "days_since_update", "developer_app_count", "min_android",
               "ad_supported", "in_app_purchases", "editors_choice"]
   ```
2. `build_matrix()` — copy the new numeric columns; emit `content_rating`
   alongside `category`.
3. `make_preprocessor()` — one-hot block becomes `["category", "content_rating"]`.
4. **Artifact-size budget in selection** (the principled fix for the 443 MB
   problem): after fitting each candidate, serialise it and reject any
   candidate over a budget (default ~10 MB, `--max-artifact-mb`). Rejected
   candidates are recorded in `metrics.json` with the reason, so the report
   stays honest. Effect: RF-300 unbounded is rejected; GB wins on merit.
5. Speed levers that are **safe** (do not change results or break export):
   - `n_jobs=-1` on both Random Forest candidates — **measured bit-identical**
     (fixed `random_state`), 1.9× faster here, scales with cores.
   - `n_iter_no_change` + `validation_fraction` on Gradient Boosting — early
     stopping; confirmed present in the signature.
   - `subsample=0.8` (stochastic GB) — ~20% cheaper per stage.
   - Cap the unbounded tree: `min_samples_leaf` / `max_depth` on Decision Tree
     and Random Forest → smaller, faster, and keeps RF exportable.
6. Memory for Track B: `OneHotEncoder(sparse_output=True)` where the estimator
   accepts sparse, or downcast numerics to `float32`. The dense design matrix
   on the full run is ~940 MB (2.31M × ~51 × 8 bytes).

#### Step 3 — `browser_export.py`: export the new model family

1. Add `GradientBoostingClassifier` to `SUPPORTED_ESTIMATORS` and export it:
   `init` = `log(class_prior_)`, per-class tree ensembles `(200, 4)`,
   `learning_rate`, `classes_`.
2. Extend the preprocessing export: 10 numeric columns + 2 categorical
   (`category`, `content_rating`).
3. Fix `make_parity_cases()` — it currently builds frames from the old feature
   list and fails with `not in index` once new columns exist. Generate the new
   features too (randomised ages/dates/flags, some unseen categories, some
   missing values).
4. Skip-and-warn (never crash) on unsupported families.

#### Step 4 — `ml_inference.js`: implement multiclass gradient boosting

1. `predictProba` for `GradientBoostingClassifier`:
   `score_k = log(prior_k) + lr × Σ_trees leaf_k`, then softmax.
2. Handle the wider feature vector (10 numeric + 2 one-hot blocks).
3. Keep `handle_unknown='ignore'` semantics for both categoricals.
4. **This is the correctness gate:** `tests/browser_inference.test.js` must
   match sklearn to ~1e-11 on every parity case before anything merges.

#### Step 5 — `app.py`: optional new inputs

Extend `TierIn` (and `RatingIn` for M1) with optional fields for the new
features, defaulting to `None`. Build the row with training medians for
numeric and the mode for categorical, and append the imputation to the
returned `assumptions` list so the UI states it.

#### Step 6 — Frontend forms  ✅ DONE — option 2 (confirmed 2026-09-19)

The M2 form now asks for **content rating, minimum Android, and the
ad-supported / in-app-purchase / Editors' Choice flags** (each with
"Not specified"), while **app age, days since the last update and developer
portfolio size** stay imputed to their training medians. Every imputation is
listed under the result with the value actually used, e.g.
*"app age (days since release) not provided → imputed to the training median
(842)"*. The M1 form is unchanged (4 fields); its imputations are listed the
same way.

Locked by a new `tests/smoke_frontend.js` assertion: the tier submit sends
`content_rating` / `min_android` / `ad_supported` / `in_app_purchases`, and a
blank Editors' Choice stays `null` (imputed, never guessed as "no").

#### Step 7 — Tests

- `tests/browser_inference.test.js`: new parity cases incl. the new features,
  unseen content ratings, and missing values.
- `tests/smoke_frontend.js`: unchanged expectations unless the form changes.

#### Step 8 — Docs

Update `README.md` metrics tables and the M2-B caveat (it will no longer be
below the majority baseline), and record the final timings here.

### 1.3 Expected outcome

| Model | Before | After |
|---|---|---|
| M2-B accuracy | 0.6729 (below 0.7242 baseline) | **0.7682** (above it) |
| M2-B macro-F1 | 0.3067 | **0.5440** |
| M2-B artifact | 1.0 MB (Decision Tree) | **1.0 MB** (Gradient Boosting) |
| M1 R² | +0.0506 | +0.0711 |
| M2-A macro-F1 | 0.8275 | 0.8385 |

Training cost (40k sample): GB-200 currently 23.5 s per fit; early stopping
and `subsample` should reduce it. Full 2.3M run projected at roughly 1–1.5 h
with exact GB, versus 15–45 min if HistGB were used — the price of keeping
browser parity.

### 1.4 Risks

| Risk | Mitigation |
|---|---|
| New features leak across the split | `developer_app_count` is computed over the whole frame; the ablation showed only ~0.015 macro-F1 depends on it. Consider computing it on training rows only and documenting the choice. |
| `content_rating` unseen at predict time | `handle_unknown='ignore'` → all-zero one-hot; already the behaviour for `category`. |
| Browser bundle grows | GB adds ~12k nodes; measure models.js and keep it under a few MB. |
| Parity breaks silently | Gate the merge on `tests/browser_inference.test.js`. |
| Selection rejects everything | Budget must leave at least one candidate; assert and fail loudly otherwise. |

### 1.5 Verification checklist — all green (2026-09-19)

- [x] `python clean.py` — 8 new columns present, no all-NaN columns
      (`app_age_days` 38,673/40,000 · `days_since_update` 40,000 ·
      `developer_app_count` 39,999 · `min_android` 38,913 · flags 40,000 ·
      `content_rating` 40,000); `apps.json` rows byte-identical to before
- [x] `python train_models.py` — M2-B **0.7682 / 0.5440**, Gradient Boosting
      selected, 0.99 MB; RF rejected at 334 MB
- [x] `node tests/browser_inference.test.js` — parity at **5e-11** (180 cases)
- [x] `node tests/smoke_frontend.js` — all assertions pass
- [x] `python app.py` → `/api/health` reports `models_loaded: true`
- [x] API prediction == browser-engine prediction (4 tier cases + 1 rating
      case; residual differences are the API's own 4-dp rounding)
- [x] Static-host / `file://` path still predicts (covered by the smoke test's
      no-backend section)
- [x] README metrics + M2-B caveat + viva script updated
- [x] 11-row mechanical sample still trains (`sample_apps.csv` → the 7 derived
      columns are all-NaN and are dropped from the feature set)

### 1.6 Open question — form UX  ✅ RESOLVED: option 2

The user chose **option 2** — expose content rating, minimum Android and the
ad / IAP / Editors' Choice flags; impute app age, days-since-update and
developer portfolio size to training medians and list each as an assumption.
Implemented (see Step 6).

## Phase 2 — Optimise and professionalise the website  ✅ DONE

**Status: complete 2026-09-19 (commit `34e6a93`).** All four sub-parts are
implemented and covered by new smoke assertions. **The layout and the chart
timings still need one look in a real browser** — this sandbox has no browser
(Playwright's Chromium download and the Debian mirrors are both unreachable),
so §2.2 and §2.3 were verified structurally, not visually. See
"Verifying Phase 2 in a browser" at the end of this section.

### What actually changed

**2.1 Hamburger.** Below **1080px** the sidebar leaves the grid and becomes a
fixed drawer (`transform:translateX(-100%)` → `.open{transform:none}`), with a
backdrop. Previously it ate 250px down to 760px and then stacked *above* the
content, pushing the dashboard down a full screen. Toggle lives in the topbar
(`aria-expanded` / `aria-controls="sidebar"`); closes on nav-link click,
backdrop click, `Escape`, and when the viewport crosses back to desktop — so it
cannot be left stuck.

**2.2 Right-edge overflow.** Root cause was the grid: a bare `1fr` track has an
automatic minimum of `auto`, so a wide Plotly container or metric table
expanded the column past the viewport instead of shrinking. Fixed with
`grid-template-columns:250px minmax(0,1fr)` + `min-width:0` on `main` and on
grid/flex children. Chart 1's legend no longer sits outside the plot area
(`x:1.02` + `margin.r:130`), which was the other candidate.
`body{overflow-x:hidden}` was **masking** this, not fixing it; it stays as a
backstop only.

**2.3 Charts 1 and 5.** Measured first: the JS data preparation is
**free** — chart 1's 48 `filter()` sweeps over 40k rows take **< 0.1 ms**, so
the cost is entirely Plotly's DOM/SVG work, not our loops. That inverted the
priority list:

| Change | Why |
|---|---|
| Lazy render via `IntersectionObserver` (400px `rootMargin`) + a yield before each plot | All six charts were plotted synchronously at bootstrap, so the main thread was blocked until every one finished — the page felt frozen wherever you happened to be looking. Biggest win. |
| Chart 1 → `scattergl` when WebGL is available (SVG fallback otherwise) | ~30k markers: that is ~30k SVG nodes vs one draw call per trace. |
| Chart 1: drop the 1px per-marker stroke | Roughly doubles paint cost for almost no visual gain at bubble size. |
| Chart 1: single-pass grouping, `max` in a loop | 48 `filter()` sweeps → one `Map`; `Math.max(...30k)` can overflow the call stack. Correctness, not speed. |
| Chart 1: no 48-row legend; hover carries the category | Unreadable at any width, and it was the thing drawn outside the plot area. |
| Chart 5: colour rank precomputed in a `Map` | `order.indexOf()` per point was O(n²). Trivial at n=48, wrong in principle. |
| Resize debounced to 150 ms | `responsive:true` already handles container changes; the window handler was relayouting all seven plots once per resize event. |

Chart 4 was **not** touched (it is not slow — a 16-bin bar chart, one trace).

**2.4 Prose.** Every section description is now a one-line claim with the
justification behind a `<details>` disclosure ("Method & caveats",
"Why this replaced the growth curve", …). Nothing was deleted: the
data-integrity caveats are still all there, just not walls of text.

### Measured render times — round 1 (browser, 40,000 rows)

The dashboard posts its own timings to the backend (`?bench=1` panel + a
temporary `/api/bench` sink), because the preview is behind a tokened proxy:
no DevTools, no shareable URL. Round 1, after the lazy-rendering work:

| chart | round 1 | root cause | fix |
|---|---|---|---|
| 1 — size × rating | **1403 ms** | 48 traces (one per category) → 48× Plotly's per-trace setup | collapse to 1 trace, per-point colour array; memoise `toLocaleString` (14 distinct installs values, 17k calls) |
| 2 — correlation | **926 ms** | 25 sweeps of 40k rows building ~1.3M pair arrays — **data prep, not Plotly** | materialise columns once into `Float64Array`, pairwise Pearson: 159.5 → 22.7 ms (7.0×) |
| 3 — last-updated month | 434 ms | Plotly drawing 1,143 stacked SVG bars | none — inherent to SVG bars |
| 4 — rating histogram | 159 ms | not slow | untouched (by request) |
| 5 — category scatter | 178 ms | — | already fixed |
| 6 — pricing | 189 ms | — | untouched |
| **total** | **3289 ms** | | |

### Measured render times — round 2 (same browser, after the fixes)

| chart | round 1 | round 2 | change |
|---|---|---|---|
| 1 — size × rating | 1403 ms | **401 ms** | −1002 ms (−71%) |
| 2 — correlation | 926 ms | **235 ms** | −691 ms (−75%) |
| 3 — last-updated month | 434 ms | **344 ms** | −90 ms (−21%) |
| 4 — rating histogram | 159 ms | **114 ms** | −45 ms (−28%, untouched) |
| 5 — category scatter | 178 ms | **132 ms** | −46 ms (−26%, untouched) |
| 6 — pricing | 189 ms | **123 ms** | −66 ms (−35%, untouched) |
| **total** | 3289 ms | **1349 ms** | **−1940 ms (−59%)** |

Charts 4, 5 and 6 were **not touched** yet all dropped 26–35%, so roughly a
quarter of the improvement is environmental (warm cache, scroll speed, less GC
pressure from the removed 1.3M throwaway arrays). Charts 1 and 2 improved
71–75% — well clear of that baseline, so the fixes account for the bulk of it.

Reading the result: the six charts now cost 114–401 ms each, paid one at a time
as each scrolls into view, instead of 3.3 s up front. No single chart is over
the ~400 ms jank threshold any more.

**Overflow, finally confirmed in a browser:** the page reported
`scrollWidth 871 ≤ innerWidth 881` at a preview width of 881 px — i.e. below
the 1080 px breakpoint, which is the narrow case that mattered. No horizontal
overflow. ✅

### Round 3 (tried and REVERTED — do not retry)

Replacing chart 1's 17,247-entry hex colour array with palette indices plus a
10-stop `marker.colorscale` — the theory being that Plotly parses every string
in a per-point colour array, so numbers would be cheaper. **It was slower
(467 ms vs 401 ms) and the categories stopped rendering.** Reverted in
`b6d7ed9`.

Conclusion: chart 1's remaining ~400 ms is Plotly's own WebGL path, not data
prep. It is not reachable by further array reshaping. Chart 1 is now 3.5× faster
than it was and no single chart exceeds the jank threshold — **the chart work
is done**.

WebGL **was** available (`scattergl` engaged), so chart 1's cost was not the
SVG fallback — it was the trace count.

Notes on the two fixes:

- Chart 1 keeps the same picture and hover: every marker keeps its category
  colour (now a per-point array) and `customdata` still names the category.
  Verified: 1 trace, 17,247 markers, 10 distinct colours, every hover valid.
- Chart 2 uses the **two-pass** Pearson (means, then products of deviations),
  not the cheaper one-pass `n·Σxy − Σx·Σy`, which loses precision to
  catastrophic cancellation at n = 40,000. Output verified identical to the
  old matrix to **1.8e-12**. The rewrite also removes a latent
  `Math.max(...40k_values)` spread, which can overflow the call stack.

**Awaiting round 2 numbers** from the same browser to confirm the effect.

### Verifying Phase 2 in a browser

1. **Overflow:** at 1440 / 1080 / 768 / 375 px, run
   `document.documentElement.scrollWidth <= window.innerWidth` in the console.
   It must be true at every width (this is the assertion `body{overflow-x:hidden}`
   was hiding).
2. **Hamburger:** below 1080px the sidebar must be off-screen until the Menu
   button is pressed; Tab to it and press Enter; `Escape` closes it. Above
   1080px the button must not exist.
3. **Chart timings:** the dashboard reports them itself. With `?bench=1` a
   panel appears bottom-right, and the same numbers are POSTed to `/api/bench`
   and printed to `python app.py`'s stdout — use that if the preview cannot be
   opened in a new tab.
4. If WebGL is unavailable in your browser, chart 1 silently falls back to
   `scatter` (SVG). The bench panel's footer line says which path ran.

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

### 2.5 Design review pass (2026-09-20) — aurora palette kept

A generic design review was pasted in (visual hierarchy, colour system, forms,
buttons, loading states, mobile, accessibility, error handling, plus a
typography/spacing scale). Much of it is a template that does not apply — it
proposes a **Google Blue (#4285f4)** rebrand and asks for SQL query editors,
query-history panels and Query/Results/Visualize tabs, none of which exist in
this dashboard. **Decision: apply what fits, keep the aurora identity.** Those
items are deliberately **not** being done.

Applied:

| Item | What changed |
|---|---|
| Focus visibility | **Bug fixed:** `--purple` / `--purple-lt` were referenced by the focus rules but never declared, so focus rings computed to no colour. Declared; one `:focus-visible` ring on every interactive element (WCAG 2.4.7, 1.4.11). |
| Skip navigation | WCAG 2.4.1 skip-to-content link. |
| Contrast | Full WCAG 2.2 AA audit, `--text-low` and `--purple-neon` corrected (see §2.6), enforced by test. |
| Loading states | Both predict buttons: disabled + `aria-busy` + spinner, cleared in a `finally`. They previously stayed clickable and were double-submittable. |
| Validation | `:user-invalid` styling — fires only after interaction, so an untouched form is not pre-flagged. |
| Optional fields | One "every field is optional" note per form. |
| Touch targets | 44 px minimum ≤768 px (WCAG 2.5.5); pills clear the 24 px AA minimum. |
| Design tokens | `--text-xs…2xl`, `--space-1…8`, `--danger/success/warning/info`. |
| Reduced motion | Spinner honours `prefers-reduced-motion`. |

**Not applied, with reasons:** Google Blue rebrand (destroys the existing
identity); SQL/query editor, query history, Query/Results/Visualize tabs (not
features of this dashboard); toast notifications (results already render
inline, so toasts would duplicate them); sortable/filterable data tables and
pagination (the tables are small model-comparison summaries, not browseable
datasets); syntax highlighting (no code output is shown).

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

> The viva material Phase 3 removed from the README is preserved in git:
> `git show 8e8edf0:README.md` — final section, "Viva explanation (verified
> work only)". Use it as the seed for items 1, 3 and 4.

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

## Phase status

| Phase | Status | Commits |
|---|---|---|
| Phase 1 — enriched features + faster training | ✅ **DONE** (2026-09-19) | `13e019f`, `4a4ea57` |
| Phase 2 — website: hamburger, overflow, charts 1 & 5, trim prose, design-review pass | ✅ **DONE** (2026-09-20) — verified in a browser: no horizontal overflow, render 3289 → 1349 ms, chart 1 sizes fixed, WCAG 2.2 AA met | `34e6a93`, `7a38901`, `3c0e6c8`, `e1305e0`, then the later UI passes `55495b5`…`8e8edf0` |
| Phase 3 — professionalise the README | ✅ **DONE** (2026-09-20) — TOC with 22 validated anchor links; 4-line lead carrying the final metrics; no-backend mode split into its own section; stale sample MD5 corrected to `7bb2de92ec61`; viva section removed (Phase 4 seed preserved in git) | `38afa0d` |
| Phase 4 — `VIVA_PREP.txt` | ⬜ last (must quote the final metrics) | — |

Branch `arena/01a0ba27-google-analyst0-1`. PR **#3** is open and **not merged**
— merging it is what republishes GitHub Pages.

## Sequencing and dependencies

```
Phase 1 (models)      ──┐
Phase 2 (website)     ──┼── independent of each other; any order
Phase 3 (README)      ──┘   (but 3 should come after 1 so the numbers are final)
Phase 4 (viva file)   ── last: must quote the FINAL metrics
```

- ~~The **open decision** in §0 (adopt enriched features?)~~ **RESOLVED** —
  adopted, shipped as Gradient Boosting (§0 + Phase 1). The metrics Phases 3
  and 4 quote are final: M2-B 0.7682 / 0.5440, M1 R² +0.0711.
- Phase 2's chart refactor touches `tests/smoke_frontend.js` assertions
  (48 traces → 1). Update those assertions deliberately.

## How to resume this plan

1. `git checkout arena/01a0ba27-google-analyst0-1 && git pull`
2. Confirm state: `node tests/browser_inference.test.js` and
   `node tests/smoke_frontend.js` must both print `ALL CHECKS PASSED`.
3. Start from the **Phase status** table above, then take phases in the order
   given. Each phase is independently shippable — commit and push at the end
   of each one rather than batching.
4. Scratch experiments live in `/tmp` (`/tmp/feat`, `/tmp/feat3`) and are
   **not** persisted — they hold the enriched-feature patches and the measured
   metrics JSON. If those numbers are wanted again, re-derive them; do not
   assume `/tmp` survived.
5. Merge PR #3 into `main` only when told to — that is what republishes GitHub
   Pages.
