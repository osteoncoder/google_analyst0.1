# APEX — Play Store Intelligence Dashboard

Aurora-themed Google Play Store analytics dashboard with two machine-learning
models (rating regression + install-tier classification).

**Dataset:** the MIT-licensed [gauthamp10/Google-Playstore-Dataset](https://github.com/gauthamp10/Google-Playstore-Dataset)
scrape (June 2021, **2,312,944 apps**, 24 attributes). The repository ships a
deterministic **40,000-row stratified sample** of it as the committed default,
so everything runs out of the box; one command downloads the full 666 MB CSV,
after which the same pipeline runs at full scale.

**Stack:** plain HTML/CSS/JS + Plotly.js from CDN for the charts. Inference
runs either through a small FastAPI backend or — on any static host, with no
Python at all — through `ml_inference.js`, which evaluates the same saved
scikit-learn pipelines exported to plain JS. Nothing retrains per request.
No build step, no npm.

```
.
├── index.html                 # page structure, sections 01-09
├── style.css                  # aurora/glass theme + ML section styles
├── data.js                    # dataset loader + JS mirror of the cleaning rules
├── charts.js                  # KPI strip + charts 01-06 (all data-driven)
├── ml_inference.js            # evaluates the exported pipelines in the browser
├── ml_dashboard.js            # sections 07-09 + bootstrap + unavailable states
├── browser_export.py          # publishes dataset + fitted models as <script> bundles
├── clean.py                   # single-source-of-truth cleaning pipeline (CLI)
├── fetch_dataset.py           # downloads the full 2.31M-row dataset + builds the sample
├── train_models.py            # reproducible scikit-learn training (M1, M2 A/B)
├── app.py                     # FastAPI: static dashboard + /api inference
├── requirements.txt           # Python dependencies
├── data/
│   ├── playstore_sample.csv       # committed 40k stratified sample (the default)
│   ├── playstore_sample.meta.json # its provenance: source, quotas, strata, tier shares
│   ├── raw/playstore_full.csv     # full 2.31M dataset — built by fetch_dataset.py (git-ignored)
│   ├── play_store.csv             # older 10,841-row export (fallback)
│   ├── sample_apps.csv            # 11-row mechanical-test sample (last resort)
│   ├── apps_cleaned.csv           # output of clean.py (ALL cleaned rows)
│   ├── apps.json                  # output of clean.py (capped export, feeds the dashboard)
│   ├── apps_bundle.js             # the same payload as a <script> (works on file://)
│   └── cleaning_report.json       # provenance: raw/cleaned/removed counts, field stats
├── ml/artifacts/              # output of train_models.py (pipelines + metrics)
│   └── browser/
│       ├── models.js              # fitted pipelines, flattened for ml_inference.js
│       └── parity_cases.json      # inputs + sklearn outputs (engine correctness fixture)
└── tests/
    ├── smoke_frontend.js          # headless Node check of the whole frontend
    └── browser_inference.test.js  # JS engine must match scikit-learn exactly
```

## Run it (full version with ML)

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt   # Windows: .venv\Scripts\pip ...

# optional: get the FULL 2.31M-row dataset (downloads 3 parts, combines them,
# and re-writes data/playstore_sample.csv + its provenance sidecar)
.venv/bin/python fetch_dataset.py --sample 40000

.venv/bin/python clean.py          # 1. clean raw data  -> data/apps_cleaned.csv, data/apps.json
.venv/bin/python train_models.py   # 2. train models    -> ml/artifacts/
.venv/bin/python app.py            # 3. serve           -> http://localhost:8000
```

Open http://localhost:8000. The KPI strip, charts 01-06 and the ML sections
all read from the same cleaned dataset; sections 07-09 call the API.

`clean.py` auto-detects its source, first match wins:

| # | Path | What it is |
|---|---|---|
| 1 | `data/raw/playstore_full.csv` | full 2,312,944-row dataset (after `fetch_dataset.py`) |
| 2 | `data/playstore_sample.csv` | committed 40,000-row stratified sample (default) |
| 3 | `data/play_store.csv` | older 10,841-row Kaggle export (fallback) |
| 4 | `data/sample_apps.csv` | 11-row sample (last resort for a fresh clone) |

So `python clean.py` works in a fresh clone, and silently switches to full
scale as soon as the full CSV exists. `--raw <path>` overrides everything.

**Every other way of opening the page — double-click `index.html`, VS Code
"Run Active File", Live Server, GitHub Pages — works fully, with no Python
process at all.** The dashboard is plain HTML/CSS/JS, so both the dataset and
the models are also published as plain `<script>` bundles:

| What | Where | Why |
|---|---|---|
| dataset | `data/apps_bundle.js` | `fetch()` is **blocked** on a `file://` page, so `data/apps.json` is unreachable there — a classic `<script>` tag is not |
| models | `ml/artifacts/browser/models.js` | a static host has no `/api/predict/*`, so the fitted pipelines are exported and evaluated in the page |

`ml_inference.js` walks those exported numbers — impute → scale → one-hot →
decision trees — so the forms return the *same* prediction the backend does,
and the UI states which engine answered. `tests/browser_inference.test.js`
replays 180 inputs and requires an exact match with scikit-learn (max error
~5e-11); `python browser_export.py` regenerates the bundles, and `clean.py` /
`train_models.py` call it automatically. Both bundles are **committed**, so a
ZIP download works offline straight away.

Sections 08-09 read the **measured** metrics snapshot
(`ml/artifacts/metrics.json` — the same file the API serves) and label it
clearly as a snapshot, so the dashboard degrades gracefully instead of going
blank. If neither engine is reachable the forms show an actionable message
plus a **Retry** button rather than inventing values, and the metrics fetch
retries once automatically, which covers opening the page while the server is
still starting.

> **The one thing to avoid:** deleting `data/apps_bundle.js` (or the
> `ml/artifacts/browser/` folder) and then opening `index.html` off the disk.
> With no bundle and no `fetch()`, the page falls back to its 11-row
> `EMBEDDED_SAMPLE` — you will see a warning banner saying so, and only 11
> records in the header. Re-run `python clean.py` to rebuild the bundle.

### Which engine is answering?

| You opened | Dataset | Predictions |
|---|---|---|
| `python app.py` → http://localhost:8000 | `data/apps.json` | FastAPI backend (`/api/predict/*`) |
| GitHub Pages / Live Server / any static host | `data/apps.json` | **in-browser** exported pipeline |
| double-click / VS Code "Run Active File" (`file://`) | `data/apps_bundle.js` | **in-browser** exported pipeline |

## Data pipeline (`clean.py`) — documented rules

1. **Identity**: rows missing `App`/`App Name` or `Category` are dropped (counted).
2. **Corrupted shifted rows**: numeric category + out-of-range numeric rating
   (the canonical export's famous shifted row) → dropped and counted. 0 in the
   primary dataset.
3. **Installs**: `"1,000,000+"` / `1000000` → integer **lower bound** of the
   reported download band. Play reports bands (10 … 1B+); the number is the
   band floor, **not** an exact download count. All UI labels say so.
4. **Reviews** (`Rating Count` in the primary dataset): integer; blank → 0.
5. **Rating**: kept only if 1 ≤ r ≤ 5. The primary dataset encodes *unrated* as
   `0.0` (≈ 47% of rows) → those become missing and are counted, so M1 trains
   on real ratings only.
6. **Size**: `86M` → 86 MB, `72K` → 72/1024 MB, `Varies with device` → missing
   (a real state, imputed later by the model pipeline, never by hand).
7. **Price**: `""`/`0` → 0.0. **Legitimate zeros are preserved**, never
   imputed to nonzero values. A dataset with no Price column gets *unknown*
   prices, not $0. Listed price is a price tag, **not revenue** — no revenue
   figure exists anywhere in the project.
8. **Last Updated**: `Jan 15, 2024` / `2024-01-15` / …; unparseable → missing.
9. **Category**: trim + title case + one documented typo fix
   (`commication` → `Communication`). No other names are altered or merged.
10. **Duplicates**: exact duplicates on
    (app, category, rating, reviews, installs, size_mb, price) → keep first.
    Same-name rows with different metadata are kept (different listings) and
    counted.
11. **Eligibility**: M1 rows = valid rating; M2 rows = valid installs bound.
12. **Dashboard export cap**: `apps.json` is fetched by the browser, so it
    carries at most `--max-json-rows` rows (default 60,000) — chosen by the
    same deterministic content-hash selection as the committed sample,
    stratified by install tier with a 250-row floor. `apps_cleaned.csv` and ML
    training always use **all** cleaned rows, and the report records exactly
    how many rows were exported.

`cleaning_report.json` records raw rows, cleaned rows, everything removed,
per-field missingness, rows used per model, and (for sampled sources) the
sampling provenance. Cleaning and feature preparation are the same code path
for training (`train_models.py`) and for the dashboard (`data/apps.json`), and
prediction uses the **saved** preprocessor inside the pipeline joblib file —
inference can never drift from training.

## ML models

### Protocol (both models, both M2 versions — identical)

- Fixed seed `42` everywhere.
- 80/20 train/test split **before any preprocessing**, via `GroupShuffleSplit`
  on app name → the same app can never land on both sides (duplicate leakage
  guard).
- The training part is split 75/25 into train/validation (same grouping).
- Imputation, one-hot encoding and scaling are fit on the **train slice only**,
  inside a `sklearn.pipeline.Pipeline`.
- Candidates are selected on **validation** metrics only; the winner is
  refit on train+validation and evaluated on the test set **exactly once**.
  No test-set tuning. (Candidate tables in section 09 also list each
  candidate's test score for transparency — selection never uses it.)

### M1 — Rating regression (`m1_rating/`)

- Target: `Rating`. Inputs: `category, size_mb, price, price_is_positive,
  log1p(reviews)`. **Rating itself and Installs are excluded from inputs.**
- Candidates: mean-prediction baseline, Linear Regression, Decision Tree,
  Random Forest, Gradient Boosting.
- Reported: actual MAE / RMSE / R² (validation + final test), per candidate.

### M2 — Four-class install-tier classification (`m2_tier_*/`)

- Target: install band of the **reported installs lower bound**,
  left-inclusive, log-spaced (default; aligned with Play's own band grid):

  | Tier | Bound |
  |---|---|
  | Under 10K | [0, 10,000) |
  | 10K-1M | [10,000, 1,000,000) |
  | 1M-100M | [1,000,000, 100,000,000) |
  | 100M+ | [100,000,000, ∞) |

  Override with one flag (propagates to training and dashboard):
  `python clean.py --tiers "Name:lo:hi,Name:lo:hi,..."` (hi may be `inf`).

- Candidates: dummy (class-prior) baseline, Logistic Regression, Decision
  Tree, Random Forest, Gradient Boosting; ranked by **validation macro-F1**
  (chosen for class imbalance — see the results below, where plain accuracy is
  actively misleading).
- Two versions, **same split, same protocol**:
  - **A — with Reviews** (`m2_tier_with_reviews/`)
  - **B — without Reviews** (`m2_tier_without_reviews/`) ← **primary dashboard
    model** (section 08). `Installs` and tier labels never appear in the
    inputs of either version.
- Reported per version: accuracy, macro-F1, weighted-F1, per-class
  precision/recall/F1/support, confusion matrix, and the A-vs-B difference.

**Why B is primary / what Reviews means:** Reviews is a strong proxy for
installs (more installed apps accumulate more reviews), so version A measures
how well a popularity signal predicts popularity. An accuracy drop in B does
**not by itself prove temporal leakage**, and removing Reviews does **not**
make every remaining feature leakage-free (category/size can still carry
popularity information). Neither model is a **pre-launch** or
**future-growth** predictor: the data is a cross-sectional store snapshot and
evaluation is in-sample-time. Section 09 states all of this on the page.

### Features available at prediction time

Exactly what the forms ask for (that is deliberate):

- M1: category, size (MB), price ($), review count.
- M2: category, size (MB), price ($).
- Omitted fields: size → training median (imputer), price → $0, reviews → 0.
- **Category normalization**: an incoming category is passed through
  `clean.py`'s own `clean_category()` (trim, underscores/hyphens → spaces,
  title case, the one documented typo fix) *before* encoding — the same
  transformation the training categories went through. A hand-typed
  `"education"` therefore matches the trained `"Education"` category instead
  of silently falling into the "unseen category" bucket. The response reports
  the category actually used (`category_used`) and whether it was changed.
- Unknown category → `handle_unknown='ignore'` one-hot (encoded as "not seen
  in training"), **and the response says so explicitly** in its assumptions
  list, naming the category and how many the model knows. A category that is
  blank after normalization (e.g. `"___"`) is rejected with a 422 rather than
  being scored as a category named " ".
- Review counts are counts: a fractional value (e.g. `1200.5`) is rounded,
  not rejected.
- The M1 prediction is clipped to the target's own domain, `[1, 5]`, and the
  response notes when clipping was applied.
- The API returns its assumptions with every prediction, and validation
  errors (422) are rendered as readable `field: reason` messages in the UI,
  not raw JSON.

## Dataset (provenance)

- **Primary:** [gauthamp10/Google-Playstore-Dataset](https://github.com/gauthamp10/Google-Playstore-Dataset)
  — a June-2021 Play Store scrape of **2,312,944 apps × 24 attributes**,
  licensed **MIT**. Upstream stores it as three `Part?.csv.tar.gz` files
  (666 MB of CSV); only Part1 carries the header, and upstream's own
  instructions are `cat Part?.csv > dataset.csv`. `fetch_dataset.py` performs
  that byte-level concatenation with a newline-safety check and writes
  `data/raw/playstore_full.csv` (git-ignored — too large for git).
- **Committed default:** `data/playstore_sample.csv` — a **40,000-row
  stratified sample** of the above, plus `data/playstore_sample.meta.json`
  recording source URL, license, quota rule, and per-stratum
  full/sampled counts. Selection is deterministic (smallest blake2b content
  hash per stratum), so it reproduces on any machine without an RNG seed.
  It is stratified by (48 categories × 4 install tiers = 211 non-empty
  strata); every non-empty stratum keeps ≥ 25 rows, which deliberately
  **over-represents the rarest tiers** (100M+ appears at 1.2% in the sample vs
  0.03% in the full data) so no class is missing from training. Full-scale
  proportions return on a full run; `tier_shares` in the sidecar records both.
- **Column mapping** (the 24-column source → the pipeline's canonical fields;
  the remaining columns are kept in the raw/sample CSV for future work):

  | Source column | Canonical | Notes |
  |---|---|---|
  | `App Name` | `app` | identity |
  | `Category` | `category` | already title-cased in this source |
  | `Rating` | `rating` | `0.0` = unrated → missing |
  | `Rating Count` | `reviews` | |
  | `Installs` | `installs` | band floor, e.g. `50,000+` |
  | `Size` | `size_mb` | `7.0M`, `72K` |
  | `Price` | `price` | 0 → free |
  | `Last Updated` | `last_updated` | `Apr 29, 2020` |

- **Fallbacks:** `data/play_store.csv` (older 10,841-row export) and
  `data/sample_apps.csv` (11 rows) remain for offline/mechanical use.
- The pipeline is dataset-agnostic: any CSV/XLSX with the same spirit of
  columns (aliases handled) works via `python clean.py --raw <path>`.

### Measured cleaning report

**Committed 40,000-row sample** (`python clean.py`, the default path):

| Step | Count |
|---|---|
| Raw rows | 40,000 |
| Dropped (missing identity / corrupted / exact duplicates) | 0 / 0 / 0 |
| Same-name variants kept (different listings) | 396 |
| **Cleaned rows** | **40,000** |
| Rows with valid rating → M1 | 22,418 (56.0%) |
| Rows with valid installs → M2 | 39,893 |
| Missing size (`Varies with device` etc.) | 1,576 (3.9%) |
| Price: free / paid / unknown | 39,303 / 697 / 0 |
| Categories | 48 |
| Tier mix: Under 10K / 10K-1M / 1M-100M / 100M+ | 29,023 / 8,461 / 1,916 / 493 |

**Full 2,312,944-row dataset** (`python fetch_dataset.py` then `python clean.py`,
measured in a 2-core / 3.8 GB sandbox: **80 s wall, 3.0 GB peak RSS**):

| Step | Count |
|---|---|
| Raw rows | 2,312,944 |
| Dropped missing identity | 5 |
| Exact duplicates removed | 717 |
| **Cleaned rows** | **2,312,222** |
| Rows with valid rating → M1 | 1,230,293 (53.2%) |
| Rows with valid installs → M2 | 2,312,115 |
| Missing size | 74,956 (3.2%) |
| Price: free / paid / unknown | 2,267,293 / 44,929 / 0 |
| Categories | 48 |
| Tier mix: Under 10K / 10K-1M / 1M-100M / 100M+ | 1,794,823 / 469,347 / 47,261 / 684 |
| `apps.json` export (capped) | 60,000 of 2,312,222 rows |

### Measured evaluation results (committed sample, seed 42, reproducible)

**M1 — Rating regression** (n=22,418; test n=4,482; selected by validation R²;
training ratings: mean 4.10, median 4.20):

| Model | MAE (test) | RMSE (test) | R² (test) |
|---|---|---|---|
| Mean baseline | 0.522 | 0.679 | −0.000 |
| Linear Regression | 0.511 | 0.671 | +0.024 |
| Decision Tree | 0.675 | 0.909 | −0.793 |
| Random Forest | 0.529 | 0.703 | −0.071 |
| **Gradient Boosting (selected)** | **0.503** | **0.661** | **+0.051** |

Honest reading: ratings in this dataset are high and tightly clustered
(median 4.2), so "always predict the mean" already achieves MAE 0.52 — the
model adds only ~0.02 MAE and R² ≈ 0.05. Rating is driven by app *quality*,
which none of these features capture. A defensible negative result, not a
failure.

**M2 — Install-tier classification** (n=39,893; test n=7,967):

| Version | Model (selected) | Accuracy (test) | Macro-F1 (test) | Weighted-F1 (test) |
|---|---|---|---|---|
| A — with Reviews | Logistic Regression | 0.907 | 0.827 | 0.906 |
| **B — without Reviews (primary)** | **Decision Tree** | **0.673** | **0.307** | **0.637** |
| Baseline — class prior | Dummy | 0.724 | 0.210 | 0.318 |

Two honest observations, both stated on the page:

- **Accuracy is misleading here.** 72.6% of the sample is `Under 10K`, so the
  do-nothing baseline scores 72.4% accuracy — *higher* than model B's 67.3%.
  On macro-F1 (the documented selection metric, chosen for exactly this
  reason) B improves 0.210 → 0.307, i.e. it carries real signal about the
  minority tiers, but the feature set is genuinely weak for this task.
- **The A→B gap is large** (macro-F1 0.827 → 0.307): Reviews acts as a strong
  proxy for install volume, which is precisely why the without-Reviews model
  is the primary one. Random-Forest importances for B are dominated by
  `size_mb` (0.79).

Full per-class P/R/F1/support, confusion matrices and per-candidate
validation/test scores are rendered in sections 09/08 from
`ml/artifacts/metrics.json`.

## What was fixed vs the original (and why it is defensible)

| Original | Problem (verified in code) | Now |
|---|---|---|
| Chart 02 "Global Reach" | hardcoded `['USA','IND','DEU','FRA','GBR']` as "representative markets" — no location data exists | Pearson correlation heatmap over genuine numeric columns (log1p for skewed Reviews/Installs); "association, not causation" stated |
| Chart 03 "Category Trajectory" | hardcoded months + growth factors `[1.0,1.18,1.42,1.75]` | "Apps by Last-Updated Month" (real dates when present); explicit data-gap card when the dataset has no date column — no invented trend |
| Chart 04 "Market Expansion" | fixed multipliers `[1.0,1.25,1.55,1.85]` | rating-distribution histogram (directly observed) |
| Chart 05 | ratings (1-5) and millions of reviews on one shared log axis | one point per category: Σ reviews (log, zero-safe) vs mean rating (unweighted); aggregation stated |
| Chart 06 "Monetization" | invented revenue (`free installs × 0.05 × $2.99`), dual axes with incompatible units | pricing mix from a real Price column; without one: explicit banner + app counts; **no revenue is ever estimated**; $0 prices preserved |
| KPI strip | invented deltas ("+18.6% MoM"), "Est. Revenue" | actual computed values only, neutral sublabels |
| `index.html` | referenced `css/style.css`, `js/*.js` that don't exist → page rendered with no CSS/JS | paths fixed; all assets verified to load |

Chart 01 (size vs rating, bubble ∝ installs) was kept — its data and
interpretation were valid — with legend/margin fixes for mobile.

## Tests / checks performed

- `python fetch_dataset.py` — 3 parts combined (2,312,944 data rows;
  headerless parts verified and handled), 40,000-row sample built from 211
  strata with per-stratum provenance recorded.
- `python clean.py` (40k sample and full 2.31M-row file) — reports inspected;
  full run: 2,312,944 → 2,312,222 rows in 80 s / 3.0 GB peak RSS, `apps.json`
  correctly capped at 60,000 rows.
- `python train_models.py` — all 15 candidate pipelines (5×M1, 5×M2×2
  versions) trained; the fixed seed makes runs reproducible.
- `node --check` on every JS file.
- `node tests/smoke_frontend.js` — assertions against the **real**
  `apps.json` + `metrics.json`: charts render from real values (counts
  cross-checked against the cleaned data), data-gap states correct when
  columns are absent, ML sections render, form submission performs a real
  inference call, and — with the API switched off — the dataset still loads
  from `data/apps_bundle.js` (40,000 rows, never the 11-row fallback) while
  both prediction forms fall through to the in-browser engine.
- `node tests/browser_inference.test.js` — 180 parity cases per model: the JS
  engine must reproduce the sklearn pipeline's own output (max abs error
  ~5e-11), plus unknown-category encoding, median imputation and the
  `apps_bundle.js` round-trip.
- Live API checks (`fastapi.testclient`): `/api/health`, `/api/metrics`,
  valid + invalid (422) + unknown-category predictions, missing-metadata
  → 503 (not 500), dot-directory protection (404).
- Earlier bug-fix pass (commit `5c0a5c7`): dead assumption branch, stale tier
  names, dead anchor, unused variable, missing-price semantics, `is_sample`
  heuristic, month-name date parsing (UTC), tier stub in the tests.

## Known limitations / unresolved

- **Sample vs full scale:** the committed models are trained on 40,000 rows
  (1.7% of the source). Numbers are representative, not full-scale; the
  dashboard and `/api/metrics` say so explicitly, and rare install tiers are
  deliberately over-sampled in it. Run `fetch_dataset.py` + the three commands
  for full-scale artifacts (expect ~80 s cleaning and noticeably longer
  training on the full 1.23M M1 rows).
- **M1's near-zero R²** is a genuine finding (see results), not a bug — the
  feature set cannot encode app quality.
- **M2-B is weak but honest:** 0.307 macro-F1 on 4 imbalanced bands is what
  category+size+price alone can achieve; plain accuracy is below the
  majority-class baseline and the page says so rather than quoting accuracy.
- **Browser payload cap:** for full-scale runs `apps.json` carries 60,000 of
  2.31M rows (rule 12). Chart shapes are stable, but exact KPI counts refer to
  the exported rows; `apps_cleaned.csv` and training use every row.
- Section 10 (sentiment) intentionally **not** implemented: it needs a review
  text dataset with verified label provenance; it will be added (TF-IDF +
  Logistic Regression) when that file is provided, with disclosure if any
  labels are automated (e.g. TextBlob-derived). No K-Means. (The primary
  dataset has no sentiment column, so the dashboard correctly omits the
  "Subjectivity" column from chart 02 rather than inventing one.)

## Viva explanation (verified work only)

> "The dashboard is a single cleaned dataset rendered end-to-end. The primary
> source is the MIT-licensed gauthamp10 Google-Playstore scrape — 2,312,944
> apps scraped in June 2021 — and the repository ships a deterministic
> 40,000-row stratified sample of it so the project runs out of the box;
> `fetch_dataset.py` rebuilds the full 666 MB CSV, which `clean.py` then
> auto-detects. On the full file, clean.py applied its documented rules in
> 80 seconds with a 3 GB peak (installs treated as band lower bounds, unrated
> apps' `0.0` ratings turned into missing — never fake zeros — 717 exact
> duplicates removed) and produced a report: 2,312,222 cleaned rows, 1,230,293
> with valid ratings for the regressor, 2,312,115 with valid installs for the
> classifier.
>
> Two scikit-learn model families are trained with a fixed seed: a rating
> regression and a four-class install-tier classifier (Under 10K, 10K-1M,
> 1M-100M, 100M-plus), each compared against a naive baseline and three other
> learners. The train/test split is done *before* any preprocessing and
> grouped by app name, so the same app can't leak across the split;
> imputation, one-hot encoding and scaling fit on training rows only; models
> are chosen on a validation slice — by macro-F1 for the classifier, precisely
> because the tiers are imbalanced — and the test set is used exactly once.
>
> The measured results are honest. The regressor reaches R² ≈ 0.05 with
> MAE ≈ 0.50, and in this dataset the median rating is 4.2, so predicting the
> mean already gives MAE 0.52: rating is driven by app quality, which these
> profile features can't capture, and I can explain why that's expected. The
> tier classifier reaches 0.83 macro-F1 *with* Reviews but 0.31 *without* —
> and in the without-Reviews model accuracy (67%) is actually below the
> do-nothing baseline (72%), because 73% of apps sit in the lowest band; that
> is exactly why I select and report macro-F1 instead of hiding behind
> accuracy, and why the without-Reviews model is the primary one.
>
> Inference runs through a FastAPI endpoint that loads the saved pipelines
> once — no retraining per request — and if the models are absent the UI shows
> an unavailable state instead of demo numbers. Every chart uses data that
> exists in the dataset: the fabricated map, growth curves and revenue
> estimates were replaced by a correlation matrix, a last-updated-month
> snapshot, a rating histogram and a pricing mix, each labelled with how it
> was aggregated and what it does not claim."
