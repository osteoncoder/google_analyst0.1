# APEX — Play Store Intelligence Dashboard

Aurora-themed Google Play Store analytics dashboard with two machine-learning
models (rating regression + install-tier classification).

**Stack (unchanged from the original project):** plain HTML/CSS/JS + Plotly.js
from CDN for the charts; a small FastAPI backend added **only** for real ML
inference (the models are saved scikit-learn pipelines — nothing retrains per
request). No build step, no npm.

```
.
├── index.html               # page structure, sections 01-09
├── style.css                # aurora/glass theme + ML section styles
├── data.js                  # dataset loader + JS mirror of the cleaning rules
├── charts.js                # KPI strip + charts 01-06 (all data-driven)
├── ml_dashboard.js          # sections 07-09 + bootstrap + unavailable states
├── clean.py                 # single-source-of-truth cleaning pipeline (CLI)
├── train_models.py          # reproducible scikit-learn training (M1, M2 A/B)
├── app.py                   # FastAPI: static dashboard + /api inference
├── requirements.txt         # Python dependencies
├── data/
│   ├── sample_apps.csv      # the project's original 11-row sample (labelled)
│   ├── play_store.csv       # <-- put your REAL dataset here
│   ├── apps_cleaned.csv     # output of clean.py
│   ├── apps.json            # output of clean.py (feeds the dashboard)
│   └── cleaning_report.json # provenance: raw/cleaned/removed counts
├── ml/artifacts/            # output of train_models.py (pipelines + metrics)
└── tests/smoke_frontend.js  # headless Node check of the whole frontend
```

## Run it (full version with ML)

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt   # Windows: .venv\Scripts\pip ...

.venv/bin/python clean.py          # 1. clean raw data  -> data/apps_cleaned.csv, data/apps.json
.venv/bin/python train_models.py   # 2. train models    -> ml/artifacts/
.venv/bin/python app.py            # 3. serve           -> http://localhost:8000
```

Open http://localhost:8000. The KPI strip, charts 01-06 and the ML sections
all read from the same cleaned dataset; sections 07-09 call the API.

**Static-only mode (no ML):** open `index.html` directly (double-click or VS
Code Live Server). Charts 01-06 still work (they fall back to the embedded
11-row sample if `data/apps.json` can't be fetched); sections 07-09 show an
explicit "model service not available" panel instead of faking results.

## Data pipeline (`clean.py`) — documented rules

1. **Identity**: rows missing `App` or `Category` are dropped (counted).
2. **Installs**: `"1,000,000+"` / `1000000` → integer **lower bound** of the
   reported download band. Play reports bands (10 … 1B+); the number is the
   band floor, **not** an exact download count. All UI labels say so.
3. **Reviews**: integer; blank → 0 (not reported).
4. **Rating**: kept only if 1 ≤ r ≤ 5.
5. **Size**: `86M` → 86 MB, `72K` → 72/1024 MB, `Varies with device` → missing
   (a real state, imputed later by the model pipeline, never by hand).
6. **Price**: `""`/`0` → 0.0. **Legitimate zeros are preserved**, never
   imputed to nonzero values. Listed price is a price tag, **not revenue** —
   no revenue figure exists anywhere in the project.
7. **Last Updated**: `Jan 15, 2024` / `2024-01-15` / …; unparseable → missing.
8. **Category**: trim + title case + one documented typo fix
   (`commication` → `Communication`, present in the sample data). No other
   names are altered or merged.
9. **Duplicates**: exact duplicates on
   (app, category, rating, reviews, installs, size_mb, price) → keep first.
   Same-name rows with different metadata are kept (different listings) and
   counted.
10. **Eligibility**: M1 rows = valid rating; M2 rows = valid installs bound.

`cleaning_report.json` records raw rows, cleaned rows, everything removed,
per-field missingness, and rows used per model. Cleaning and feature
preparation are the same code path for training (`train_models.py`) and for
the dashboard (`data/apps.json`), and prediction uses the **saved**
preprocessor inside the pipeline joblib file — inference can never drift from
training.

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
  No test-set tuning.

### M1 — Rating regression (`m1_rating/`)

- Target: `Rating`. Inputs: `category, size_mb, price, price_is_positive,
  log1p(reviews)`. **Rating itself and Installs are excluded from inputs.**
- Candidates: mean-prediction baseline, Linear Regression, Decision Tree,
  **Random Forest**, Gradient Boosting.
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
  Tree, **Random Forest**, Gradient Boosting.
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
- Unknown category → `handle_unknown='ignore'` one-hot (encoded as "not seen
  in training"). The API returns its assumptions with every prediction.

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

## Dataset (provenance)

- **Primary (bundled):** `data/play_store.csv` — the canonical *Google Play
  Store Apps* export (10,841 rows, 13 columns) originally by Friedrich
  Cantmorgen on Kaggle (`lava18/google-play-store-apps`), licensed **CC-BY 3.0**
  (attribution required — kept here), obtained via a public GitHub mirror
  (`subhasushi/Google_Playstore_Apps`).
- **Fallback (bundled):** `data/sample_apps.csv` — the project's original
  11-row sample, used automatically only if `play_store.csv` is absent.
- The pipeline is dataset-agnostic: any CSV/XLSX with the same columns
  (aliases handled) works via `python clean.py --raw <path>`.

### Measured cleaning report (full dataset)

| Step | Count |
|---|---|
| Raw rows | 10,841 |
| Corrupted shifted rows dropped (numeric category + out-of-range rating; "Life Made WI-Fi Touchscreen Photo Frame") | 1 |
| Exact duplicates removed | 485 |
| Same-name variants kept (different listings) | 1,217 |
| **Cleaned rows** | **10,355** |
| Rows with valid rating → M1 | 8,890 |
| Rows with valid installs → M2 | 10,355 |
| Missing size ("Varies with device" etc.) | 1,526 |
| Price: free / paid / unknown | 9,590 / 765 / 0 |

### Measured evaluation results (full dataset, seed 42, reproducible)

**M1 — Rating regression** (n=8,890; test n=1,798; selected by validation R²):

| Model | MAE (test) | RMSE (test) | R² (test) |
|---|---|---|---|
| Mean baseline | 0.382 | 0.538 | −0.000 |
| Linear Regression | 0.365 | 0.524 | +0.053 |
| Decision Tree | 0.478 | 0.719 | −0.785 |
| Random Forest | 0.375 | 0.548 | −0.036 |
| **Gradient Boosting (selected)** | **0.358** | **0.527** | **+0.043** |

Honest reading: rating is barely predictable from category/size/price/
review volume (R² ≈ 0.04) — it is driven by app *quality*, which none of
these features capture. This is a defensible negative result, not a failure.

**M2 — Install-tier classification** (n=10,355; test n=2,066;
class counts: Under 10K 3,150 · 10K-1M 3,153 · 1M-100M 3,573 · 100M+ 479):

| Version | Model (selected) | Accuracy (test) | Macro-F1 (test) | Weighted-F1 (test) |
|---|---|---|---|---|
| A — with Reviews | Logistic Regression | 0.906 | 0.886 | 0.905 |
| **B — without Reviews (primary)** | **Gradient Boosting** | **0.510** | **0.383** | **0.488** |
| Baseline — class prior | Dummy | 0.345 | 0.128 | — |

The A→B drop (≈ 40 pts accuracy) is the measurable signature of Reviews
acting as a strong proxy for install volume — the exact reason B is the
primary dashboard model. Full per-class P/R/F1/support and the confusion
matrix are rendered in section 09/08 from `ml/artifacts/metrics.json`.

## Tests / checks performed

- `python clean.py` (full dataset) — report inspected: 10,841 raw → 1
  corrupted row dropped → 485 duplicates removed → 10,355 cleaned; all
  `Last Updated` values parsed (0 missing).
- `python train_models.py` — all 15 candidate pipelines (5×M1, 5×M2×2 versions)
  trained; **two consecutive runs produce byte-identical metrics** (seed 42).
- `node --check` on all three JS files.
- `node tests/smoke_frontend.js` — 19/19 assertions: charts render from real
  values (counts cross-checked against the cleaned data), data-gap states
  correct when columns are absent, ML sections render from `metrics.json`,
  form submission performs a real inference call.
- Live API checks: `/api/health`, `/api/metrics`, valid + invalid (422) +
  unknown-category predictions, dot-directory protection (404).

## Known limitations / unresolved

- M1's near-zero R² is a genuine finding (see results above), not a bug —
  the feature set cannot encode app quality.
- M2-B (without Reviews) is deliberately conservative: 51% accuracy / 0.38
  macro-F1 on 4 broad bands is what category+size+price alone can achieve.
- Section 10 (sentiment) intentionally **not** implemented: it needs a review
  text dataset with verified label provenance; it will be added (TF-IDF +
  Logistic Regression) when that file is provided, with disclosure if any
  labels are automated (e.g. TextBlob-derived). No K-Means.
- The bundled full dataset is the public CC-BY 3.0 export — if your own
  export differs slightly (columns, date format), the pipeline handles it
  (`--raw`), and the cleaning report will show exactly what changed.

## Viva explanation (verified work only)

> "The dashboard is a single cleaned dataset rendered end-to-end. On the full
> 10,841-row Play Store export, `clean.py` applies documented rules — install
> bands treated as lower bounds, zero prices preserved, one corrupted
> shifted row detected and dropped, 485 exact duplicates removed — and writes
> a report: 10,355 cleaned rows, 8,890 with valid ratings for the regressor,
> 10,355 with valid installs for the classifier.
>
> Two scikit-learn model families are trained with a fixed seed: a rating
> regression and a four-class install-tier classifier (Under 10K, 10K-1M,
> 1M-100M, 100M-plus), each compared against a naive baseline and three other
> learners. The train/test split is done *before* any preprocessing and
> grouped by app name, so the same app can't leak across the split;
> imputation, one-hot encoding and scaling fit on training rows only; models
> are chosen on a validation slice and the test set is used exactly once. Two
> consecutive training runs reproduce identical metrics.
>
> The measured results, shown in section 09, are honest: the regressor
> reaches R² ≈ 0.04 with MAE ≈ 0.36 — rating is driven by app quality, which
> these profile features can't capture, and I can explain why that's expected.
> The tier classifier hits 90.6% accuracy *with* Reviews but only 51.0%
> *without*; that 40-point gap is the measurable signature of Reviews being a
> strong proxy for installs, which is why the without-Reviews model is the
> primary dashboard model and the page states exactly that, including the
> caveat that removing Reviews doesn't make every other feature leakage-free.
>
> Inference runs through a FastAPI endpoint that loads the saved pipelines
> once — no retraining per request — and if the models are absent the UI
> shows an unavailable state instead of demo numbers. Every chart uses data
> that exists in the dataset: the fabricated map, growth curves and revenue
> estimates were replaced by a correlation matrix, a last-updated-month
> snapshot, a rating histogram and a pricing mix, each labelled with how it
> was aggregated and what it does not claim."
