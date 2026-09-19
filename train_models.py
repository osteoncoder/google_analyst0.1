#!/usr/bin/env python3
"""
train_models.py — reproducible scikit-learn training for the two core models.

M1  Rating regression
    target : Rating (1-5)
    inputs : category, size_mb, price, price_is_positive, reviews_log
             (Rating itself and Installs are NOT inputs)
M2  Install-tier classification (4 bands, defined in clean.py)
    version A : base inputs + reviews_log
    version B : base inputs WITHOUT reviews_log   <-- primary dashboard model

Protocol (identical for every candidate and both M2 versions):
  1. 80/20 train/test split BEFORE any preprocessing, via GroupShuffleSplit
     on the app name, so the same app can never appear on both sides of the
     split (duplicate/leakage guard).
  2. The training part is further split 75/25 into train/validation (same
     grouping). Preprocessing (imputation, one-hot encoding, scaling) is fit
     on the TRAIN slice only, inside a Pipeline.
  3. Candidates are compared and selected on VALIDATION metrics only.
  4. The selected candidate is refit on train+validation and evaluated on the
     held-out test set exactly once.
  5. Fixed random seed (SEED) everywhere. No tuning against the test set.

Run:
    python train_models.py            # reads data/apps_cleaned.csv (from clean.py)
    python train_models.py --data path/to/apps_cleaned.csv

Candidate selection also enforces an ARTIFACT-SIZE BUDGET (--max-artifact-mb,
default 10 MB). A model that cannot be shipped — to the FastAPI backend, to
ml/artifacts/browser/models.js, or to a GitHub Pages checkout — is not a
candidate, however good its validation score: it is fitted, measured, recorded
with its score and its size, and then excluded. Unbounded Random Forest is the
canonical example (443 MB on the 40k sample); Gradient Boosting gets ~99% of
its lift in ~1 MB and wins on merit once the budget is applied.

Run:
    python train_models.py            # reads data/apps_cleaned.csv (from clean.py)
    python train_models.py --data path/to/apps_cleaned.csv
    python train_models.py --max-artifact-mb 25
    python train_models.py --fast     # cheaper: early stopping + capped tree depth

Artifacts:
    ml/artifacts/m1_rating/{pipeline.joblib, meta.json}
    ml/artifacts/m2_tier_with_reviews/{pipeline.joblib, meta.json}
    ml/artifacts/m2_tier_without_reviews/{pipeline.joblib, meta.json}
    ml/artifacts/metrics.json   (combined; consumed by dashboard section 09)
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

ROOT = Path(__file__).resolve().parent
ART = ROOT / "ml" / "artifacts"
SEED = 42

sys.path.insert(0, str(ROOT))
from clean import DEFAULT_TIER_BOUNDS, install_tier  # noqa: E402

# Numeric inputs that exist in every supported dataset.
CORE_NUM = ["size_mb", "price", "price_is_positive"]
# Engineered by clean.py rule 13. Any of them that is entirely missing (an
# older export without those raw columns, or the 11-row mechanical sample) is
# dropped from the feature set instead of being fed to the median imputer as an
# all-NaN column, which would propagate NaN into every prediction.
DERIVED_NUM = [
    "app_age_days", "days_since_update", "developer_app_count",
    "min_android", "ad_supported", "in_app_purchases", "editors_choice",
]
# Categorical inputs, one-hot encoded in this order.
CAT_COLS = ["category", "content_rating"]
BASE_NUM = CORE_NUM + DERIVED_NUM

BYTES_PER_MB = 1024.0 * 1024.0
# Measured on this project's artifacts: a serialized sklearn tree costs
# ~93-97 bytes per node (m1's GB regressor 92 B/node, m2-B's decision tree
# 96.5 B/node). Used only to decide whether an exact measurement is affordable.
TREE_NODE_BYTES = 100.0


class MeanBaseline(RegressorMixin, BaseEstimator):
    """Naive baseline: always predict the training-set mean rating."""

    def fit(self, X, y):
        self.mean_ = float(np.asarray(y, dtype=float).mean())
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def make_preprocessor(num_cols: list[str], cat_cols: list[str] = None) -> ColumnTransformer:
    """Imputation + encoding + scaling fit on training rows only (inside Pipeline)."""
    if cat_cols is None:
        cat_cols = ["category"]
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )


def make_pipeline(est, num_cols: list[str], cat_cols: list[str] = None) -> Pipeline:
    return Pipeline([("prep", make_preprocessor(num_cols, cat_cols)), ("model", est)])


# ---------------------------------------------------------------------------
# artifact-size budget
# ---------------------------------------------------------------------------
def _tree_nodes(est) -> int:
    """Total node count across every tree a fitted estimator holds."""
    total = 0
    estimators = getattr(est, "estimators_", None)
    if estimators is not None:
        for t in np.asarray(estimators, dtype=object).ravel():
            tree = getattr(t, "tree_", None)
            if tree is not None:
                total += int(tree.node_count)
    elif getattr(est, "tree_", None) is not None:
        total += int(est.tree_.node_count)
    return total


def estimate_artifact_mb(pipe: Pipeline) -> float:
    """Cheap size estimate — no serialization, so it is safe for huge forests."""
    est = pipe.named_steps["model"]
    nbytes = _tree_nodes(est) * TREE_NODE_BYTES
    coef = getattr(est, "coef_", None)
    if coef is not None:
        nbytes += np.asarray(coef, dtype=float).nbytes
    return nbytes / BYTES_PER_MB


def artifact_size_mb(pipe: Pipeline, budget_mb: float) -> tuple[float, str]:
    """Exact size when measuring it is affordable, otherwise the estimate.

    A full pickle of an unbounded forest can be hundreds of megabytes, so the
    (cheap) node-count estimate decides first: only models already close to the
    budget get measured for real.
    """
    est_mb = estimate_artifact_mb(pipe)
    if est_mb <= budget_mb * 4:
        try:
            return len(pickle.dumps(pipe)) / BYTES_PER_MB, "measured"
        except Exception:                                    # pragma: no cover
            pass
    return est_mb, "estimated from tree node count"


def split_groups(n: int, groups: np.ndarray, test_size: float, seed: int):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    # NOTE: split signature is (X, y=None, groups=None) — pass groups by keyword.
    tr, te = next(gss.split(np.zeros((n, 1)), None, groups))
    return tr, te


def usable_numeric(df: pd.DataFrame, cols: list[str]) -> tuple[list[str], list[str]]:
    """Split `cols` into (columns with at least one real value, dropped ones)."""
    keep, dropped = [], []
    for c in cols:
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any():
            keep.append(c)
        else:
            dropped.append(c)
    return keep, dropped


def build_matrix(df: pd.DataFrame, include_reviews: bool,
                 num_cols: list[str], cat_cols: list[str] = None) -> pd.DataFrame:
    """Feature matrix. NOTE: Installs and tier labels never appear here."""
    if cat_cols is None:
        cat_cols = ["category"]
    out = pd.DataFrame(index=df.index)
    out["app"] = df["app"].astype(str).str.strip()
    for c in cat_cols:
        out[c] = (df[c].fillna("__missing__").astype(str) if c in df.columns
                  else "__missing__")
    for c in num_cols:
        out[c] = (pd.to_numeric(df[c], errors="coerce").astype(float) if c in df.columns
                  else np.nan)
    # Derived, not a raw column — keep it even though apps_cleaned.csv has no
    # such field (a NaN price is "not paid", i.e. 0, never 1).
    if "price_is_positive" in num_cols:
        out["price_is_positive"] = (out["price"] > 0).astype(float)
    if include_reviews:
        out["reviews_log"] = np.log1p(df["reviews"].astype(float).clip(lower=0))
    return out


def scores_regression(pred, y) -> dict:
    y = np.asarray(y, dtype=float)
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
    }


def scores_classification(pred, y, labels: list[str]) -> dict:
    pred, y = np.asarray(pred), np.asarray(y)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, labels=labels, average="weighted", zero_division=0)),
    }


def full_classification(pred, y, labels: list[str]) -> dict:
    out = scores_classification(pred, y, labels)
    p, r, f, sup = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    out["per_class"] = {
        name: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(sup[i]),
        }
        for i, name in enumerate(labels)
    }
    out["confusion"] = {
        "matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "labels": list(labels),
    }
    return out


def extract_rf_importances(pipe: Pipeline) -> list[list] | None:
    """Feature importances of the Random Forest candidate (if it trained)."""
    est = pipe.named_steps["model"]
    if not hasattr(est, "feature_importances_"):
        return None
    names = list(pipe.named_steps["prep"].get_feature_names_out())
    imps = est.feature_importances_
    order = np.argsort(imps)[::-1]
    return [[str(names[i]), float(imps[i])] for i in order if imps[i] > 0]


def run_task(
    task_dir: Path,
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    num_cols: list[str],
    is_regression: bool,
    candidates: dict,
    extra_meta: dict,
    cat_cols: list[str] = None,
    budget_mb: float = 10.0,
) -> dict:
    """One complete train/validate/select/evaluate run for a task/version."""
    if cat_cols is None:
        cat_cols = ["category"]
    n = len(X)
    tr_idx, te_idx = split_groups(n, groups, 0.2, SEED)
    tr_full = X.iloc[tr_idx]
    tr2, val2 = split_groups(len(tr_full), groups[tr_idx], 0.25, SEED + 1)
    tr_idx2, val_idx2 = tr_full.index[tr2], tr_full.index[val2]

    Xtr, ytr = X.loc[tr_idx2], y.loc[tr_idx2]
    Xval, yval = X.loc[val_idx2], y.loc[val_idx2]
    # te_idx is POSITIONAL (from GroupShuffleSplit) → .iloc, not .loc
    Xte, yte = X.iloc[te_idx], y.iloc[te_idx]
    n_train, n_val, n_test = len(Xtr), len(Xval), len(Xte)
    # Use the canonical class order when supplied (e.g. tier order), else first-seen.
    labels = (extra_meta.get("labels") or list(pd.unique(y))) if not is_regression else None

    candidate_results: dict[str, dict] = {}
    rf_pipe = None
    for name, factory in candidates.items():
        try:
            pipe = make_pipeline(factory(), num_cols, cat_cols)
            pipe.fit(Xtr, ytr)
            val_s = scores_regression(pipe.predict(Xval), yval) if is_regression \
                else scores_classification(pipe.predict(Xval), yval, labels)
            test_s = scores_regression(pipe.predict(Xte), yte) if is_regression \
                else scores_classification(pipe.predict(Xte), yte, labels)
            rec: dict = {"val": val_s, "test": test_s}
            # Shippability gate: a model nobody can deploy is not a candidate.
            size_mb, basis = artifact_size_mb(pipe, budget_mb)
            rec["artifact_mb"] = round(size_mb, 3)
            rec["artifact_size_basis"] = basis
            if size_mb > budget_mb:
                rec["rejected"] = {
                    "reason": f"artifact {size_mb:,.0f} MB exceeds the "
                              f"{budget_mb:g} MB shipping budget",
                    "note": ("excluded from selection: it cannot be served by the API, "
                             "exported to the browser bundle, or committed to git at a "
                             "sensible size. Its validation score is recorded anyway."),
                }
            candidate_results[name] = rec
            if name.startswith("Random Forest"):
                rf_pipe = pipe
        except Exception as e:  # e.g. class absent in tiny train slice
            candidate_results[name] = {"error": f"{type(e).__name__}: {e}"}

    # Select on VALIDATION only, among candidates that can actually be shipped.
    sel_metric = "r2" if is_regression else "macro_f1"
    ranked = [
        (name, r["val"][sel_metric])
        for name, r in candidate_results.items()
        if "val" in r and not r.get("rejected")
    ]
    if not ranked:
        detail = json.dumps(candidate_results, indent=2)
        if any("val" in r for r in candidate_results.values()):
            sys.exit(f"{task_dir.name}: every candidate that trained was rejected by the "
                     f"{budget_mb:g} MB artifact budget — raise --max-artifact-mb. {detail}")
        sys.exit(f"{task_dir.name}: every candidate failed to train — see candidate_results: {detail}")
    ranked.sort(key=lambda t: (-t[1], not t[0].startswith("Random Forest")))
    selected = ranked[0][0]

    # Refit selected on train+val, evaluate on test exactly once
    final = make_pipeline(candidates[selected](), num_cols, cat_cols)
    final.fit(pd.concat([Xtr, Xval]), pd.concat([ytr, yval]))
    if is_regression:
        test_full = scores_regression(final.predict(Xte), yte)
    else:
        test_full = full_classification(final.predict(Xte), yte, labels)

    rf_imp = extract_rf_importances(rf_pipe) if rf_pipe is not None else None

    # Prediction-time fallback for a categorical field the caller leaves out:
    # the training-set MODE, so an omitted field means "assume the most common
    # value" rather than "an unknown category". Exported to the browser bundle
    # and used by app.py, so both engines agree.
    cat_defaults = {}
    for c in cat_cols:
        if c in Xtr.columns:
            vc = Xtr[c].astype(str).value_counts()
            cat_defaults[c] = str(vc.index[0]) if len(vc) else "__missing__"

    meta = {
        "task": "regression" if is_regression else "classification",
        "model": selected,
        "selection_metric": f"validation {sel_metric}",
        "features": list(num_cols) + list(cat_cols),
        "numeric_features": list(num_cols),
        "categorical_features": list(cat_cols),
        "categorical_defaults": cat_defaults,
        "n_total": n,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "candidates": candidate_results,
        "test_metrics": test_full,
        "rf_feature_importances": rf_imp,
        "importances_from": "Random Forest candidate" if rf_imp else None,
        "artifact_budget_mb": budget_mb,
        **extra_meta,
    }

    task_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, task_dir / "pipeline.joblib")
    # Record what the artifact really costs on disk — the number the budget is
    # about, and the number the README quotes.
    meta["artifact_size_mb"] = round(
        (task_dir / "pipeline.joblib").stat().st_size / BYTES_PER_MB, 3)
    (task_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("Run:")[0])
    ap.add_argument("--data", default=str(ROOT / "data" / "apps_cleaned.csv"))
    ap.add_argument("--max-artifact-mb", type=float, default=10.0,
                    help="reject any candidate whose serialized pipeline is larger "
                         "than this (default 10 MB — it has to be committed, served "
                         "and exported to the browser bundle)")
    ap.add_argument("--fast", action="store_true",
                    help="cheaper training: gradient boosting early stopping and a "
                         "depth-capped decision tree. Changes the fitted model "
                         "(fewer/deeper-limited trees), so numbers differ from the "
                         "default run — it exists for the optional full-scale run.")
    args = ap.parse_args()
    budget_mb = args.max_artifact_mb

    path = Path(args.data)
    if not path.exists():
        sys.exit(f"{path} not found — run `python clean.py` first.")
    df = pd.read_csv(path)

    report_path = path.parent / "cleaning_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    sampling = report.get("sampling") or {}
    is_sample = bool(report.get("is_sample", False))
    # Honest provenance: a sample is a sample, whatever its size. When the
    # source is the committed stratified sample, say what it is a sample OF.
    sample_note = None
    if is_sample or sampling:
        if sampling.get("full_rows"):
            sample_note = (f"trained on a {report.get('cleaned_rows', len(df)):,}-row stratified sample "
                           f"of the {sampling['full_rows']:,}-row Google-Playstore dataset "
                           f"(full-scale proportions differ; see data/playstore_sample.meta.json)")
        else:
            sample_note = f"trained on a {report.get('cleaned_rows', len(df)):,}-row sample dataset"
    dataset_info = {
        "source": report.get("source_file", str(path)),
        "is_sample": is_sample,
        "rows_cleaned": report.get("cleaned_rows", len(df)),
        "md5": hashlib_md5(path),
        "full_dataset_rows": sampling.get("full_rows"),
        "sample_note": sample_note,
    }
    # Tier bands come from clean.py (via the report) — single source of truth.
    raw_bounds = report.get("tier_bounds") or [list(b) for b in DEFAULT_TIER_BOUNDS]
    tier_bounds = [(b[0], float("inf") if b[1] is None else b[1], b[2]) for b in raw_bounds]
    TIER_NAMES = [b[2] for b in tier_bounds]
    n = len(df)
    small = n < 100          # mechanical-test sample: numbers verify the pipeline only

    # Which of clean.py rule 13's engineered columns carry real information in
    # THIS dataset (an older export or the 11-row sample has none of them).
    derived_used, derived_dropped = usable_numeric(df, DERIVED_NUM)
    num_extra = CORE_NUM + derived_used
    if derived_dropped:
        print(f"note         : no usable values for {derived_dropped} — "
              f"excluded from the feature set (dataset lacks those raw columns)")
    feature_note = {
        "engineered_features": derived_used,
        "engineered_features_dropped_all_nan": derived_dropped,
        "derived_from": ("clean.py rule 13 — Released / Scraped Time / Developer Id / "
                         "Minimum Android / Ad Supported / In App Purchases / "
                         "Editors Choice / Content Rating"),
    }

    # --fast: cheaper fits for the optional full-scale run. Both change the
    # fitted model, so they are opt-in and the console says so.
    gb_kwargs = {"n_estimators": 200, "learning_rate": 0.1, "random_state": SEED}
    dt_kwargs = {"random_state": SEED}
    if args.fast:
        gb_kwargs.update(n_iter_no_change=10, validation_fraction=0.1, tol=1e-4)
        dt_kwargs.update(max_depth=12, min_samples_leaf=5)

    # ---------------- M1: rating regression ----------------
    m1_df = df[df["rating"].notna()]
    X1 = build_matrix(m1_df, include_reviews=True, num_cols=num_extra + ["reviews_log"],
                      cat_cols=CAT_COLS)
    y1 = m1_df["rating"].astype(float)
    m1_meta = run_task(
        ART / "m1_rating", X1, y1, X1["app"].values, num_extra + ["reviews_log"],
        is_regression=True, cat_cols=CAT_COLS, budget_mb=budget_mb,
        candidates={
            "Mean baseline (train mean)": lambda: MeanBaseline(),
            "Linear Regression": lambda: LinearRegression(),
            "Decision Tree": lambda: DecisionTreeRegressor(**dt_kwargs),
            # n_jobs=-1 is bit-identical (fixed random_state) and ~1.9x faster.
            "Random Forest": lambda: RandomForestRegressor(
                n_estimators=300, random_state=SEED, n_jobs=-1),
            "Gradient Boosting": lambda: GradientBoostingRegressor(**gb_kwargs),
        },
        extra_meta={
            "dataset": dataset_info,
            "target": "Rating",
            "inputs_note": ("category, content_rating, size_mb, price, price_is_positive, "
                            "log1p(reviews) + the engineered app-profile features. "
                            "Rating (target) and Installs are excluded from the inputs."),
            **feature_note,
        },
    )

    # ---------------- M2: install-tier classification, two versions ----------------
    m2_df = df[df["installs"].notna()]
    y2 = m2_df["installs"].map(lambda v: install_tier(v, tier_bounds))
    m2_df = m2_df[y2.notna()]
    y2 = y2[m2_df.index]

    X2_base = build_matrix(m2_df, include_reviews=False, num_cols=num_extra, cat_cols=CAT_COLS)
    X2_rev = build_matrix(m2_df, include_reviews=True, num_cols=num_extra + ["reviews_log"],
                          cat_cols=CAT_COLS)
    groups2 = m2_df["app"].values
    class_counts = {name: int((y2 == name).sum()) for name in TIER_NAMES}

    def m2_candidates():
        return {
            "Dummy baseline (class prior)": lambda: DummyClassifier(strategy="prior", random_state=SEED),
            "Logistic Regression": lambda: LogisticRegression(max_iter=2000, random_state=SEED),
            "Decision Tree": lambda: DecisionTreeClassifier(**dt_kwargs),
            "Random Forest": lambda: RandomForestClassifier(
                n_estimators=300, random_state=SEED, n_jobs=-1),
            "Gradient Boosting": lambda: GradientBoostingClassifier(**gb_kwargs),
        }

    m2_common = {
        "target": "install tier (4 bands)",
        "tier_bounds": [[lo, (hi if np.isfinite(hi) else None), name] for lo, hi, name in tier_bounds],
        "labels": TIER_NAMES,
        "class_counts_full_data": class_counts,
        **feature_note,
    }
    m2a_meta = run_task(
        ART / "m2_tier_with_reviews", X2_rev, y2, groups2, num_extra + ["reviews_log"],
        is_regression=False, candidates=m2_candidates(),
        cat_cols=CAT_COLS, budget_mb=budget_mb,
        extra_meta={**m2_common, "dataset": dataset_info, "version": "with_reviews",
                    "inputs_note": ("Base inputs + the engineered app-profile features "
                                    "+ log1p(reviews).")},
    )
    m2b_meta = run_task(
        ART / "m2_tier_without_reviews", X2_base, y2, groups2, num_extra,
        is_regression=False, candidates=m2_candidates(),
        cat_cols=CAT_COLS, budget_mb=budget_mb,
        extra_meta={**m2_common, "dataset": dataset_info, "version": "without_reviews",
                    "inputs_note": ("Base inputs + the engineered app-profile features — "
                                    "no Reviews (primary dashboard model).")},
    )

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "dataset": dataset_info,
        "small_dataset_warning": small,
        "trained_on_sample": is_sample or bool(sampling),
        "protocol": (
            "80/20 GroupShuffleSplit on app name (leakage guard), fit BEFORE preprocessing; "
            "75/25 train/val for model selection; selected model refit on train+val; "
            "test set used exactly once; no test-set tuning."
        ),
        "selection": (
            f"best validation metric among candidates whose serialized pipeline fits the "
            f"{budget_mb:g} MB shipping budget; oversized candidates are reported with "
            f"their scores and the reason they were excluded."
        ),
        "artifact_budget_mb": budget_mb,
        "fast_mode": bool(args.fast),
        "models": {
            "m1_rating": m1_meta,
            "m2_tier": {
                "with_reviews": m2a_meta,
                "without_reviews": m2b_meta,
            },
        },
    }
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    # Also publish the fitted pipelines as a <script>-loadable bundle so the
    # prediction forms keep working with no backend (static host / file://
    # page), evaluated in the browser by ml_inference.js.
    try:
        from browser_export import write_models_bundle
        write_models_bundle()
    except BaseException as exc:            # SystemExit too — never abort training
        print(f"\nWARNING: browser model bundle not exported ({exc}).\n"
              f"         Predictions will need the backend (python app.py).")

    # ---------------- console summary ----------------
    print("=== Training summary ===")
    print(f"dataset      : {dataset_info['source']} (rows={n})")
    print(f"features     : {len(num_extra)} numeric + {len(CAT_COLS)} categorical"
          f"{f' (+{len(derived_used)} engineered)' if derived_used else ''}"
          f" · artifact budget {budget_mb:g} MB"
          f"{' · FAST mode (early stopping / capped depth)' if args.fast else ''}")
    if sample_note:
        print(f"note         : {sample_note}")

    def candidate_line(name, r, regression):
        """One candidate: score, size, and whether the shipping budget rejected it."""
        if "val" not in r:
            print(f"  {name:<28} FAILED: {r['error']}")
            return
        tag = (f"val R2={r['val']['r2']:+.3f}  test MAE={r['test']['mae']:.3f} "
               f"RMSE={r['test']['rmse']:.3f} R2={r['test']['r2']:+.3f}") if regression else (
               f"val acc={r['val']['accuracy']:.3f} macroF1={r['val']['macro_f1']:.3f}"
               f"  |  test acc={r['test']['accuracy']:.3f} macroF1={r['test']['macro_f1']:.3f}")
        if r.get("rejected"):
            print(f"  {name:<28} REJECTED ({r['rejected']['reason']})")
            print(f"  {'':<28} {tag}")
        else:
            print(f"  {name:<28} {tag}  [{r.get('artifact_mb', 0):.1f} MB]")

    print(f"\nM1 rating regression  (n={m1_meta['n_total']}, test={m1_meta['n_test']})")
    for name, r in m1_meta["candidates"].items():
        candidate_line(name, r, regression=True)
    print(f"  selected: {m1_meta['model']} (by validation R2); final test: "
          f"MAE={m1_meta['test_metrics']['mae']:.3f} RMSE={m1_meta['test_metrics']['rmse']:.3f} "
          f"R2={m1_meta['test_metrics']['r2']:+.3f}")

    for ver in ("with_reviews", "without_reviews"):
        m = metrics["models"]["m2_tier"][ver]
        print(f"\nM2 install tier — {ver}  (n={m['n_total']}, test={m['n_test']})")
        for name, r in m["candidates"].items():
            candidate_line(name, r, regression=False)
        tm = m["test_metrics"]
        print(f"  selected: {m['model']} (by validation macro-F1); final test: "
              f"acc={tm['accuracy']:.3f} macroF1={tm['macro_f1']:.3f} weightedF1={tm['weighted_f1']:.3f}")
    print("\nArtifacts written to ml/artifacts/ (pipeline.joblib + meta.json per model, metrics.json)")
    if small:
        print("\nNOTE: fewer than 100 rows — these numbers verify the pipeline mechanically "
              "only and must NOT be quoted as project results. Re-run with the full dataset.")
    elif sample_note:
        print("\nNOTE: sample training — numbers are representative but not full-scale. "
              "`python fetch_dataset.py --sample 40000` + re-running clean.py and train_models.py "
              "reproduces them on the full 2.31M-row dataset.")


def hashlib_md5(p: Path) -> str:
    import hashlib

    return hashlib.md5(p.read_bytes()).hexdigest()[:12]


if __name__ == "__main__":
    main()
