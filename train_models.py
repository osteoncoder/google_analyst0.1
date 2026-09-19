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

Artifacts:
    ml/artifacts/m1_rating/{pipeline.joblib, meta.json}
    ml/artifacts/m2_tier_with_reviews/{pipeline.joblib, meta.json}
    ml/artifacts/m2_tier_without_reviews/{pipeline.joblib, meta.json}
    ml/artifacts/metrics.json   (combined; consumed by dashboard section 09)
"""
from __future__ import annotations

import argparse
import json
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

BASE_NUM = ["size_mb", "price", "price_is_positive"]


class MeanBaseline(RegressorMixin, BaseEstimator):
    """Naive baseline: always predict the training-set mean rating."""

    def fit(self, X, y):
        self.mean_ = float(np.asarray(y, dtype=float).mean())
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def make_preprocessor(num_cols: list[str]) -> ColumnTransformer:
    """Imputation + encoding + scaling fit on training rows only (inside Pipeline)."""
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
                ["category"],
            ),
        ],
        remainder="drop",
    )


def make_pipeline(est, num_cols: list[str]) -> Pipeline:
    return Pipeline([("prep", make_preprocessor(num_cols)), ("model", est)])


def split_groups(n: int, groups: np.ndarray, test_size: float, seed: int):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    # NOTE: split signature is (X, y=None, groups=None) — pass groups by keyword.
    tr, te = next(gss.split(np.zeros((n, 1)), None, groups))
    return tr, te


def build_matrix(df: pd.DataFrame, include_reviews: bool) -> pd.DataFrame:
    """Feature matrix. NOTE: Installs and tier labels never appear here."""
    out = pd.DataFrame(index=df.index)
    out["app"] = df["app"].astype(str).str.strip()
    out["category"] = df["category"].fillna("__missing__").astype(str)
    out["size_mb"] = df["size_mb"].astype(float)
    out["price"] = df["price"].astype(float)
    out["price_is_positive"] = (df["price"] > 0).astype(float)
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
) -> dict:
    """One complete train/validate/select/evaluate run for a task/version."""
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
            pipe = make_pipeline(factory(), num_cols)
            pipe.fit(Xtr, ytr)
            val_s = scores_regression(pipe.predict(Xval), yval) if is_regression \
                else scores_classification(pipe.predict(Xval), yval, labels)
            test_s = scores_regression(pipe.predict(Xte), yte) if is_regression \
                else scores_classification(pipe.predict(Xte), yte, labels)
            candidate_results[name] = {"val": val_s, "test": test_s}
            if name.startswith("Random Forest"):
                rf_pipe = pipe
        except Exception as e:  # e.g. class absent in tiny train slice
            candidate_results[name] = {"error": f"{type(e).__name__}: {e}"}

    # Select on VALIDATION only
    sel_metric = "r2" if is_regression else "macro_f1"
    ranked = [
        (name, r["val"][sel_metric])
        for name, r in candidate_results.items()
        if "val" in r
    ]
    if not ranked:
        sys.exit(f"{task_dir.name}: every candidate failed to train — see candidate_results: "
                 f"{json.dumps(candidate_results, indent=2)}")
    ranked.sort(key=lambda t: (-t[1], not t[0].startswith("Random Forest")))
    selected = ranked[0][0]

    # Refit selected on train+val, evaluate on test exactly once
    final = make_pipeline(candidates[selected](), num_cols)
    final.fit(pd.concat([Xtr, Xval]), pd.concat([ytr, yval]))
    if is_regression:
        test_full = scores_regression(final.predict(Xte), yte)
    else:
        test_full = full_classification(final.predict(Xte), yte, labels)

    rf_imp = extract_rf_importances(rf_pipe) if rf_pipe is not None else None

    meta = {
        "task": "regression" if is_regression else "classification",
        "model": selected,
        "selection_metric": f"validation {sel_metric}",
        "features": list(num_cols) + ["category"],
        "n_total": n,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "candidates": candidate_results,
        "test_metrics": test_full,
        "rf_feature_importances": rf_imp,
        "importances_from": "Random Forest candidate" if rf_imp else None,
        **extra_meta,
    }

    task_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, task_dir / "pipeline.joblib")
    (task_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("Run:")[0])
    ap.add_argument("--data", default=str(ROOT / "data" / "apps_cleaned.csv"))
    args = ap.parse_args()

    path = Path(args.data)
    if not path.exists():
        sys.exit(f"{path} not found — run `python clean.py` first.")
    df = pd.read_csv(path)

    report_path = path.parent / "cleaning_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    dataset_info = {
        "source": report.get("source_file", str(path)),
        "is_sample": bool(report.get("is_sample", False)),
        "rows_cleaned": report.get("cleaned_rows", len(df)),
        "md5": hashlib_md5(path),
    }
    # Tier bands come from clean.py (via the report) — single source of truth.
    raw_bounds = report.get("tier_bounds") or [list(b) for b in DEFAULT_TIER_BOUNDS]
    tier_bounds = [(b[0], float("inf") if b[1] is None else b[1], b[2]) for b in raw_bounds]
    TIER_NAMES = [b[2] for b in tier_bounds]
    n = len(df)
    small = n < 100

    # ---------------- M1: rating regression ----------------
    m1_df = df[df["rating"].notna()]
    X1 = build_matrix(m1_df, include_reviews=True)
    y1 = m1_df["rating"].astype(float)
    m1_meta = run_task(
        ART / "m1_rating", X1, y1, X1["app"].values, BASE_NUM + ["reviews_log"],
        is_regression=True,
        candidates={
            "Mean baseline (train mean)": lambda: MeanBaseline(),
            "Linear Regression": lambda: LinearRegression(),
            "Decision Tree": lambda: DecisionTreeRegressor(random_state=SEED),
            "Random Forest": lambda: RandomForestRegressor(n_estimators=300, random_state=SEED),
            "Gradient Boosting": lambda: GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.1, random_state=SEED),
        },
        extra_meta={
            "dataset": dataset_info,
            "target": "Rating",
            "inputs_note": ("category, size_mb, price, price_is_positive, log1p(reviews). "
                            "Rating (target) and Installs are excluded from the inputs."),
        },
    )

    # ---------------- M2: install-tier classification, two versions ----------------
    m2_df = df[df["installs"].notna()]
    y2 = m2_df["installs"].map(lambda v: install_tier(v, tier_bounds))
    m2_df = m2_df[y2.notna()]
    y2 = y2[m2_df.index]

    X2_base = build_matrix(m2_df, include_reviews=False)
    X2_rev = build_matrix(m2_df, include_reviews=True)
    groups2 = m2_df["app"].values
    class_counts = {name: int((y2 == name).sum()) for name in TIER_NAMES}

    def m2_candidates():
        return {
            "Dummy baseline (class prior)": lambda: DummyClassifier(strategy="prior", random_state=SEED),
            "Logistic Regression": lambda: LogisticRegression(max_iter=2000, random_state=SEED),
            "Decision Tree": lambda: DecisionTreeClassifier(random_state=SEED),
            "Random Forest": lambda: RandomForestClassifier(n_estimators=300, random_state=SEED),
            "Gradient Boosting": lambda: GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.1, random_state=SEED),
        }

    m2_common = {
        "target": "install tier (4 bands)",
        "tier_bounds": [[lo, (hi if np.isfinite(hi) else None), name] for lo, hi, name in tier_bounds],
        "labels": TIER_NAMES,
        "class_counts_full_data": class_counts,
    }
    m2a_meta = run_task(
        ART / "m2_tier_with_reviews", X2_rev, y2, groups2, BASE_NUM + ["reviews_log"],
        is_regression=False, candidates=m2_candidates(),
        extra_meta={**m2_common, "dataset": dataset_info, "version": "with_reviews",
                    "inputs_note": "Base inputs + log1p(reviews)."},
    )
    m2b_meta = run_task(
        ART / "m2_tier_without_reviews", X2_base, y2, groups2, BASE_NUM,
        is_regression=False, candidates=m2_candidates(),
        extra_meta={**m2_common, "dataset": dataset_info, "version": "without_reviews",
                    "inputs_note": "Base inputs only — no Reviews (primary dashboard model)."},
    )

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "dataset": dataset_info,
        "small_dataset_warning": small,
        "protocol": (
            "80/20 GroupShuffleSplit on app name (leakage guard), fit BEFORE preprocessing; "
            "75/25 train/val for model selection; selected model refit on train+val; "
            "test set used exactly once; no test-set tuning."
        ),
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

    # ---------------- console summary ----------------
    print("=== Training summary ===")
    print(f"dataset      : {dataset_info['source']} (rows={n}"
          f"{', SAMPLE - not for final results' if small else ''})")
    print(f"\nM1 rating regression  (n={m1_meta['n_total']}, test={m1_meta['n_test']})")
    for name, r in m1_meta["candidates"].items():
        if "val" in r:
            print(f"  {name:<28} val R2={r['val']['r2']:+.3f}  test MAE={r['test']['mae']:.3f} "
                  f"RMSE={r['test']['rmse']:.3f} R2={r['test']['r2']:+.3f}")
        else:
            print(f"  {name:<28} FAILED: {r['error']}")
    print(f"  selected: {m1_meta['model']} (by validation R2); final test: "
          f"MAE={m1_meta['test_metrics']['mae']:.3f} RMSE={m1_meta['test_metrics']['rmse']:.3f} "
          f"R2={m1_meta['test_metrics']['r2']:+.3f}")

    for ver in ("with_reviews", "without_reviews"):
        m = metrics["models"]["m2_tier"][ver]
        print(f"\nM2 install tier — {ver}  (n={m['n_total']}, test={m['n_test']})")
        for name, r in m["candidates"].items():
            if "val" in r:
                print(f"  {name:<28} val acc={r['val']['accuracy']:.3f} macroF1={r['val']['macro_f1']:.3f}"
                      f"  |  test acc={r['test']['accuracy']:.3f} macroF1={r['test']['macro_f1']:.3f}")
            else:
                print(f"  {name:<28} FAILED: {r['error']}")
        tm = m["test_metrics"]
        print(f"  selected: {m['model']} (by validation macro-F1); final test: "
              f"acc={tm['accuracy']:.3f} macroF1={tm['macro_f1']:.3f} weightedF1={tm['weighted_f1']:.3f}")
    print("\nArtifacts written to ml/artifacts/ (pipeline.joblib + meta.json per model, metrics.json)")
    if small:
        print("\nNOTE: fewer than 100 rows — these numbers verify the pipeline mechanically "
              "only and must NOT be quoted as project results. Re-run with the full dataset.")


def hashlib_md5(p: Path) -> str:
    import hashlib

    return hashlib.md5(p.read_bytes()).hexdigest()[:12]


if __name__ == "__main__":
    main()
