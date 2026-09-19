#!/usr/bin/env python3
"""
clean.py — documented, reusable cleaning pipeline for the APEX Play Store dashboard.

Dataset-agnostic: it works on the canonical Kaggle "Google Play Store Apps"
export (10,841 rows), on the bundled 11-row sample, and on any CSV/XLSX with
the same spirit of columns (aliases handled). Consumed by:
  * train_models.py   (ML training)
  * data/apps.json    (dashboard frontend charts 01-06)

Run:
    python clean.py                        # auto-picks data/play_store.csv, else data/sample_apps.csv
    python clean.py --raw my_data.xlsx     # explicit source (CSV or XLSX)
    python clean.py --tiers "Name:lo:hi,..."   # override the 4 install bands

Writes:
    <out>/apps_cleaned.csv      cleaned rows
    <out>/apps.json             cleaned rows + report (fetched by the dashboard)
    <out>/cleaning_report.json  provenance: raw/cleaned/removed counts, field stats

Documented rules
----------------
1.  App and Category are the identity columns: rows missing either are dropped
    (they cannot be identified or grouped).
2.  Corrupted shifted rows: the canonical Play export contains a famous row
    ("Life Made WI-Fi Touchscreen Photo Frame") whose cells are shifted left —
    its Category cell holds "1.9" (the rating) and its Rating cell holds "19".
    Rule: if the Category value is purely numeric AND the Rating cell parses to
    a number outside 1-5, the row is dropped and counted as
    `dropped_corrupted_rows` (its app names are listed in the report). This is
    the only row-level heuristic drop in the pipeline.
3.  Installs  : "1,000,000+" / "1000000" / 1000000 -> integer LOWER BOUND of the
    reported download band. Google Play reports installs in bands (10, 50,
    100, ... 1B+); the printed number is the band floor, NOT the exact download
    count. Blank/invalid -> missing (row kept, field counted).
4.  Reviews   : integer. Blank ("") is treated as 0 (review count not reported).
5.  Rating    : float, kept only if 1.0 <= r <= 5.0, else missing.
6.  Size      : "86M" -> 86.0 MB, "72K" -> 72/1024 MB, "Varies with device" ->
    missing. Missing size is a real state (common on Play), not an error.
7.  Price     : "" or "0" -> 0.0. Legitimate zero prices are preserved, never
    imputed to a nonzero value. "$1.99" -> 1.99. type = Paid / Free / Unknown.
    Listed price is a price tag, NOT observed revenue.
8.  Last Updated: parsed from "Jan 15, 2024", "January 15, 2024", "2024-01-15",
    "15-Jan-2024", "01/15/2024". Unparseable -> missing (counted, row kept).
9.  Category  : trimmed; underscores/hyphens turned into spaces; title-cased.
    "ART_AND_DESIGN" -> "Art And Design" is a display normalization (no names
    are merged). ONE documented spelling fix: "Commication" -> "Communication"
    (typo present in the project's sample data).
10. Duplicates: exact duplicates on (app, category, rating, reviews, installs,
    size_mb, price) -> keep first. Same app name with different metadata -> kept
    (possibly different listings), only counted.
11. Eligibility: M1 (rating regression) rows = rows with a valid rating.
                 M2 (install tier) rows   = rows with a valid installs bound.

Install-tier bands (M2 target) — default: log-spaced, left-inclusive, aligned
with Play's own band structure:

    Under 10K : [0, 10,000)
    10K-1M    : [10,000, 1,000,000)
    1M-100M   : [1,000,000, 100,000,000)
    100M+     : [100,000,000, infinity)

Override:  python clean.py --tiers "Under 10K:0:10000,10K-1M:10000:1000000,1M-100M:1000000:100000000,100M+:100000000:inf"
The chosen bands are written to the cleaning report and used by training and
the dashboard automatically — one change, consistent everywhere.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

# The single documented spelling correction (typo exists in the sample data).
CATEGORY_TYPO_MAP = {"Commication": "Communication"}

DEFAULT_TIER_BOUNDS = [
    (0, 10_000, "Under 10K"),
    (10_000, 1_000_000, "10K-1M"),
    (1_000_000, 100_000_000, "1M-100M"),
    (100_000_000, float("inf"), "100M+"),
]


def parse_tier_arg(s: str) -> list:
    """Parse --tiers 'Name:lo:hi,Name:lo:hi,...' (hi may be 'inf')."""
    bounds = []
    for part in s.split(","):
        bits = [p.strip() for p in part.split(":")]
        if len(bits) != 3:
            sys.exit(f"Bad --tiers part: {part!r} (expected Name:lo:hi)")
        name, lo, hi = bits
        hi_f = float("inf") if hi.lower() == "inf" else float(hi)
        bounds.append((float(lo), hi_f, name))
    bounds.sort(key=lambda b: b[0])
    return bounds


def install_tier(installs, bounds=None):
    """Map an installs lower bound to its band name (None if missing)."""
    if bounds is None:
        bounds = DEFAULT_TIER_BOUNDS
    if installs is None or (isinstance(installs, float) and not np.isfinite(installs)):
        return None
    for lo, hi, name in bounds:
        if lo <= installs < hi:
            return name
    return None


def _title_case(s: str) -> str:
    return re.sub(r"\w\S*", lambda m: m.group(0)[0].upper() + m.group(0)[1:].lower(), s)


def clean_category(v):
    """Rule 9: trim, underscores/hyphens -> spaces, title case, typo fix."""
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() in ("nan", "none"):
        return None
    s = re.sub(r"[_\-]+", " ", s)
    t = _title_case(s)
    return CATEGORY_TYPO_MAP.get(t, t)


def parse_installs(v):
    """Rule 3: reported download band LOWER BOUND -> int, else NaN."""
    if v is None:
        return np.nan
    if isinstance(v, (int, np.integer)):
        return int(v) if v >= 0 else np.nan
    if isinstance(v, (float, np.floating)):
        return int(v) if np.isfinite(v) and v >= 0 else np.nan
    s = str(v).strip().replace(",", "").replace(" ", "")
    if s == "" or s.lower() in ("nan", "none"):
        return np.nan
    if s.endswith("+"):
        s = s[:-1]
    if not re.fullmatch(r"\d+", s):
        return np.nan
    return int(s)


def parse_reviews(v):
    """Rule 4: int >= 0; blank/missing -> 0 (documented)."""
    if v is None:
        return 0
    if isinstance(v, (int, np.integer)):
        return max(0, int(v))
    if isinstance(v, (float, np.floating)):
        return int(v) if np.isfinite(v) and v > 0 else 0
    s = str(v).strip().replace(",", "")
    if s == "" or s.lower() in ("nan", "none"):
        return 0
    try:
        f = float(s)
        return int(f) if np.isfinite(f) and f >= 0 else 0
    except ValueError:
        return 0


def parse_price(v):
    """Rule 7: ''/0 -> 0.0 (free, preserved); '$1.99' -> 1.99; unparseable -> NaN."""
    if v is None:
        return 0.0
    if isinstance(v, (int, np.integer)):
        return float(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else 0.0
    s = str(v).strip()
    if s == "" or s.lower() in ("nan", "none"):
        return 0.0
    s = s.replace("$", "").replace(",", "")
    try:
        f = float(s)
        return f if np.isfinite(f) else np.nan
    except ValueError:
        return np.nan


def parse_size_mb(v):
    """Rule 6: '86M' -> 86.0, '72K' -> 72/1024, 'Varies with device' -> NaN."""
    if v is None:
        return np.nan
    if isinstance(v, (int, np.integer)):
        return float(v) if v > 0 else np.nan
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) and v > 0 else np.nan
    s = str(v).strip().upper().replace(",", "")
    if s in ("", "NAN", "NONE", "VARIES WITH DEVICE"):
        return np.nan
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([KM]?)", s)
    if not m:
        return np.nan
    val = float(m.group(1))
    if m.group(2) == "K":
        val /= 1024.0
    return val


def parse_rating(v):
    """Rule 5: float in [1, 5], else NaN."""
    if v is None:
        return np.nan
    try:
        f = float(v)
    except (TypeError, ValueError):
        return np.nan
    return f if np.isfinite(f) and 1.0 <= f <= 5.0 else np.nan


def parse_last_updated(v):
    """Rule 8: several common Play-export formats; None if unparseable."""
    if v is None:
        return None
    if isinstance(v, pd.Timestamp):
        return v.date() if pd.notna(v) else None
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none"):
        return None
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%Y-%m-%d", "%d-%b-%Y", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    try:
        ts = pd.to_datetime(s, errors="raise")
        return ts.date()
    except Exception:
        return None


def parse_sentiment(v):
    """Optional TextBlob-style subjectivity score in [0, 1]; NaN otherwise."""
    try:
        f = float(v)
        return f if np.isfinite(f) and 0.0 <= f <= 1.0 else np.nan
    except (TypeError, ValueError):
        return np.nan


def find_col(df: pd.DataFrame, *names):
    """Case/whitespace-insensitive column lookup."""
    for n in names:
        for c in df.columns:
            if str(c).strip().lower() == n.lower():
                return c
    return None


def load_raw(path: Path) -> pd.DataFrame:
    """Read the raw export as strings so parsing stays explicit and auditable."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=str)
    if path.suffix.lower() in (".xlsx", ".xls"):
        try:
            return pd.read_excel(path, dtype=str)
        except ImportError:
            sys.exit("openpyxl is required for .xlsx input: pip install openpyxl")
    sys.exit(f"Unsupported file type: {path.suffix} (use .csv or .xlsx)")


def _raw_rating_out_of_range(s: str) -> bool:
    try:
        f = float(s)
        return np.isfinite(f) and (f < 1.0 or f > 5.0)
    except (TypeError, ValueError):
        return False


def clean(df_raw: pd.DataFrame, tier_bounds: list = None) -> tuple[pd.DataFrame, dict]:
    """Apply the documented rules; return (cleaned df, report dict)."""
    if tier_bounds is None:
        tier_bounds = DEFAULT_TIER_BOUNDS
    tier_names = [b[2] for b in tier_bounds]

    n_raw = len(df_raw)
    report = {"raw_rows": n_raw}

    app_col = find_col(df_raw, "App", "app", "Name")
    cat_col = find_col(df_raw, "Category", "category", "Genre")
    if app_col is None or cat_col is None:
        sys.exit(f"Missing identity columns. Found: {list(df_raw.columns)}")

    rating_col = find_col(df_raw, "Rating", "rating")
    reviews_col = find_col(df_raw, "Reviews", "reviews", "Reviews count")
    size_col = find_col(df_raw, "Size", "size")
    installs_col = find_col(df_raw, "Installs", "installs")
    price_col = find_col(df_raw, "Price", "price")
    updated_col = find_col(df_raw, "Last Updated", "last_updated", "LastUpdated")
    sentiment_col = find_col(df_raw, "Sentiment_Subjectivity", "sentiment_subjectivity")

    # Rule 1: identity
    a = df_raw[app_col]
    c = df_raw[cat_col]
    has_id = a.notna() & a.astype(str).str.strip().ne("") & c.notna() & c.astype(str).str.strip().ne("")
    report["dropped_missing_identity"] = int((~has_id).sum())
    df = df_raw[has_id].copy()

    df["app"] = df[app_col].astype(str).str.strip()

    # Rule 9: category normalization (display) + typo fix
    before = df[cat_col].astype(str).str.strip()
    df["category"] = df[cat_col].map(clean_category)
    report["categories_normalized"] = int((before != df["category"].fillna(before)).sum())
    report["categories_typo_fixed"] = int(
        before.str.strip().map(lambda s: _title_case(re.sub(r"[_\-]+", " ", s)) in CATEGORY_TYPO_MAP).sum()
    )

    # Rules 3-8: field parsing
    def src(col):
        return df[col].tolist() if col else [None] * len(df)

    df["rating"] = [parse_rating(v) for v in src(rating_col)]
    df["reviews"] = [parse_reviews(v) for v in src(reviews_col)]
    df["installs"] = [parse_installs(v) for v in src(installs_col)]
    df["size_mb"] = [parse_size_mb(v) for v in src(size_col)]
    df["price"] = [parse_price(v) for v in src(price_col)]
    df["last_updated"] = [parse_last_updated(v) for v in src(updated_col)]
    df["sentiment_subjectivity"] = [parse_sentiment(v) for v in src(sentiment_col)]

    df["type"] = np.where(
        df["price"] > 0, "Paid", np.where(df["price"].isna(), "Unknown", "Free")
    )

    # Rule 2: corrupted shifted rows (numeric category + out-of-range numeric rating)
    cat_raw = df[cat_col].astype(str).str.strip()
    rating_raw = (df[rating_col].astype(str).str.strip() if rating_col
                  else pd.Series([""] * len(df), index=df.index))
    corrupted = (
        cat_raw.map(lambda s: bool(re.fullmatch(r"\d+(?:\.\d+)?", s)))
        & df["rating"].isna()
        & rating_raw.map(_raw_rating_out_of_range)
    )
    report["dropped_corrupted_rows"] = int(corrupted.sum())
    report["corrupted_row_apps"] = df.loc[corrupted, "app"].head(5).tolist()
    df = df[~corrupted].copy()

    # Rule 10: duplicates
    key = ["app", "category", "rating", "reviews", "installs", "size_mb", "price"]
    dup_mask = df.duplicated(subset=key, keep="first")
    report["exact_duplicates_removed"] = int(dup_mask.sum())
    df = df[~dup_mask].copy()
    report["same_name_variants_kept"] = int(df["app"].duplicated(keep=False).sum())

    # Field-level missingness (after dedup)
    report["missing_installs"] = int(df["installs"].isna().sum())
    report["missing_rating"] = int(df["rating"].isna().sum())
    report["missing_size_mb"] = int(df["size_mb"].isna().sum())
    report["missing_price_unparseable"] = int(df["price"].isna().sum())
    report["missing_last_updated"] = int(df["last_updated"].isna().sum())

    # Rule 11: model eligibility
    report["cleaned_rows"] = len(df)
    report["rows_for_m1_rating"] = int(df["rating"].notna().sum())
    report["rows_for_m2_tier"] = int(df["installs"].notna().sum())

    # Pricing / category / tier summaries
    report["price_counts"] = {
        "free": int((df["price"] == 0).sum()),
        "paid": int((df["price"] > 0).sum()),
        "unknown": int(df["price"].isna().sum()),
    }
    report["category_counts"] = df["category"].value_counts().to_dict()
    tiers = df["installs"].map(lambda v: install_tier(v, tier_bounds))
    report["tier_counts"] = {name: int((tiers == name).sum()) for name in tier_names}

    out = df[
        ["app", "category", "rating", "reviews", "installs", "size_mb",
         "price", "type", "last_updated", "sentiment_subjectivity"]
    ].reset_index(drop=True)
    return out, report


def _jsonable_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "app": r["app"],
                "category": r["category"],
                "rating": None if pd.isna(r["rating"]) else round(float(r["rating"]), 2),
                "reviews": int(r["reviews"]),
                "installs": None if pd.isna(r["installs"]) else int(r["installs"]),
                "size_mb": None if pd.isna(r["size_mb"]) else round(float(r["size_mb"]), 3),
                "price": None if pd.isna(r["price"]) else round(float(r["price"]), 2),
                "type": r["type"],
                "last_updated": r["last_updated"].isoformat() if r["last_updated"] else None,
                "sentiment_subjectivity": None if pd.isna(r["sentiment_subjectivity"])
                else round(float(r["sentiment_subjectivity"]), 3),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("Run:")[0])
    ap.add_argument("--raw", default=None, help="path to raw CSV/XLSX (auto-detect if omitted)")
    ap.add_argument("--out-dir", default=str(DATA_DIR))
    ap.add_argument("--tiers", default=None,
                    help='install bands override: "Name:lo:hi,..." (hi may be inf)')
    args = ap.parse_args()

    tier_bounds = parse_tier_arg(args.tiers) if args.tiers else DEFAULT_TIER_BOUNDS

    if args.raw:
        raw_path = Path(args.raw)
    elif (DATA_DIR / "play_store.csv").exists():
        raw_path = DATA_DIR / "play_store.csv"
    elif (DATA_DIR / "sample_apps.csv").exists():
        raw_path = DATA_DIR / "sample_apps.csv"
    else:
        sys.exit("No dataset found. Put your raw data at data/play_store.csv "
                 "(or pass --raw). The bundled sample is data/sample_apps.csv.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    is_sample = raw_path.name.startswith("sample")

    df_raw = load_raw(raw_path)
    cleaned, report = clean(df_raw, tier_bounds)

    report.update(
        {
            "source_file": str(raw_path.relative_to(ROOT)) if raw_path.is_relative_to(ROOT) else str(raw_path),
            "is_sample": is_sample,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tier_bounds": [[lo, (hi if np.isfinite(hi) else None), name] for lo, hi, name in tier_bounds],
        }
    )

    # Write outputs
    cleaned.to_csv(out_dir / "apps_cleaned.csv", index=False)
    (out_dir / "cleaning_report.json").write_text(json.dumps(report, indent=2))
    (out_dir / "apps.json").write_text(
        json.dumps({"source": report["source_file"], "is_sample": is_sample,
                    "generated_at": report["generated_at"], "report": report,
                    "rows": _jsonable_rows(cleaned)}, indent=2)
    )

    # Human-readable summary (used in README / viva)
    print("=== Cleaning report ===")
    for k, v in report.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for kk, vv in v.items():
                print(f"   {kk}: {vv}")
        else:
            print(f"{k}: {v}")
    if is_sample:
        print("\nWARNING: ran on the 11-row SAMPLE dataset. "
              "Replace with the full dataset (data/play_store.csv) before final results.")


if __name__ == "__main__":
    main()
