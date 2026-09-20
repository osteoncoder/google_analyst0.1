#!/usr/bin/env python3
"""
clean.py — documented, reusable cleaning pipeline for the APEX Play Store dashboard.

Dataset-agnostic: it works on the primary Google-Playstore dataset
(gauthamp10/Google-Playstore-Dataset, 2,312,944 rows, loaded via fetch_dataset.py),
on the committed 40,000-row sample of it (data/playstore_sample.csv), on the
older 10,841-row Kaggle export (data/play_store.csv), on the bundled 11-row
sample, and on any CSV/XLSX with the same spirit of columns (aliases handled).
Consumed by:
  * train_models.py   (ML training)
  * data/apps.json    (dashboard frontend charts 01-06)

Run:
    python clean.py                        # auto-picks the source (see order below)
    python clean.py --raw my_data.xlsx     # explicit source (CSV or XLSX)
    python clean.py --tiers "Name:lo:hi,..."   # override the 4 install bands

Source resolution (first match wins):
    1. data/raw/playstore_full.csv     full 2.3M-row dataset (fetch_dataset.py)
    2. data/playstore_sample.csv       committed stratified 40k sample of it
    3. data/play_store.csv             older 10,841-row Kaggle export (fallback)
    4. data/sample_apps.csv            11-row mechanical-test sample (last resort)

Writes:
    <out>/apps_cleaned.csv      cleaned rows
    <out>/apps.json             cleaned rows + report (fetched by the dashboard)
    <out>/cleaning_report.json  provenance: raw/cleaned/removed counts, field stats

Column aliases understood for the primary dataset (gauthamp10, 24 columns):
    App Name -> app          Rating Count -> reviews      Price     -> price
    Category -> category     Installs     -> installs     Last Updated -> last_updated
    Rating   -> rating       Size         -> size_mb
    Plus, since rule 13, the columns behind the derived app-profile features:
    Released, Scraped Time, Developer Id, Minimum Android, Content Rating,
    Ad Supported, In App Purchases, Editors Choice.
    (Still unconsumed: App Id, Free, Currency, Maximum Installs, Developer
     Website, Developer Email, Privacy Policy.)

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
    count. Blank/invalid -> missing (row kept, field counted).4.  Reviews   : integer. Blank ("") is treated as 0 (review count not reported).
5.  Rating    : float, kept only if 1.0 <= r <= 5.0, else missing. The primary
    dataset encodes "not yet rated" as 0.0 (~47% of its rows) — those become
    missing (counted), never fake zeros, so M1 trains only on real ratings.
6.  Size      : "86M" -> 86.0 MB, "72K" -> 72/1024 MB, "Varies with device" ->
    missing. Missing size is a real state (common on Play), not an error.
7.  Price     : "" or "0" -> 0.0. Legitimate zero prices are preserved, never
    imputed to a nonzero value. "$1.99" -> 1.99. A dataset with NO Price column
    at all gets unknown (NaN) prices, not $0 — it is not assumed to be
    all-free (mirrors data.js). type = Paid / Free / Unknown.
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
12. Dashboard export cap: apps.json is what the browser fetches, so the full
    2.3M-row dataset cannot be exported wholesale. Rows beyond
    --max-json-rows (default 60000) are omitted from apps.json using the same
    deterministic content-hash selection as the committed sample, stratified by
    install tier with a per-tier floor — the dashboard then shows a clearly
    labelled "N of M rows" note. The cleaned CSV and the ML training always use
    ALL cleaned rows; only the browser payload is capped.
13. Derived app-profile features (added later than rules 1-12, hence the
    number): the primary dataset's extra columns become eight model inputs —
    app_age_days, days_since_update, developer_app_count, min_android,
    ad_supported, in_app_purchases, editors_choice (numeric) and
    content_rating (categorical). A dataset without those raw columns simply
    yields NaN/None, which train_models.py drops from the feature set; a
    negative date difference (Released after Scraped Time) is NaN, never 0.
    These columns are written to apps_cleaned.csv for training but are NOT
    added to apps.json, so the browser payload does not grow.

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
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

# Source resolution order (first existing file wins) — see module docstring.
SOURCE_CANDIDATES = [
    "raw/playstore_full.csv",   # full 2.31M-row dataset built by fetch_dataset.py
    "playstore_sample.csv",     # committed deterministic stratified 40k sample
    "play_store.csv",           # older 10,841-row Kaggle export (fallback)
    "sample_apps.csv",          # 11-row mechanical-test sample (last resort)
]
# A file counts as a SAMPLE when its name says so AND it is far smaller than the
# full dataset — so a real multi-hundred-thousand-row export that merely has
# "sample" in its name is not mislabelled (and vice versa: a small full export
# is not claimed to be a sample).
SAMPLE_ROW_LIMIT = 200_000
DEFAULT_MAX_JSON_ROWS = 60_000
JSON_TIER_FLOOR = 250           # min dashboard rows per install tier when capping

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


def allocate_quotas(counts: dict, target: int, floor: int) -> dict:
    """Proportional (largest-remainder) quotas with a per-stratum floor.

    Used both by fetch_dataset.py (which rows go into the committed sample) and
    by clean.py (which rows go into the capped dashboard payload), so the two
    selections follow identical, documented rules.
    """
    total = sum(counts.values())
    if target >= total:
        return dict(counts)
    quotas = {s: min(n, floor) for s, n in counts.items()}
    remaining = target - sum(quotas.values())
    if remaining > 0:
        pool = {s: counts[s] - quotas[s] for s in counts}
        pool_total = sum(pool.values())
        if pool_total:
            ideal = {s: remaining * pool[s] / pool_total for s in counts if pool[s] > 0}
            base = {s: int(v) for s, v in ideal.items()}
            leftovers = remaining - sum(base.values())
            order = sorted(ideal, key=lambda s: (-(ideal[s] - base[s]), -counts[s], str(s)))
            for s in order[:leftovers]:
                base[s] += 1
            for s, add in base.items():
                quotas[s] = min(counts[s], quotas[s] + add)
    return quotas


def _str_col(df: pd.DataFrame, col: str) -> np.ndarray:
    """Column as plain Python strings, missing -> 'nan'.

    Explicit because `astype(str)` is dtype-version dependent: pandas < 3 turns
    NaN into the literal 'nan', while pandas 3's str dtype keeps a real missing
    value — which then propagates through string concatenation as a float.
    """
    s = df[col]
    return s.astype(object).where(s.notna(), "nan").astype(str).to_numpy(dtype=object)


def _row_hashes(df: pd.DataFrame) -> np.ndarray:
    """Deterministic content hash per row (stable across machines and runs)."""
    cols = ("app", "category", "rating", "reviews", "installs", "size_mb", "price")
    values = [_str_col(df, c) for c in cols]
    keys = ["\x1f".join(parts) for parts in zip(*values)]
    return np.array(
        [int.from_bytes(hashlib.blake2b(k.encode("utf-8", "replace"), digest_size=8).digest(), "big")
         for k in keys],
        dtype=np.uint64,
    )


def cap_dashboard_rows(df: pd.DataFrame, tiers: pd.Series, max_rows: int,
                       floor: int = JSON_TIER_FLOOR) -> tuple[pd.DataFrame, dict | None]:
    """Keep at most `max_rows` rows for apps.json — stratified by install tier,
    chosen by the smallest content hash (rule 12). Returns (df, note-or-None)."""
    if max_rows is None or max_rows <= 0 or len(df) <= max_rows:
        return df, None
    tier_vals = tiers.fillna("__unknown__").astype(str).values
    counts = {str(k): int(v) for k, v in pd.Series(tier_vals).value_counts().items()}
    quotas = allocate_quotas(counts, max_rows, floor)
    hashes = _row_hashes(df)
    keep: list[int] = []
    for tier, quota in quotas.items():
        if quota <= 0:
            continue
        idx = np.flatnonzero(tier_vals == tier)
        if len(idx) <= quota:
            keep.extend(idx.tolist())
        else:
            keep.extend(idx[np.argsort(hashes[idx], kind="stable")[:quota]].tolist())
    keep_idx = np.sort(np.array(keep, dtype=np.int64))
    note = {
        "exported": int(len(keep_idx)),
        "available": int(len(df)),
        "rule": f"stratified by install tier (floor {floor} rows/tier), smallest content hash",
        "note": (f"apps.json carries {len(keep_idx):,} of {len(df):,} cleaned rows so the browser "
                 f"payload stays small; apps_cleaned.csv and ML training use ALL cleaned rows."),
    }
    return df.iloc[keep_idx].reset_index(drop=True), note


def clean_category(v):
    """Rule 9: trim, underscores/hyphens -> spaces, title case, typo fix.

    A value that is blank after normalization (e.g. "___" or "-") is missing,
    not a category named " ".
    """
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() in ("nan", "none"):
        return None
    s = re.sub(r"[_\-]+", " ", s)
    if s.strip() == "":
        return None
    t = _title_case(s).strip()
    return CATEGORY_TYPO_MAP.get(t, t) or None


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
    """Rule 7: ''/0 -> 0.0 (free, preserved); '$1.99' -> 1.99; column missing
    (None) -> NaN (unknown, mirrors data.js parsePrice(null) — a dataset without
    a Price column is NOT assumed to be all-free); unparseable -> NaN."""
    if v is None:
        return np.nan
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


# --- Rule 13 (derived app-profile features) -------------------------------
#
# The primary dataset carries columns the original pipeline never parsed
# (Released, Scraped Time, Developer Id, Minimum Android, Ad Supported,
# In App Purchases, Editors Choice, Content Rating). They are turned into
# model inputs here — one place, one definition — so `clean.py ->
# apps_cleaned.csv -> train_models.py` stays a single source of truth, and so
# a dataset that lacks them (the 10,841-row Kaggle export, the 11-row sample)
# simply yields NaNs that the training pipeline's imputer handles.
DERIVED_NUMERIC = [
    "app_age_days",          # Scraped Time - Released, in days
    "days_since_update",     # Scraped Time - Last Updated, in days
    "developer_app_count",   # listings published by the same Developer Id
    "min_android",           # "5.0 and up" -> 5.0
    "ad_supported",          # True/False -> 1.0/0.0
    "in_app_purchases",      # True/False -> 1.0/0.0
    "editors_choice",        # True/False -> 1.0/0.0
]
DERIVED_CATEGORICAL = ["content_rating"]     # "Everyone", "Teen", "Mature 17+", ...
DERIVED_COLUMNS = DERIVED_NUMERIC + DERIVED_CATEGORICAL


def parse_min_android(v):
    """Rule 13: '5.0 and up' -> 5.0. Non-numeric lead ('Varies with device') -> NaN."""
    if v is None:
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
        f = float(v)
        return f if np.isfinite(f) and 0.0 <= f <= 20.0 else np.nan
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none"):
        return np.nan
    m = re.match(r"(\d+(?:\.\d+)?)", s)
    if not m:
        return np.nan
    try:
        f = float(m.group(1))
    except ValueError:
        return np.nan
    return f if np.isfinite(f) and 0.0 <= f <= 20.0 else np.nan


def parse_bool_flag(v):
    """Rule 13: True/False (string or real bool) -> 1.0/0.0; anything else -> NaN."""
    if v is None:
        return np.nan
    if isinstance(v, (bool, np.bool_)):
        return 1.0 if v else 0.0
    s = str(v).strip().lower()
    if s in ("true", "1", "1.0", "yes", "y", "t"):
        return 1.0
    if s in ("false", "0", "0.0", "no", "n", "f"):
        return 0.0
    return np.nan


def _date_series(values) -> pd.Series | None:
    """Timestamps for one raw column, parsed with Rule 8's parser; None if absent."""
    return pd.to_datetime([parse_last_updated(v) for v in values], errors="coerce")


def find_col(df: pd.DataFrame, *names):
    """Case/whitespace-insensitive column lookup."""
    for n in names:
        for c in df.columns:
            if str(c).strip().lower() == n.lower():
                return c
    return None


def load_raw(path: Path) -> pd.DataFrame:
    """Read the raw export as strings so parsing stays explicit and auditable.

    CSV files are read twice: the header first, so only the alias columns the
    pipeline actually consumes are loaded. That keeps the full 2.31M-row file
    (~666 MB, 24 columns) inside a few GB of RAM instead of many.
    """
    if path.suffix.lower() == ".csv":
        header = pd.read_csv(path, nrows=0)
        consumed = _consumed_columns(header)
        keep = [c for c in header.columns if c in consumed]
        return pd.read_csv(path, dtype=str, usecols=keep or None)
    if path.suffix.lower() in (".xlsx", ".xls"):
        try:
            return pd.read_excel(path, dtype=str)
        except ImportError:
            sys.exit("openpyxl is required for .xlsx input: pip install openpyxl")
    sys.exit(f"Unsupported file type: {path.suffix} (use .csv or .xlsx)")


def _consumed_columns(header: pd.DataFrame) -> set:
    """Alias columns the pipeline reads (identity + parsed fields)."""
    wanted = [
        ("App", "Name", "App Name"), ("Category", "Genre"),
        ("Rating",), ("Reviews", "Reviews count", "Rating Count"),
        ("Installs", "Minimum Installs"), ("Size",), ("Price",),
        ("Last Updated", "LastUpdated"),
        ("Sentiment_Subjectivity", "sentiment_subjectivity"),
        # Rule 13: raw columns behind the derived app-profile features.
        # load_raw() reads with usecols, so a column NOT listed here is never
        # loaded at all — the derived feature is then silently all-NaN.
        ("Released",), ("Scraped Time",), ("Developer Id",), ("Content Rating",),
        ("Minimum Android",), ("Ad Supported",), ("In App Purchases",),
        ("Editors Choice",),
    ]
    out = set()
    for aliases in wanted:
        col = find_col(header, *aliases)
        if col is not None:
            out.add(col)
    return out


def resolve_raw_path(explicit: str | None) -> Path:
    """Explicit --raw wins; otherwise the documented candidate order."""
    if explicit:
        p = Path(explicit)
        if not p.exists():
            sys.exit(f"--raw {p} not found")
        return p
    for name in SOURCE_CANDIDATES:
        p = DATA_DIR / name
        if p.exists():
            return p
    sys.exit(
        "No dataset found. Options:\n"
        "  * full 2.31M-row dataset:  python fetch_dataset.py --sample 40000\n"
        "  * committed 40k sample:    data/playstore_sample.csv\n"
        "  * explicit file:           python clean.py --raw my_data.csv"
    )


def load_sampling_meta(raw_path: Path) -> dict | None:
    """Read the <name>.meta.json sidecar written by fetch_dataset.py, if present."""
    sidecar = raw_path.parent / (raw_path.stem + ".meta.json")
    if not sidecar.exists():
        return None
    try:
        meta = json.loads(sidecar.read_text())
    except json.JSONDecodeError:
        return None
    # Keep provenance compact: the full stratum table stays in the sidecar.
    meta.pop("strata", None)
    meta["sidecar"] = sidecar.name
    return meta


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

    app_col = find_col(df_raw, "App", "app", "Name", "App Name")
    cat_col = find_col(df_raw, "Category", "category", "Genre")
    if app_col is None or cat_col is None:
        sys.exit(f"Missing identity columns. Found: {list(df_raw.columns)}")

    rating_col = find_col(df_raw, "Rating", "rating")
    # "Rating Count" is the primary dataset's review column.
    reviews_col = find_col(df_raw, "Reviews", "reviews", "Reviews count", "Rating Count")
    size_col = find_col(df_raw, "Size", "size")
    # Prefer the printed band string; "Minimum Installs" is the same band floor
    # in numeric form and is used when the string column is absent.
    installs_col = find_col(df_raw, "Installs", "installs", "Minimum Installs")
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

    # Rule 13: derived app-profile features.
    # Placed AFTER the identity/type fields and BEFORE the corrupted-row and
    # duplicate filters, so every derived column inherits exactly the same
    # row set as the fields it is computed from.
    released_col = find_col(df_raw, "Released", "released")
    scraped_col = find_col(df_raw, "Scraped Time", "scraped time", "ScrapedTime")
    dev_col = find_col(df_raw, "Developer Id", "developer id", "Developer")
    content_col = find_col(df_raw, "Content Rating", "content rating")
    android_col = find_col(df_raw, "Minimum Android", "minimum android")
    ad_col = find_col(df_raw, "Ad Supported", "ad supported")
    iap_col = find_col(df_raw, "In App Purchases", "in app purchases")
    editors_col = find_col(df_raw, "Editors Choice", "editors choice")

    scraped_dt = _date_series(src(scraped_col)) if scraped_col else None
    released_dt = _date_series(src(released_col)) if released_col else None
    # df["last_updated"] holds datetime.date objects (Rule 8) -> align the dtype.
    updated_dt = pd.to_datetime(df["last_updated"], errors="coerce")

    def _day_diff(a, b):
        """(a - b) in whole days; negative (impossible dates) -> NaN, never 0."""
        if a is None or b is None:
            return pd.Series(np.nan, index=df.index, dtype=float)
        # pd.to_datetime(list) yields a DatetimeIndex, not a Series — align both.
        left = pd.Series(np.asarray(a), index=df.index)
        right = pd.Series(np.asarray(b), index=df.index)
        out = (left - right).dt.days.astype("float64")
        return out.where(out >= 0)

    df["app_age_days"] = _day_diff(scraped_dt, released_dt)
    df["days_since_update"] = _day_diff(scraped_dt, updated_dt)
    def _present(s):
        """Boolean mask: this date column exists AND parsed for this row."""
        if s is None:
            return pd.Series(False, index=df.index)
        return pd.Series(np.asarray(s), index=df.index).notna()

    both_age = _present(scraped_dt) & _present(released_dt)
    both_upd = _present(scraped_dt) & _present(updated_dt)
    report["negative_date_diffs_dropped_to_nan"] = {
        "app_age_days": int((df["app_age_days"].isna() & both_age).sum()),
        "days_since_update": int((df["days_since_update"].isna() & both_upd).sum()),
    }

    if dev_col:
        dev_key = df[dev_col].astype(str).str.strip()
        blank = dev_key.isin(("", "nan", "none"))
        counts = dev_key.value_counts()
        df["developer_app_count"] = dev_key.map(counts).astype(float)
        # A blank developer id is "unknown", not a one-app portfolio.
        df.loc[blank, "developer_app_count"] = np.nan
        report["developer_portfolio_note"] = (
            "developer_app_count counts listings per Developer Id over the whole "
            "cleaned dataset (not per split); the ablation in the README shows how "
            "much of the model's lift depends on it."
        )
    else:
        df["developer_app_count"] = np.nan

    df["min_android"] = [parse_min_android(v) for v in src(android_col)]
    df["ad_supported"] = [parse_bool_flag(v) for v in src(ad_col)]
    df["in_app_purchases"] = [parse_bool_flag(v) for v in src(iap_col)]
    df["editors_choice"] = [parse_bool_flag(v) for v in src(editors_col)]
    df["content_rating"] = [clean_category(v) for v in src(content_col)]

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

    # Rule 13 coverage: a derived column that is entirely NaN is useless as a
    # model input (and would make the median imputer produce NaN), so it is
    # reported here and dropped from the feature set by train_models.py.
    report["derived_features"] = {
        col: {
            "non_null": int(df[col].notna().sum()),
            "mean": (round(float(df[col].mean()), 3) if df[col].notna().any() else None),
            "median": (round(float(df[col].median()), 3) if df[col].notna().any() else None),
        }
        for col in DERIVED_NUMERIC
    }
    report["derived_features"]["content_rating"] = {
        "non_null": int(df["content_rating"].notna().sum()),
        "values": {str(k): int(v) for k, v in df["content_rating"].value_counts().items()},
    }
    report["derived_features_all_nan"] = [
        c for c in DERIVED_NUMERIC if not df[c].notna().any()
    ]

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
        + DERIVED_COLUMNS
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
    ap.add_argument("--max-json-rows", type=int, default=DEFAULT_MAX_JSON_ROWS,
                    help=f"cap on rows exported to apps.json (default {DEFAULT_MAX_JSON_ROWS}; 0 = no cap)")
    args = ap.parse_args()

    tier_bounds = parse_tier_arg(args.tiers) if args.tiers else DEFAULT_TIER_BOUNDS

    raw_path = resolve_raw_path(args.raw)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_raw(raw_path)
    # "is_sample" means: this file is a documented subset of a larger dataset.
    # Require BOTH a sample-ish name and a clearly-smaller-than-full row count,
    # so an export merely named like a sample is not mislabelled.
    is_sample = "sample" in raw_path.stem.lower() and len(df_raw) < SAMPLE_ROW_LIMIT
    cleaned, report = clean(df_raw, tier_bounds)

    sampling = load_sampling_meta(raw_path)  # provenance for sample files
    report.update(
        {
            "source_file": str(raw_path.relative_to(ROOT)) if raw_path.is_relative_to(ROOT) else str(raw_path),
            "is_sample": is_sample,
            "sampling": sampling,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tier_bounds": [[lo, (hi if np.isfinite(hi) else None), name] for lo, hi, name in tier_bounds],
        }
    )

    # Rule 12: the browser payload is capped; CSV/training keep every cleaned row.
    if args.max_json_rows and len(cleaned) > args.max_json_rows:
        tiers = cleaned["installs"].map(lambda v: install_tier(v, tier_bounds))
        json_df, cap_note = cap_dashboard_rows(cleaned, tiers, args.max_json_rows)
    else:
        json_df, cap_note = cleaned, None
    report["dashboard_export"] = cap_note or {
        "exported": int(len(json_df)),
        "available": int(len(cleaned)),
        "note": "all cleaned rows exported (under the cap)",
    }

    # Write outputs
    cleaned.to_csv(out_dir / "apps_cleaned.csv", index=False)
    (out_dir / "cleaning_report.json").write_text(json.dumps(report, indent=2, default=str))
    # apps.json is fetched by the browser: compact separators, no indent
    # (cleaning_report.json stays human-readable).
    (out_dir / "apps.json").write_text(
        json.dumps({"source": report["source_file"], "is_sample": is_sample,
                    "sampling": sampling, "generated_at": report["generated_at"],
                    "report": report, "rows": _jsonable_rows(json_df)},
                   separators=(",", ":"))
    )

    # Rule 12b: ALSO write the same payload as a <script>-loadable bundle.
    # fetch() is blocked on file:// pages (VS Code "Run Active File",
    # double-click), which used to silently drop the dashboard onto its
    # 11-row fallback. data.js loads this bundle when the fetch fails.
    try:
        from browser_export import write_dataset_bundle
        write_dataset_bundle()
    except Exception as exc:                                # pragma: no cover
        print(f"\nWARNING: data/apps_bundle.js could not be regenerated ({exc}).\n"
              f"         The dashboard still works over http(s); a file:// page will\n"
              f"         fall back to the 11-row sample.")

    # Human-readable summary (used in README / viva)
    print("=== Cleaning report ===")
    for k, v in report.items():
        if k == "category_counts":
            print(f"category_counts: {len(v)} categories")
            continue
        if isinstance(v, dict):
            print(f"{k}:")
            for kk, vv in v.items():
                if kk == "strata" or (isinstance(vv, dict) and len(vv) > 12):
                    print(f"   {kk}: <{len(vv)} entries>")
                else:
                    print(f"   {kk}: {vv}")
        elif isinstance(v, list):
            print(f"{k}: {v[:4]}{' …' if len(v) > 4 else ''}")
        else:
            print(f"{k}: {v}")
    if is_sample:
        full = (sampling or {}).get("full_rows")
        if full:
            print(f"\nNOTE: ran on a {len(df_raw):,}-row SAMPLE of the {full:,}-row "
                  f"Google-Playstore dataset. Results are representative but not full-scale — "
                  f"run `python fetch_dataset.py --sample {len(df_raw)}` once and re-run "
                  f"`python clean.py` (it will pick data/raw/playstore_full.csv automatically) "
                  f"for full-scale numbers.")
        else:
            print("\nWARNING: ran on the 11-row SAMPLE dataset. "
                  "Replace with a real dataset (see data/playstore_sample.csv) before final results.")
    if cap_note:
        print(f"\nNOTE: apps.json exports {cap_note['exported']:,} of {cap_note['available']:,} "
              f"cleaned rows (browser payload cap, rule 12).")


if __name__ == "__main__":
    main()
