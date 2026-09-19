#!/usr/bin/env python3
"""
fetch_dataset.py — download the full 2.31M-row Google-Playstore dataset and build
the deterministic stratified sample that this project ships as its default data.

Upstream
--------
    https://github.com/gauthamp10/Google-Playstore-Dataset  (MIT license)
    June-2021 Play Store scrape: 2,312,944 apps, 24 attributes, in three parts:

        Part1.csv.tar.gz   (header + 800,000 rows)
        Part2.csv.tar.gz   (         800,000 rows, NO header)
        Part3.csv.tar.gz   (         712,944 rows, NO header)

    Only Part1 carries the CSV header — upstream's own instructions are
    `cat Part?.csv > Googple-Playstore-Dataset.csv`. This script does the same
    byte-level concatenation, with a newline-safety check, and writes one
    canonical CSV:  <out>/playstore_full.csv

Usage
-----
    python fetch_dataset.py                     # download + combine (full CSV)
    python fetch_dataset.py --sample 40000      # also write data/playstore_sample.csv
    python fetch_dataset.py --parts-dir DIR     # skip download, use extracted parts
    python fetch_dataset.py --cleanup           # delete downloaded/extracted parts after

Why the committed data is a SAMPLE
----------------------------------
The full CSV (666 MB) and its derived artifacts (cleaned CSV, dashboard JSON,
trained pipelines) are far beyond practical git limits. So the repository ships
`data/playstore_sample.csv` plus `data/playstore_sample.meta.json` (selection
parameters, stratum counts, source URL — full provenance). `python clean.py`
and `python train_models.py` then run out of the box on the sample, and the
identical commands on the full CSV reproduce full-scale numbers.

The sample is NOT a random head/tail slice: it is stratified by
(category x install tier) with proportional quotas, and rows are chosen by a
deterministic content hash — so it is reproducible on any machine, covers every
category and every install band, and keeps the tier proportions of the full
dataset. Tiers come from clean.py (single source of truth).
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import itertools
import json
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

sys.path.insert(0, str(ROOT))
# Tier bands + quota allocation come from clean.py — one implementation only.
from clean import DEFAULT_TIER_BOUNDS, allocate_quotas, install_tier  # noqa: E402

REPO = "gauthamp10/Google-Playstore-Dataset"
BASE_URL = f"https://raw.githubusercontent.com/{REPO}/main/dataset"
PARTS = ["Part1.csv.tar.gz", "Part2.csv.tar.gz", "Part3.csv.tar.gz"]

# Sample discipline
DEFAULT_SAMPLE_ROWS = 40_000
STRATUM_FLOOR = 25          # min rows per non-empty (category x tier) stratum
CHUNK = 200_000
IDENTITY_COLS = ("App Name", "App Id", "Category", "Rating", "Installs", "Size", "Price")


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# download / extract
# --------------------------------------------------------------------------- #
def download(url: str, dest: Path, retries: int = 3, timeout: int = 120) -> None:
    """Streamed download with retries; skips files already present."""
    if dest.exists() and dest.stat().st_size > 0:
        log(f"  already present: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            log(f"  downloading {dest.name} (attempt {attempt}/{retries}) …")
            with urllib.request.urlopen(url, timeout=timeout) as r, part.open("wb") as fh:
                total = int(r.headers.get("Content-Length") or 0)
                done = 0
                while True:
                    buf = r.read(1 << 20)
                    if not buf:
                        break
                    fh.write(buf)
                    done += len(buf)
                    if total:
                        print(f"\r    {100 * done / total:5.1f}%  {done / 1e6:6.1f}/{total / 1e6:.1f} MB",
                              end="", flush=True)
            print()
            part.replace(dest)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if part.exists():
                part.unlink()
            log(f"    failed: {type(e).__name__}: {e}")
            if attempt == retries:
                sys.exit(
                    f"\nDownload failed for {url}\n"
                    f"Fallback: clone the repository and pass the extracted parts explicitly:\n"
                    f"    git clone --depth 1 https://github.com/{REPO}.git\n"
                    f"    python fetch_dataset.py --parts-dir {REPO.split('/')[1]}/dataset"
                )
            time.sleep(2 * attempt)


def fetch_parts(work_dir: Path) -> list[Path]:
    """Download the three tar.gz parts and extract the CSVs."""
    work_dir.mkdir(parents=True, exist_ok=True)
    csvs: list[Path] = []
    for name in PARTS:
        tarball = work_dir / name
        csv_path = work_dir / name.replace(".tar.gz", "")
        if not csv_path.exists():
            download(f"{BASE_URL}/{name}", tarball)
            log(f"  extracting {name} …")
            with tarfile.open(tarball) as tf:
                tf.extractall(work_dir)
        csvs.append(csv_path)
    return csvs


def combine_parts(csvs: list[Path], out: Path) -> int:
    """Byte-level concatenation of the parts (only the first has a header).

    Returns the number of data rows written (header excluded).
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    data_rows = 0
    log(f"combining {len(csvs)} parts -> {out}")
    with out.open("wb") as dst:
        for i, src in enumerate(csvs):
            size_before = dst.tell()
            with src.open("rb") as fh:
                # newline safety: guarantee the previous part ended with '\n'
                if size_before:
                    pass
                while True:
                    buf = fh.read(1 << 22)
                    if not buf:
                        break
                    dst.write(buf)
            # verify the tail byte is a newline so the next part starts on a new line
            with src.open("rb") as fh:
                fh.seek(-1, 2)
                if fh.read(1) != b"\n":
                    dst.write(b"\n")
            with src.open("rb") as fh:
                rows = sum(1 for _ in fh)
            data_rows += rows - (1 if i == 0 else 0)  # only part 1 has a header
            log(f"  {src.name}: {rows:,} lines ({rows - (1 if i == 0 else 0):,} data rows)")
    return data_rows


def resolve_parts(parts_dir: Path) -> list[Path]:
    """Use already-extracted Part?.csv files from an external clone."""
    if not parts_dir.exists():
        sys.exit(f"--parts-dir {parts_dir} does not exist")
    found = sorted(parts_dir.glob("Part*.csv"))
    if not found:
        found = sorted(parts_dir.glob("*.tar.gz"))
        if found:
            log(f"extracting {len(found)} tarballs in {parts_dir} …")
            for tb in found:
                with tarfile.open(tb) as tf:
                    tf.extractall(parts_dir)
            found = sorted(parts_dir.glob("Part*.csv"))
    if len(found) < 2:
        sys.exit(f"expected Part1.csv..Part3.csv in {parts_dir}, found: {[p.name for p in found]}")
    return found


# --------------------------------------------------------------------------- #
# deterministic stratified sampling
# --------------------------------------------------------------------------- #
def _row_key(row: dict) -> int:
    """Stable content hash — selection does not depend on chunking or RNG state."""
    payload = "\x1f".join(str(row.get(c, "")) for c in IDENTITY_COLS)
    return int.from_bytes(hashlib.blake2b(payload.encode("utf-8", "replace"), digest_size=8).digest(), "big")


def count_strata(path: Path, columns: list[str]) -> tuple[dict, int]:
    """Pass 1: rows per (category x install tier) stratum."""
    counts: dict[tuple[str, str], int] = {}
    total = 0
    for chunk in pd.read_csv(path, dtype=str, usecols=["Category", "Installs"], chunksize=CHUNK):
        cat = chunk["Category"].fillna("__missing__").astype(str).str.strip()
        tier = chunk["Installs"].map(_installs_to_tier)
        for key, n in pd.DataFrame({"cat": cat, "tier": tier}).value_counts().items():
            counts[key] = counts.get(key, 0) + int(n)
        total += len(chunk)
    return counts, total


def _installs_to_tier(v):
    """Installs band floor -> tier name via clean.py's default bands."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "__missing__"
    s = str(v).strip().replace(",", "").replace("+", "")
    if not s.isdigit():
        return "__missing__"
    return install_tier(int(s), DEFAULT_TIER_BOUNDS) or "__missing__"


def build_sample(path: Path, out: Path, target: int) -> dict:
    """Pass 2: keep the `quota` smallest content hashes per stratum (exact, deterministic)."""
    log(f"pass 1/2: counting strata in {path.name} …")
    counts, total = count_strata(path, list(IDENTITY_COLS))
    quotas = allocate_quotas(counts, target, STRATUM_FLOOR)
    log(f"  {total:,} rows in {len(counts)} strata; sampling {sum(quotas.values()):,} …")

    heaps: dict[tuple[str, str], list] = {}
    seq = itertools.count()          # monotonic tiebreaker (rows are never compared)
    columns: list[str] | None = None
    log("pass 2/2: streaming rows and selecting by content hash …")
    for chunk in pd.read_csv(path, dtype=str, chunksize=CHUNK):
        if columns is None:
            columns = list(chunk.columns)
        for row in chunk.to_dict("records"):
            key = (str(row.get("Category", "") or "__missing__").strip(), _installs_to_tier(row.get("Installs")))
            quota = quotas.get(key, 0)
            if quota <= 0:
                continue
            h = _row_key(row)
            heap = heaps.setdefault(key, [])
            if len(heap) < quota:
                heapq.heappush(heap, (-h, next(seq), row))
            elif h < -heap[0][0]:
                heapq.heapreplace(heap, (-h, next(seq), row))

    rows = [r for heap in heaps.values() for _, _, r in heap]
    sort_cols = [c for c in IDENTITY_COLS if c in (columns or [])]
    df = pd.DataFrame(rows, columns=columns).sort_values(sort_cols, kind="stable")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log(f"wrote {out} ({len(df):,} rows, {out.stat().st_size / 1e6:.1f} MB)")

    # Tier shares: the per-stratum floor deliberately over-samples rare tiers
    # (e.g. 100M+) so every stratum can be split for training. Recorded here so
    # the difference is measurable, not hidden.
    tier_full: dict[str, int] = {}
    tier_sampled: dict[str, int] = {}
    for (cat, tier) in counts:
        tier_full[tier] = tier_full.get(tier, 0) + counts[(cat, tier)]
        tier_sampled[tier] = tier_sampled.get(tier, 0) + len(heaps.get((cat, tier), []))

    return {
        "sample_rows": len(df),
        "full_rows": total,
        "strategy": "stratified by (category x install tier), proportional quotas (largest-remainder), "
                    "rows selected by the smallest content hash — reproducible without an RNG seed",
        "stratum_floor": STRATUM_FLOOR,
        "floor_note": (f"every non-empty stratum keeps at least {STRATUM_FLOOR} rows, so rare strata "
                       f"(few categories of 100M+ apps) are over-represented in the sample relative to "
                       f"the full dataset — see tier_shares. Full-scale proportions return on a full run."),
        "tier_shares": {
            tier: {
                "full": tier_full[tier],
                "sampled": tier_sampled.get(tier, 0),
                "full_pct": round(100 * tier_full[tier] / total, 4),
                "sample_pct": round(100 * tier_sampled.get(tier, 0) / len(df), 4),
            }
            for tier in sorted(tier_full, key=lambda t: -tier_full[t])
        },
        "strata": {
            f"{cat} | {tier}": {"full": counts[(cat, tier)], "sampled": len(heaps.get((cat, tier), []))}
            for cat, tier in sorted(counts)
        },
        "column_mapping": {
            "app": "App Name",
            "category": "Category",
            "rating": "Rating",
            "reviews": "Rating Count",
            "installs": "Installs (band floor, e.g. '50,000+')",
            "size_mb": "Size",
            "price": "Price",
            "last_updated": "Last Updated",
        },
    }


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("Usage")[0])
    ap.add_argument("--out", default=str(RAW_DIR / "playstore_full.csv"),
                    help="where the combined full CSV goes (default: data/raw/playstore_full.csv)")
    ap.add_argument("--work-dir", default=str(RAW_DIR / "_parts"),
                    help="where tar.gz parts are downloaded/extracted")
    ap.add_argument("--parts-dir", default=None,
                    help="use already-extracted Part*.csv from an external clone (skips download)")
    ap.add_argument("--sample", type=int, default=None, metavar="N",
                    help=f"also write a stratified sample of N rows (default when --sample-only: "
                         f"{DEFAULT_SAMPLE_ROWS})")
    ap.add_argument("--sample-out", default=str(DATA_DIR / "playstore_sample.csv"))
    ap.add_argument("--sample-only", action="store_true",
                    help="skip build of the full CSV (use with --parts-dir / an existing --out)")
    ap.add_argument("--cleanup", action="store_true",
                    help="delete downloaded tarballs and extracted parts when done")
    args = ap.parse_args()

    root_gitignored = RAW_DIR
    log(f"Google-Playstore dataset  ·  source: https://github.com/{REPO} (MIT)")
    raw_path = Path(args.out)

    if args.sample_only:
        if not raw_path.exists():
            sys.exit(f"--sample-only needs an existing combined CSV at {raw_path}")
        log(f"using existing combined CSV: {raw_path}")
    else:
        parts = resolve_parts(Path(args.parts_dir)) if args.parts_dir else fetch_parts(Path(args.work_dir))
        rows = combine_parts(parts, raw_path)
        log(f"wrote {raw_path} ({raw_path.stat().st_size / 1e6:.1f} MB, {rows:,} data rows)")

    if args.sample or args.sample_only:
        target = args.sample or DEFAULT_SAMPLE_ROWS
        meta = build_sample(raw_path, Path(args.sample_out), target)
        meta.update({
            "source_repo": f"https://github.com/{REPO}",
            "license": "MIT",
            "scraped": "2021-06",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "full_csv": str(raw_path),
        })
        meta_path = Path(args.sample_out).parent / (Path(args.sample_out).stem + ".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2))
        log(f"wrote {meta_path} (provenance: source, quotas, strata)")
        log("\nNext: python clean.py && python train_models.py")

    if args.cleanup:
        for p in [Path(args.work_dir)]:
            if p.exists():
                shutil.rmtree(p)
                log(f"cleaned up {p}")


if __name__ == "__main__":
    main()
