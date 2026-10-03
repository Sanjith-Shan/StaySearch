"""Download Inside Airbnb snapshots and stream them straight to Parquet.

Disk is the binding constraint on the build machine, so a CSV is never decompressed to disk.
Each `.gz` is fetched to a temporary file, read by DuckDB, written to Parquet with only the
columns the pipeline needs, and deleted. Free space is checked before every download.

Quarter discovery: the Get the Data page lists only each city's latest snapshot, but earlier
quarters are still served at the same URL pattern. We probe a window of dates around each
expected quarter with HEAD requests and keep the date that answers 200.

Data: Inside Airbnb (https://insideairbnb.com/get-the-data/), licensed CC BY 4.0.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import numpy as np

from staysearch.config import CITIES, DATA_DIR, HORIZON_BITS

BASE = "https://data.insideairbnb.com"
MIN_FREE_GB = 2.0

LISTING_COLS = [
    "id", "last_scraped", "host_id", "host_since", "host_is_superhost", "host_listings_count",
    "host_identity_verified", "host_response_rate", "host_acceptance_rate",
    "neighbourhood_cleansed", "neighbourhood_group_cleansed", "latitude", "longitude",
    "property_type", "room_type", "accommodates", "bathrooms", "bathrooms_text", "bedrooms", "beds",
    "amenities", "price", "price_quote_price_per_night", "minimum_nights", "maximum_nights",
    "availability_30", "availability_60", "availability_90", "availability_365",
    "number_of_reviews", "number_of_reviews_ltm", "number_of_reviews_l30d",
    "estimated_occupancy_l365d", "first_review", "last_review",
    "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
    "review_scores_checkin", "review_scores_communication", "review_scores_location",
    "review_scores_value", "instant_bookable", "calculated_host_listings_count", "reviews_per_month",
]


def free_gb() -> float:
    return shutil.disk_usage(Path.home()).free / 1e9


def url_for(city: str, date: str, fname: str) -> str:
    return f"{BASE}/{CITIES[city]['path']}/{date}/data/{fname}"


def head_ok(url: str) -> bool:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status == 200
    except Exception:
        return False


def discover_quarters(city: str, latest: str, n_back: int = 3, window: int = 30) -> list[str]:
    """Probe for snapshots roughly every 91 days before `latest`. Returns dates oldest first."""
    found = [latest]
    anchor = dt.date.fromisoformat(latest)
    for _ in range(n_back):
        target = anchor - dt.timedelta(days=91)
        days = [target + dt.timedelta(days=o) for o in range(-window, window + 1)]
        days = [d for d in days if d < anchor - dt.timedelta(days=30)]
        urls = [url_for(city, d.isoformat(), "listings.csv.gz") for d in days]
        with ThreadPoolExecutor(16) as ex:
            ok = list(ex.map(head_ok, urls))
        hits = [d for d, o in zip(days, ok) if o]
        if not hits:
            print(f"[discover] {city}: no snapshot near {target}", file=sys.stderr)
            break
        best = min(hits, key=lambda d: abs((d - target).days))
        found.append(best.isoformat())
        anchor = best
    return sorted(found)


def fetch(url: str, dest: Path) -> None:
    if free_gb() < MIN_FREE_GB:
        raise SystemExit(f"stop: only {free_gb():.2f} GB free, budget floor is {MIN_FREE_GB} GB")
    subprocess.run(["curl", "-sSfL", "--retry", "3", "-o", str(dest), url], check=True)


def ingest_listings(con, gz: Path, out: Path) -> int:
    cols = {r[0] for r in con.execute(
        f"describe select * from read_csv('{gz}', all_varchar=true, header=true)").fetchall()}
    sel = ", ".join(f'"{c}"' if c in cols else f"NULL::VARCHAR as \"{c}\"" for c in LISTING_COLS)
    con.execute(f"""copy (select {sel} from read_csv('{gz}', all_varchar=true, header=true))
                    to '{out}' (format parquet, compression zstd)""")
    return con.execute(f"select count(*) from '{out}'").fetchone()[0]


def ingest_calendar(con, gz: Path, out: Path, snap: str) -> dict:
    """Pack each listing's calendar into a HORIZON_BITS bitset, day 0 = snapshot date.

    Keeps per-listing nightly min/max nights only as the listing-level min (first night) since the
    pipeline uses the listing's own minimum_nights/maximum_nights; the per-night columns are dropped.
    """
    snap_d = dt.date.fromisoformat(snap)
    rows = con.execute(f"""
        select cast(listing_id as bigint) as lid,
               date_diff('day', DATE '{snap}', cast(date as date)) as d,
               available = 't' as a
        from read_csv('{gz}', all_varchar=true, header=true)""").fetchnumpy()
    lid, d, a = rows["lid"], rows["d"].astype(np.int64), rows["a"].astype(bool)
    n_rows, n_open = len(lid), int(a.sum())
    keep = (d >= 0) & (d < HORIZON_BITS)
    lid, d, a = lid[keep], d[keep], a[keep]
    uniq, inv = np.unique(lid, return_inverse=True)
    bits = np.zeros((len(uniq), HORIZON_BITS), dtype=bool)
    covered = np.zeros((len(uniq), HORIZON_BITS), dtype=bool)
    bits[inv, d] = a
    covered[inv, d] = True
    packed = np.packbits(bits, axis=1, bitorder="little")
    cov = np.packbits(covered, axis=1, bitorder="little")
    import pyarrow as pa
    import pyarrow.parquet as pq
    tbl = pa.table({
        "listing_id": pa.array(uniq, pa.int64()),
        "open_bits": pa.array([b.tobytes() for b in packed], pa.binary(HORIZON_BITS // 8)),
        "covered_bits": pa.array([b.tobytes() for b in cov], pa.binary(HORIZON_BITS // 8)),
    })
    pq.write_table(tbl, out, compression="zstd")
    return {"calendar_rows": n_rows, "calendar_open_rows": n_open, "calendar_listings": len(uniq),
            "calendar_start": snap_d.isoformat()}


def ingest_reviews(con, gz: Path, out: Path) -> int:
    con.execute(f"""copy (select cast(id as bigint) as review_id, cast(listing_id as bigint) as listing_id,
                           cast(date as date) as date, cast(reviewer_id as bigint) as reviewer_id
                    from read_csv('{gz}', all_varchar=true, header=true)
                    where reviewer_id is not null)
                    to '{out}' (format parquet, compression zstd)""")
    return con.execute(f"select count(*) from '{out}'").fetchone()[0]


def run_city(city: str, latest: str, snapshots: list[str] | None, tmp: Path) -> dict:
    con = duckdb.connect()
    snaps = snapshots or discover_quarters(city, latest)
    info = {"city": city, "snapshots": snaps, "per_snapshot": {}}
    for snap in snaps:
        rec = {}
        for kind, fname in (("listings", "listings.csv.gz"), ("calendar", "calendar.csv.gz")):
            out = DATA_DIR / kind / city / f"{snap}.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists():
                continue
            gz = tmp / f"{city}_{snap}_{fname}"
            fetch(url_for(city, snap, fname), gz)
            rec[f"{kind}_gz_bytes"] = gz.stat().st_size
            if kind == "listings":
                rec["listings_rows"] = ingest_listings(con, gz, out)
            else:
                rec.update(ingest_calendar(con, gz, out, snap))
            gz.unlink()
        info["per_snapshot"][snap] = rec
    out = DATA_DIR / "reviews" / f"{city}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        gz = tmp / f"{city}_reviews.csv.gz"
        fetch(url_for(city, snaps[-1], "reviews.csv.gz"), gz)
        info["reviews_gz_bytes"] = gz.stat().st_size
        info["reviews_rows"] = ingest_reviews(con, gz, out)
        info["reviews_snapshot"] = snaps[-1]
        gz.unlink()
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="*", default=list(CITIES))
    args = ap.parse_args()
    tmp = DATA_DIR / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    manifest_path = DATA_DIR / "snapshots.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for city in args.cities:
        print(f"[download] {city} (free {free_gb():.2f} GB)", flush=True)
        prev = manifest.get(city, {})
        info = run_city(city, CITIES[city]["latest"], prev.get("snapshots"), tmp)
        for k, v in prev.items():
            info.setdefault(k, v)
        for snap, rec in prev.get("per_snapshot", {}).items():
            info["per_snapshot"].setdefault(snap, {}).update({k: v for k, v in rec.items()
                                                             if k not in info["per_snapshot"][snap]})
        manifest[city] = info
        manifest_path.write_text(json.dumps(manifest, indent=1))
        print(f"[download] {city} done: {info['snapshots']}", flush=True)


if __name__ == "__main__":
    main()
