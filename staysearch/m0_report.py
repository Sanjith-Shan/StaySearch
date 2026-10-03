"""results/m0_data.jsonl: per-city counts that M0 must match against the source files."""
from __future__ import annotations

import json

import duckdb

from staysearch.config import DATA_DIR
from staysearch.features import PREP
from staysearch import results


def main() -> None:
    results.reset("m0_data")
    man = json.loads((DATA_DIR / "snapshots.json").read_text())
    lab = json.loads((PREP / "labels_summary.json").read_text())
    upd = json.loads((PREP / "updates_summary.json").read_text())
    con = duckdb.connect()
    for city, info in man.items():
        per_snap = {}
        for rank, snap in enumerate(info["snapshots"]):
            raw = con.execute(f"select count(*) from '{DATA_DIR}/listings/{city}/{snap}.parquet'").fetchone()[0]
            prep = con.execute(f"select count(*), sum((price_src='missing')::int), sum((price_src='carried')::int) "
                               f"from '{PREP}/listings_s{rank + 1}.parquet' where city='{city}'").fetchone()
            rec = info["per_snapshot"].get(snap, {})
            per_snap[f"S{rank + 1}"] = {"date": snap, "listings_source_rows": raw,
                                        "listings_after_parse": prep[0], "price_missing": prep[1],
                                        "price_carried_forward": prep[2],
                                        "calendar_rows": rec.get("calendar_rows"),
                                        "calendar_listings": rec.get("calendar_listings"),
                                        "calendar_open_share": (rec["calendar_open_rows"] / rec["calendar_rows"]
                                                                if rec.get("calendar_rows") else None)}
        results.append("m0_data", {"kind": "city", "city": city, "snapshots": per_snap,
                                   "reviews_source_rows": info.get("reviews_rows"),
                                   "reviews_snapshot": info.get("reviews_snapshot"),
                                   "labels": lab["counts"].get(city, {}),
                                   "update_stream": upd["per_city"].get(city, {})})
    results.append("m0_data", {"kind": "total", **{k: v for k, v in lab.items() if k != "counts"},
                               "update_events_total": upd["total_events"], "update_base_date": upd["base_date"]})


if __name__ == "__main__":
    main()
