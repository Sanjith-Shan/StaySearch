# StaySearch

StaySearch is a vacation-rental search and relevance system built on public data: a two-tower
retrieval model trained in PyTorch, a clustered (IVF) index whose cluster id is an ordinary Apache
Lucene filter, an O(1) availability structure that takes live calendar updates, and LightGBM and
neural rankers, served from one Kotlin service through ONNX Runtime. It rebuilds a published
search design (credited in [DESIGN.md](DESIGN.md)) on real listings, calendars and reviews from
[Inside Airbnb](https://insideairbnb.com/get-the-data/) (CC BY 4.0) for ten cities, to measure the
trade-offs that design describes. **What it does not claim:** there are no searches, clicks or
bookings in public data, so the positive label is a **reviewed stay** (a home the guest went on to
review, which Inside Airbnb estimates covers about half of stays), and the search behind each
positive (dates, nights, guests, area) is **reconstructed** from the review date and the listing's
rules; experiment exp8 measures how much that reconstruction leaks. Hard negatives are substitutes
for the behavioural negatives a real search log would give. A closed calendar night is booked or
blocked by the host, indistinguishably. Price is one quote per listing, not per night, and some
snapshots have none (carried forward point in time). Scale is one laptop, not a marketplace. This
project is not affiliated with, endorsed by, or connected to Airbnb or Inside Airbnb.

<!-- RESULTS-START -->
Results are filled in from `NUMBERS.md` once each milestone lands.
<!-- RESULTS-END -->

## How it works

```
 offline (Python)                                     online (Kotlin, one JVM)
 Inside Airbnb gz -> Parquet (DuckDB)                 /search {place|bbox, checkin, checkout, guests, guest_id?}
 labels + point-in-time eligible sets + samplers        1. query tower (ONNX Runtime), guest history from a KV map
 two-tower training (PyTorch)                           2. nprobe nearest IVF centroids
 listing tower batch -> embeddings                      3. Lucene: cluster_id IN (...) AND LatLonPoint box
 k-means (FAISS) -> centroids, cluster ids                 AND capacity AND nights AND availability (custom Query)
 LightGBM + DNN rankers -> ONNX                         4. exact distance on the candidates, top K
 snapshot diffs -> real update stream                   5. ranker (ONNX) over K, top T
                                                      update consumer: calendar/price/listing events -> in-place
                                                         availability store + Lucene updateDocument + NRT refresh
```

- **Labels** ([DESIGN.md](DESIGN.md) section 2): a review becomes a reviewed stay; check-out, nights,
  guests and area are reconstructed; the eligible set is every listing in the area that was open
  for those nights in the calendar of the snapshot before the stay, and positives that were not
  open then are dropped and counted.
- **Time split:** towers train on stays before the first snapshot, rankers on the next quarter,
  validation on the one after, and every reported number on the last quarter (March to June 2026).
- **Update stream:** real calendar flips, price changes and new or removed listings between the
  March and June 2026 snapshots, replayed in-process at 100 to 10,000 events/s (not a broker).

## Layout

| Path | What |
|---|---|
| `staysearch/ingest/download.py` | Quarter discovery by probing, streaming gz to Parquet under a disk budget |
| `staysearch/features.py` | Point-in-time listing features, S2 cells, amenities vocabulary |
| `staysearch/labels.py` | Reviewed stays, reconstructed searches, splits, eligibility, trips, history |
| `staysearch/updates.py` | The update stream from snapshot diffs |
| `staysearch/samplers.py` | Random, geo-hard, price-hard and model-mined negatives, false-negative guard |
| `staysearch/train_towers.py`, `towers.py` | Two-tower model and training |
| `staysearch/evaluate.py`, `baselines.py` | Point-in-time evaluation and the M1 baselines |
| `service/` | The Kotlin service: Lucene index, availability store, ONNX models, experiments |
| `results/*.jsonl` | Every measurement, with machine and load |

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
python -m staysearch.ingest.download      # about 270 MB of Parquet for 10 cities x 4 snapshots
python -m staysearch.features && python -m staysearch.labels && python -m staysearch.updates
python -m staysearch.m0_report
python -m staysearch.baselines
python -m staysearch.train_towers --name mix_l2 --sampler mix --similarity l2
pytest -q tests
cd service && ./gradlew build
```

## Data and license

Listing, calendar and review data: Inside Airbnb, https://insideairbnb.com/get-the-data/, licensed
under a Creative Commons Attribution 4.0 International License. The data is not in this repository;
the download script fetches it. Inside Airbnb's assumptions page is the source for the review rate
and length-of-stay figures used in the reconstruction.
