# Contract notes

Decisions taken where `BUNDLE_FORMAT.md` was silent or ambiguous. Python (`staysearch/qfeatures.py`)
follows the same choices.

1. **Golden file layouts.** `query_features.jsonl`: `{"request", "place_idx" (resolved), "qfeat"}`.
   `ranker_features.jsonl`: `{"request", "qemb", "listing_id", "rank", "x"}` (`qemb` included so the test does
   not depend on ONNX numerics; `rank` is the 0-based retrieval rank used for cross feature 14, since it
   depends on the candidate list; `x` has width `listing_static_dim + 15`). `request` uses the `/search` names: `place_idx`, `bbox`
   (`{min_lat, min_lng, max_lat, max_lng}` or null), `checkin`, `checkout`, `guests`, `guest_id`, `search_date`.
2. **Search area.** A place search uses the place's bbox, its `(lat, lng)` as the centre and its
   `area_km2`; feature 1 is 1 only for a city place. A bbox search uses the bbox centre and its
   planar area, feature 1 = 0, and the city place nearest the bbox centre (haversine, ties to the
   lowest index). If both are given, the bbox defines the area and the given place is the tower place.
3. Cross feature 1 on a bbox search uses the bbox centre and area.
4. Haversine uses a sphere of radius 6371.0 km (both sides).
5. A missing `search_date` means `as_of`. A negative lead goes in bucket 0 and is clamped to 0 for cross feature 8.
6. Cross feature 3 is `ln(1 + max(0, nights - min_nights))`.
7. Cross feature 6 is `(ln(1 + price) - 5) - hist_log_price_mean_norm`, where the history value is
   `mean(v)` over the guest's earlier listings, with `v = ln(1 + price) - 5` for a listing with a quote and `v = 0` without one (`qfeatures.history_block`). The service only copies this value from `guest_history.f32`, so the definition lives in Python alone.
8. Cross feature 11 reads guest block index `2 + room_type`.
9. `avail_b64` uses the same packing as one `availability.bin` row.
10. Updates: a `new` event for an existing id is an upsert; events for unknown or removed ids are
    skipped and counted; a price event does not update `ranker_static` (price-derived static
    features go stale until the next daily batch, as in the paper's design).
11. Dot similarity in HNSW uses Lucene's `MAXIMUM_INNER_PRODUCT` (`DOT_PRODUCT` requires unit vectors).
12. The ranker output is read by the name `score`, falling back to the first output; `[B]` and `[B,1]` both work.
13. Nights outside the horizon are closed. `horizon_days` must be a multiple of 8 (448 in the real bundle).
14. `manifest.synthetic` (optional boolean) marks the fake bundle so its measurements are labelled synthetic.
15. Tolerance: an absolute 1e-5 is finer than float32 resolution for cross feature 0 (a squared
    distance between 128 and 256 has float32 spacing of about 1.5e-5). Tests use
    `max(1e-5, 1e-6 * |expected|)`, and the service computes feature 0 in double before the cast.
16. Cross features 12 and 13 (gaps) are capped at 28 nights. gap_before walks back from `checkin - 1`
    and stops after night 0; gap_after starts at the checkout night. Nights before night 0 or at or
    beyond the horizon are closed. The service reads them from the live availability store under the
    row's seqlock (gap_after is `min(run[checkout], 28)`, exact because 28 < 255), so updates applied
    since load are reflected; golden rows use the bundle's availability with no updates.
17. Cross feature 14 is `ln(1 + min(rank, K)) / ln(1 + K)` with `K = manifest.ranker_rank_k` (default
    100) and rank the 0-based position in the retrieval top K (distance ascending, ties by row index,
    the order `topK` returns). A negative rank is treated as 0.
18. `/search` `k` defaults to `ranker_rank_k` when a ranker is used and to 500 with `ranker: "none"`;
    an explicit `k` always wins. `/meta` reports the effective default and both values. The
    experiment CLIs (`latency`, `replay-invariant`, `exp4`, `flexdates`) keep their explicit `--k 500` default.
19. Models refuses to start if `ranker_feat_dim != listing_static_dim + 15`.
