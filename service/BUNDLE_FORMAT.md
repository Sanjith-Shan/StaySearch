# Serving bundle format

The offline Python pipeline writes one directory, the **bundle**, and the Kotlin service reads it
at startup. Nothing else crosses the boundary. All binary arrays are little-endian, row-major, with
no header; their shapes come from `manifest.json`. Row order is the same in every per-listing file
(row `i` of `listings.jsonl` is row `i` of `embeddings.f32`, `availability.bin` and so on).

## manifest.json

```json
{
  "version": 1,
  "as_of": "2026-03-20",            // the snapshot this bundle represents (the search clock's "now")
  "base_date": "2026-03-01",        // calendar day 0 for every bitset and every event night offset
  "horizon_days": 448,              // bits per listing in availability.bin (7 x 64)
  "n_listings": 123456,
  "dim": 64,                        // embedding dimension
  "similarity": "l2",               // "l2": smaller squared distance is better; "dot": larger is better
  "n_clusters": 128,
  "n_places": 900,
  "q_feat_dim": 87,                 // width of the query tower's "qfeat" input
  "listing_static_dim": 28,         // width of ranker_static.f32 rows
  "ranker_feat_dim": 40,            // listing_static_dim + 12 cross features
  "models": {"query_tower": "query_tower.onnx", "ranker": "ranker.onnx", "ranker_gbdt": "ranker_gbdt.onnx"},
  "cities": ["seattle", "..."]
}
```

## Per-listing files

| File | Shape | Content |
|---|---|---|
| `listings.jsonl` | N lines | `{"id": long, "city": str, "place_city": int, "place_nbhd": int, "lat": double, "lng": double, "accommodates": int, "min_nights": int, "max_nights": int, "price": double, "room_type": int}` . `room_type`: 0 entire home/apt, 1 private room, 2 shared room, 3 hotel room. `place_*` are indices into `places.jsonl`. `price` is the listing-level nightly quote in local currency (0 if unknown) |
| `embeddings.f32` | N x dim float32 | listing tower output |
| `cluster_ids.i32` | N int32 | IVF cluster of each listing (nearest centroid under `similarity`) |
| `centroids.f32` | n_clusters x dim float32 | k-means centroids |
| `availability.bin` | N x (horizon_days/8) bytes | bit `d` of listing `i` is 1 iff night `base_date + d` is open. Byte `j` holds nights `8j .. 8j+7`, least significant bit first. Nights outside the snapshot's calendar are 0 (closed) |
| `ranker_static.f32` | N x listing_static_dim float32 | listing-side ranker features, already normalized |

## places.jsonl

One line per place, line number = place index:
`{"idx": int, "kind": "city"|"nbhd", "city": str, "name": str, "lat": double, "lng": double, "min_lat": double, "min_lng": double, "max_lat": double, "max_lng": double, "area_km2": double}`.
The bbox of a place is the bbox of its listings. A search by `place_id` uses that bbox as the map box.

## guest_history.i64 + guest_history.f32

`guest_history.i64` holds G guest ids (sorted ascending, for binary search), `guest_history.f32`
holds G x 70 floats: `[has_history(=1), hist_log_price_mean_norm, mix_entire, mix_private, mix_shared, mix_hotel, hist_vec(64)]`.
These are query tower inputs 17..86. A guest that is absent gets 70 zeros.

## Query tower ONNX (`query_tower.onnx`)

Inputs: `place_idx` int64 [B], `qfeat` float32 [B, q_feat_dim]. Output: `qemb` float32 [B, dim].

`qfeat` layout (the service computes 0..16 itself and copies 17..86 from guest history):

| idx | feature |
|---|---|
| 0 | `ln(1 + area_km2) / 10` |
| 1 | 1 if the search is a whole city, else 0 |
| 2 | `guests / 16` |
| 3 | `ln(1 + nights) / ln(31)` |
| 4..9 | lead-time bucket one-hot, lead = checkin minus search date in days: `[<=1]`, `[2,7]`, `[8,14]`, `[15,30]`, `[31,60]`, `[>=61]` |
| 10..16 | check-in day of week one-hot, Monday first |
| 17..86 | guest history block above |

The place for a bbox-only search is the city place whose centre is nearest the bbox centre;
feature 1 is then 0 and the area is the bbox area (`km_per_deg_lat = 111.32`, longitude scaled by `cos(centre lat)`).

## Ranker ONNX (`ranker.onnx` DNN, `ranker_gbdt.onnx` LightGBM)

Input `x` float32 [B, ranker_feat_dim]; output `score` float32 [B] or [B, 1] (higher is better).
Row = `ranker_static[i]` (listing_static_dim floats) followed by 12 cross features:

| offset | feature |
|---|---|
| 0 | retrieval distance (squared L2 between qemb and listing embedding), or `-dot` under dot similarity |
| 1 | `haversine_km(listing, place centre) / sqrt(1 + area_km2)` |
| 2 | `ln(1 + price) - ln(guests)` (0 if price is 0) |
| 3 | `ln(1 + nights - min_nights)` |
| 4 | `guests / accommodates` |
| 5 | `dot(hist_vec, listing embedding)` (0 without history) |
| 6 | `(ln(1 + price) - 5) / 1 - hist_log_price_mean_norm` (0 without history or price) |
| 7 | has_history |
| 8 | `ln(1 + lead_days) / ln(366)` |
| 9 | `ln(1 + nights) / ln(31)` |
| 10 | `guests / 16` |
| 11 | history room-type mix at this listing's room type (0 without history) |

`golden/query_features.jsonl` and `golden/ranker_features.jsonl` hold requests with the exact
feature vectors Python computes; the Kotlin tests assert agreement to 1e-5.
`golden/parity_queries.jsonl` holds `{"place_idx", "qfeat":[...], "qemb":[...]}` from PyTorch for the ONNX parity test (max abs diff < 1e-4).

## Queries and updates

`queries_test.jsonl`: `{"qid", "place_idx", "checkin": "YYYY-MM-DD", "checkout", "guests", "guest_id": long|null, "search_date", "positive_id": long}`.

`updates.jsonl` (replayed in file order, already shuffled):
- `{"t":"avail","id":long,"night":int,"open":bool}` night is an offset from `base_date`
- `{"t":"price","id":long,"price":double}`
- `{"t":"remove","id":long}`
- `{"t":"new","listing":{...listings.jsonl fields...},"emb":[dim floats],"static":[listing_static_dim floats],"avail_b64":"..."}` (cluster id is assigned online: nearest centroid)
