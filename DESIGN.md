# StaySearch design

StaySearch rebuilds a published search design on public data to measure the trade-offs that design
describes. It is not a clone of any company's system, it is not affiliated with Airbnb or with
Inside Airbnb, and nothing here claims bookings, conversion or revenue.

## 1. Sources and credits

The design is taken from published work. The contribution of this repo is the rebuild on public
data and the measurements.

| Source | What StaySearch takes from it |
|---|---|
| Abdool et al., *Applying Embedding-Based Retrieval to Airbnb Search*, arXiv 2601.06873 (2026), and the companion Airbnb Tech Blog post *Embedding-Based Retrieval for Airbnb Search* | The whole first stage: a two-tower model with an offline listing tower and an online query tower kept small (place embedding, guests, area size); trip-based sampling; hard negatives; recall as the retrieval metric and replay recall@100 against the first-stage ranker's top K; IVF chosen over HNSW under filtering and real-time updates, with the cluster id applied as an ordinary search filter; the dot-product vs Euclidean clustering finding (Table 2, Figure 8); the compact precomputed availability structure for flexible dates (section 5.5.1) |
| Haldar et al., *Applying Deep Learning to Airbnb Search*, KDD'19 (arXiv 1810.09591) | GBDT then DNN ranking lineage, pairwise loss, NDCG of the chosen listing as the ranking metric |
| Haldar et al., *Improving Deep Learning for Airbnb Search*, KDD'20 | Position as a feature with dropout and zero at inference (M5), cold start framing |
| Haldar et al., *Learning To Rank Diversely at Airbnb*, CIKM'23 (arXiv 2210.07774) | The diversity re-ranking option (M6) |
| Grbovic and Cheng, *Real-time Personalization using Embeddings for Search Ranking at Airbnb*, KDD'18 | Guest history as a mean of past listing embeddings, price and room-type preference features |
| Huang et al., *Embedding-based Retrieval in Facebook Search*, KDD'20 | Hard-negative mining, the false-negative problem in mined negatives |
| Yi et al., *Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations*, RecSys'19 | In-batch sampled softmax with the logQ correction |
| Xiong et al., *Approximate Nearest Neighbor Negative Contrastive Learning* (ANCE), ICLR'21 | Model-mined negatives refreshed from the previous epoch's retriever |
| Bachrach et al., *Speeding Up the Xbox Recommender System Using a Euclidean Transformation for Inner-Product Spaces*, RecSys'14 | The inner-product to Euclidean transform, cited by 2601.06873 for the clustering finding |
| Apache Lucene 10 (`KnnFloatVectorField`, `KnnFloatVectorQuery` with a filter, `LatLonPoint`, `IndexWriter.updateDocument`, `SearcherManager`), FAISS (`IndexIVFFlat`, `IndexHNSWFlat`), ONNX Runtime | The engines both index strategies run in |
| Inside Airbnb, *Get the Data* and *Data Assumptions* (https://insideairbnb.com), CC BY 4.0 | All listing, calendar and review data; the 50% review rate and the length-of-stay assumptions |
| Airbnb Help Center, review policy: guests have 14 days after checkout to write a review | The check-out reconstruction window |

## 2. Labels and the eligible set

This section was written before any training code, and the code follows it.

### 2.1 What the data can and cannot say

Inside Airbnb publishes, per city and per quarterly snapshot, the listings (one row per listing,
with a single price quote), a forward-looking calendar (one row per listing-night: open or not,
plus min and max nights) and every public review (listing, date, reviewer id). It publishes no
searches, impressions, clicks, wishlists or bookings. A closed night is booked **or** blocked by the
host and the two cannot be told apart. Inside Airbnb itself assumes roughly half of stays leave a
review. Measured on this machine (see `results/m0_data.jsonl`): some quarterly snapshots carry no
price at all (Seattle 2025-12-28 and 2026-03-29 have no `price` column values), and snapshot sizes
swing between quarters (Seattle has 4,550 listings on 2026-03-29 and 7,769 on 2026-06-15).

### 2.2 The positive: a reviewed stay

A review `(reviewer_id, listing_id, review_date)` is read as "this guest stayed at this listing and
later reviewed it". In code, results and prose it is a **reviewed stay** (`reviewed_stay`), never a
booking. It is a proxy with heavy, non-random missingness: guests who review differ from guests who
do not, and stays that went badly may be under- or over-reviewed.

### 2.3 The reconstructed search behind each positive

The review does not say when the guest searched, for which dates, or for how many guests. Each of
these is reconstructed, with a fixed seed, and every reconstructed field carries the `recon_`
prefix in code:

- **Check-out** is uniform in `[review_date - 14, review_date - 1]`, because reviews are written
  within 14 days of check-out.
- **Nights** is `max(listing.minimum_nights, s)` capped at `maximum_nights`, where `s` is drawn from
  a shifted geometric distribution (`1 + Geometric`) whose mean is the city's average length of stay
  as Inside Airbnb models it: 5.5 nights in San Francisco, the 3-night default elsewhere. Inside
  Airbnb publishes only the mean, so the shape is our assumption.
- **Guests** is drawn from one global party-size distribution, truncated at the listing's
  `accommodates`: P(1, 2, 3, 4, 5, 6, 7, 8) = (0.22, 0.42, 0.11, 0.13, 0.05, 0.04, 0.015, 0.015).
  No public source gives this distribution; it is our assumption and it is stated as one.
- **Location** is the listing's neighbourhood (a narrow search) or the whole city (a broad search),
  50/50 by a seeded coin, with the area's geographic size (the bbox of its listings, in km2) as a
  query feature, as in 2601.06873's query tower.
- **Search date.** For a point-in-time evaluation positive, the search is placed at the snapshot
  date S that defines its eligible set (section 2.4), so lead time is `checkin - S`. For older
  training positives, which have no snapshot before them, the lead time is drawn uniform in
  `[0, 90]` days.

**Leakage, measured, not assumed away.** Truncating guests at `accommodates` and nights at the
listing's min and max nights leaks a little of the positive into the query (a query for 7 guests
rules out small homes, and the positive is never one of them by construction). Experiment exp8
reruns retrieval with guests and nights zeroed in the query tower and reports the gap. If dropping
them changes recall by more than a couple of points, the leak is real and the README says so.

### 2.4 The point-in-time eligible set

For a held-out positive whose stay nights fall after a snapshot S, the **eligible set** is every
listing in S that lies in the search area (city, or neighbourhood bbox) and that, in S's calendar:
is open for every night of the stay; has `minimum_nights <= nights <= maximum_nights`; and has
`accommodates >= guests`. The positive must itself be in S and eligible. If it is not, the
positive is **dropped**, and dropped positives are counted per city and per reason (not in S, not
open in S's calendar, rules) in `results/m0_data.jsonl`.

This is the step that makes the evaluation a marketplace evaluation: dates genuinely constrain
results, and a model is scored only against homes the guest could actually have booked at that
moment. It also has a sharp consequence worth saying plainly: a guest who booked before S has
already closed those nights in S's calendar, so only stays that were still open at S survive. The
survivors are the stays a guest booked after S, which is what a search at time S would have to
find.

### 2.5 The time split

Per city, with the four quarterly snapshots S1 (about 2025-09) < S2 (2025-12) < S3 (2026-03) < S4
(2026-06):

| Split | Stays | Listing features from | Eligible set | Used for |
|---|---|---|---|---|
| tower-train | review before S1 (2023-09-01 onward) | S1 | none (availability unknown) | two-tower training |
| rank-train | check-in after S1, review before S2 | S1 | S1 calendar | ranker training (the towers never see these) |
| val | check-in after S2, check-out before S3 | S2 | S2 calendar | model selection |
| test | check-in after S3, review up to S4 | S3 | S3 calendar | every reported number |

Nothing from the test window reaches training. Engagement features (`number_of_reviews_ltm`,
review scores, occupancy) are read from the snapshot in the table, so they are as of that date.
Price is carried forward: a listing's quote is taken from the latest snapshot at or before the
split's snapshot that has a quote, and is missing otherwise (with a missing indicator). The towers
and the ranker are trained on disjoint stays so that the ranker learns from retrieval scores of
stays the towers never saw.

### 2.6 Trip grouping

A guest's reviewed stays in one city whose review dates fall within 30 days of each other form one
**trip**. The trip, not the single review, is the training unit, as in 2601.06873's trip-based
sampling: each epoch draws one positive per trip, and the trip's other listings are never used as
negatives for it. Most trips have one review; the share is in `results/m0_data.jsonl`.

### 2.7 Negatives, substituted

The paper samples negatives from listings the guest was shown, viewed or wishlisted and did not
book. None of that is public. StaySearch substitutes four samplers, each its own experiment arm
(exp2). **This is a substitution for behavioural negatives, not a reproduction of them.**

1. **Random**: uniform from the positive's city (the "too easy" baseline the paper describes).
2. **Geo-hard**: same S2 level-14 cell, else same neighbourhood; same capacity band; open the same
   nights where a calendar exists.
3. **Price-hard**: same neighbourhood, price quote within 20% of the positive's.
4. **Model-mined**: the top-ranked non-positive listings from the previous epoch's retriever over
   the positive's city (ANCE-style). False-negative guard: a listing the same guest reviewed at any
   time is never used as that guest's negative.

Every arm also uses in-batch negatives with the logQ correction. Batches are drawn within a city,
so in-batch negatives are plausible alternatives rather than homes on another continent.

### 2.8 Guest history

For each positive, the guest's history is their earlier reviewed stays, strictly before the
search date, in any loaded city. Features: whether any exist, the mean of those listings' tower
embeddings (stop-gradient), the mean normalized log price, and the room-type mix. Most guests have
no history; every metric is reported on the full set and on the with-history slice.

## 3. Retrieval: the two towers

**Listing tower (offline, the "daily batch").** Inputs: S2 cell embeddings at levels 10, 12 and 14
(16 dims each), room type and property type embeddings, 27 standardized numeric fields (capacity,
bedrooms, bathrooms, log price quote, min and max nights, availability fractions, review counts and
scores, estimated occupancy, host tenure, listing age, superhost), three missing-value flags, and a
200-amenity multi-hot. MLP 256-128-64. All engagement fields are read from the snapshot the split
uses, so they are as of that date. `staysearch.towers.embed_listings` is the batch job; its output
is what the service loads.

**Query tower (online).** A 32-dim place embedding (each city and each neighbourhood is a place)
plus 87 features: log area, city-or-neighbourhood flag, guests, log nights, lead-time bucket,
check-in day of week, and the guest-history block (has history, mean normalized log price,
room-type mix, mean of the history listings' embeddings with stop-gradient). MLP 128-128-64.
Kept small on purpose, as in 2601.06873 section 4.5: it is the only model on the request path
before the index.

**Loss.** In-batch sampled softmax with the logQ correction of Yi et al. (subtract the log of each
listing's share of positives in its city, since batches are drawn within a city), plus explicit
hard negatives from the sampler under test, masked by the false-negative guard. Logits are a learned
scale times the dot product, or times the negative squared L2 distance for the Euclidean variant.
Ten epochs of up to 600,000 trips each; the epoch with the best validation recall@100 is kept.

**Variants trained (exp2, exp3, exp8):** in-batch only, random, geo-hard, price-hard, model-mined,
and the mix (one of each), all Euclidean; the mix with dot product; the mix without guest history;
the mix with guests and nights zeroed in the query tower (the leakage check).

## 4. The index: IVF as a filter, HNSW on the same documents

2601.06873 chose IVF over HNSW because, with filters and about 10,000 real-time updates per
second, HNSW's memory grew too large and updates were hard; IVF stores only centroids and a
cluster id per listing, and the cluster id is applied as an ordinary filter in the search engine.
StaySearch builds both strategies in one Apache Lucene 10 index over the same documents:

- **IVF path.** FAISS k-means (128 clusters, swept 64 and 256) over the listing embeddings; each
  document stores its cluster id as an `IntField`. A query takes the `nprobe` nearest centroids and
  runs one Lucene `BooleanQuery` of filters: `cluster_id IN (...)`, a `LatLonPoint` box, capacity,
  min and max nights, and the availability query (section 5). The vectors live outside Lucene in a
  float array keyed by row; the service computes exact distances on the filtered candidates and
  keeps the top K.
- **HNSW path.** The same documents also carry a `KnnFloatVectorField` (Euclidean), queried with
  `KnnFloatVectorQuery(field, q, k, filter)` using the same filter minus the cluster clause.
  Lucene pre-filters inside graph traversal, so there is no post-filter recall loss to correct.
- **Exact path.** All filtered documents, exact distance: the ground truth for recall vs exact.

**Updates.** A calendar flip changes the in-memory availability store in place and is visible to
the next query with no index work. For the index-placement experiment (exp4), every event also
rewrites the listing's Lucene document with `IndexWriter.updateDocument`, which deletes and re-adds
it; in HNSW mode the re-added document carries its vector, so the graph segment must index it and
later merges must rebuild graphs. A `SearcherManager` refreshes every second. Update-to-visible lag
is measured with sequence numbers captured at refresh. New listings get their tower embedding from
the batch and a cluster id from the nearest centroid at ingest (the cold-start path).

**The clustering finding (exp3).** The paper found dot-product clustering imbalanced (one or two
clusters holding about 20% of listings) and Euclidean balanced, with far better recall at small
nprobe (Table 2). StaySearch trains a dot-product and a Euclidean tower, clusters each with the
matching metric, and reports the size distribution and recall@100 vs exact at nprobe 1 to 64. It
also tries the inner-product-to-Euclidean transform of Bachrach et al. on the dot model, which the
paper cites as not fixing the imbalance.

## 5. Availability: an O(1) check that updates in place

Per listing, a bitset of 448 nights from the bundle's base date (7 x 64 bits) and a run-length
array, `run[d]` = consecutive open nights starting at night `d`, stored as bytes capped at 255 (a
stored 255 means "keep checking at d + 255", so a check is at most two lookups over 448 nights). A
stay `[a, b)` is open iff `run[a] >= b - a`. A flip at night `d` updates the bit and recomputes
`run` only for nights at or before `d`, stopping as soon as a value is unchanged. One writer and
many readers share it through a per-listing seqlock. Lucene sees it through a custom two-phase
`AvailabilityQuery` that reads each document's row and asks the store, so availability never
needs an index write. Flexible dates (plus or minus 3 days on each end, 49 windows) are 49 lookups
into the same arrays, returned as a 49-bit mask. This is the compact structure of 2601.06873
section 5.5.1, rebuilt at our scale.

The **hard invariant**: every returned listing is open for every requested night in the store's
current state (checked against the bitset night by night, independently of the run array), meets
min and max nights, fits the guests, and lies inside the box. It is a JUnit test, a Kotest property
test over random places, boxes, dates, guests and modes, and a CI replay over recorded queries;
one violation fails the build.

## 6. Ranking

Rankers re-rank the two-tower's top K = 100 from the eligible set. Inputs are the listing's static
features (the tower's numeric fields, missing flags, room type, amenity count) and 12 query-listing
cross features: retrieval distance, distance to the area centre scaled by its size, log price per
guest, nights beyond the minimum, guests over capacity, lead time, nights, guests, and four
guest-history features (embedding affinity, price gap to the guest's usual, room-type match,
has-history). The Python and Kotlin implementations of these are held together by golden files.

The lineage follows Haldar et al. (KDD'19): LightGBM LambdaMART first, then a pairwise DNN
(logistic loss of the reviewed stay against each other candidate), then the DNN with the history
features. Rankers train on the rank_train split, select on validation, and report on test.
**Training lists keep only queries where retrieval found the reviewed stay** (BUG_LOG #6).

**Replay recall (exp7).** 2601.06873 section 5.2 scores retrieval by the recall of the first-stage
ranker's top results. StaySearch runs its ranker over each test query's full eligible set, takes
its top 100 as ground truth, and reports what fraction each retrieval method's top 100 recovers.

## 7. Position bias (M5, simulated)

There are no impressions in public data, so position bias can only be studied on simulated logs,
and every such number is labelled SIMULATED. Real candidate lists are shown by a logging policy
that mixes the trained ranker's score with popularity and noise; simulated guests examine position
`p` with probability `p^-eta` and choose among examined listings in proportion to
`exp(utility / tau)`. A DNN trained with `log(1 + position)` as a feature (dropped out at 15%, set
to 0 at inference, as in Haldar et al. KDD'20) is compared with one trained without it, by Kendall
tau and NDCG against the true utility, at three bias strengths.

## 8. Extensions (M6)

- **Diversity (M6a).** Greedy re-ranking of the ranker's top 10 after *Learning To Rank Diversely*:
  each slot takes the candidate with the best ranker score minus lambda times its maximum
  similarity (location and price) to listings already placed. The result is a curve of NDCG of the
  reviewed stay against intra-list diversity, not a single winner.
- **Flexible dates (M6b).** Eligible-set growth and latency of one 49-window search against 49
  separate searches, in the service.
- **Map-and-list demo (M6c).** A static page served by the Kotlin service: pick a city or drag the
  map, set dates and guests, and see ranked results, per-stage latency, and the invariant status.

## 9. What is not here, on purpose

No LLM anywhere. No text search or BM25: the query is structured (place, dates, guests) and listing
text is never matched at query time. No interleaving or IPS counterfactual learning to rank. No
sharded or replicated serving tier and no broker: the update stream is an in-process replay from a
file. No claim about bookings, conversion or revenue.

