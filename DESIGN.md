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

<!-- Sections 3+ (system, index, ranking, experiments) are filled in as each milestone lands. -->
