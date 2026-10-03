package staysearch.search

import org.HdrHistogram.ConcurrentHistogram
import org.apache.lucene.document.IntField
import org.apache.lucene.document.LatLonPoint
import org.apache.lucene.index.DocValues
import org.apache.lucene.index.LeafReaderContext
import org.apache.lucene.index.NumericDocValues
import org.apache.lucene.index.ReaderUtil
import org.apache.lucene.search.BooleanClause
import org.apache.lucene.search.BooleanQuery
import org.apache.lucene.search.CollectorManager
import org.apache.lucene.search.IndexSearcher
import org.apache.lucene.search.KnnFloatVectorQuery
import org.apache.lucene.search.Query
import org.apache.lucene.search.ScoreMode
import org.apache.lucene.search.SimpleCollector
import staysearch.bundle.Bundle
import staysearch.data.ListingTable
import staysearch.index.AvailabilityQuery
import staysearch.index.ListingIndex

enum class Mode { IVF, HNSW, EXACT;

    companion object {
        fun parse(s: String) = when (s.lowercase()) {
            "ivf" -> IVF
            "hnsw" -> HNSW
            "exact" -> EXACT
            else -> throw IllegalArgumentException("mode must be ivf, hnsw or exact, got '$s'")
        }
    }
}

/** Per-stage latency, microseconds. */
data class StageTimings(
    val queryTower: Long,
    val probe: Long,
    val filter: Long,
    val distance: Long,
    val rank: Long,
    val total: Long,
)

data class SearchResult(
    val id: Long,
    val lat: Double,
    val lng: Double,
    val price: Double,
    val accommodates: Int,
    val minNights: Int,
    val maxNights: Int,
    val roomType: Int,
    val distance: Float,
    val score: Float?,
)

data class SearchResponse(
    val mode: String,
    val ranker: String,
    val placeIdx: Int,
    val nights: Int,
    val candidates: Int,
    val retrieved: Int,
    val timingsUs: StageTimings,
    val results: List<SearchResult>,
)

/** Retrieval output: rows ordered by distance (ascending), plus how many docs passed the filter. */
class Retrieved(val rows: IntArray, val dist: FloatArray, val nCandidates: Int)

/** HdrHistograms (microseconds) for every stage, shared by all requests. */
class StageStats {
    val stages = linkedMapOf(
        "query_tower" to hist(), "probe" to hist(), "filter" to hist(),
        "distance" to hist(), "rank" to hist(), "total" to hist(),
    )

    private fun hist() = ConcurrentHistogram(3)

    fun record(t: StageTimings) {
        stages["query_tower"]!!.recordValue(t.queryTower); stages["probe"]!!.recordValue(t.probe)
        stages["filter"]!!.recordValue(t.filter); stages["distance"]!!.recordValue(t.distance)
        stages["rank"]!!.recordValue(t.rank); stages["total"]!!.recordValue(t.total)
    }

    fun summary(): Map<String, Map<String, Any>> = stages.mapValues { (_, h) -> summarize(h) }

    fun reset() = stages.values.forEach { it.reset() }

    companion object {
        fun summarize(h: org.HdrHistogram.AbstractHistogram): Map<String, Any> = mapOf(
            "count" to h.totalCount,
            "p50_us" to h.getValueAtPercentile(50.0),
            "p90_us" to h.getValueAtPercentile(90.0),
            "p99_us" to h.getValueAtPercentile(99.0),
            "p999_us" to h.getValueAtPercentile(99.9),
            "max_us" to h.maxValue,
            "mean_us" to h.mean,
        )
    }
}

/**
 * The online path: query features and tower, IVF probe, Lucene filter (with the availability
 * query), exact distance to top K, ranker over K to top T.
 *
 * Modes:
 * - ivf: the nprobe nearest centroids become a `cluster IN (...)` filter clause; exact distance over
 *   every doc that passes the filter.
 * - hnsw: KnnFloatVectorQuery(vec, qemb, k, filter) over the same filter minus the cluster clause
 *   (Lucene pre-filters; the "filter" stage is the whole KNN query). Distances are then recomputed
 *   from the table's vectors.
 * - exact: every doc that passes the filter (no cluster clause), exact distance. Ground truth for
 *   recall vs exact.
 *
 * Lucene's LatLonPoint box is quantized (about 1e-7 degrees), so a listing a hair outside the box
 * can pass it. Every candidate is re-checked against the exact box in doubles before distances.
 */
class SearchPipeline(
    val bundle: Bundle,
    val table: ListingTable,
    val index: ListingIndex,
    val models: Models,
) {
    val stats = StageStats()

    fun search(req: SearchRequest, record: Boolean = true): SearchResponse {
        val t0 = System.nanoTime()
        val mode = Mode.parse(req.mode)
        require(req.k in 1..100_000) { "k out of range" }
        require(req.t >= 0) { "t must be >= 0" }
        if (req.ranker != "none") require(req.ranker in models.rankerNames) { "ranker '${req.ranker}' not available (have ${models.rankerNames} and none)" }
        val ctx = Features.resolve(req, bundle)
        val qemb = models.embedQuery(ctx.placeIdx, Features.queryFeatures(ctx, bundle.manifest.qFeatDim))
        val t1 = System.nanoTime()
        val timing = LongArray(3) // probe, filter, distance (ns)
        val r = retrieve(ctx, qemb, mode, req.k, req.nprobe, req.flexible, timing)
        val t4 = System.nanoTime()

        val order: IntArray
        val scores: FloatArray?
        if (req.ranker == "none" || r.rows.isEmpty()) {
            order = IntArray(minOf(req.t, r.rows.size)) { it }
            scores = null
        } else {
            val w = models.rankerWidth
            val n = r.rows.size
            val x = FloatArray(n * w)
            for (i in 0 until n) Features.rankerRow(ctx, table, r.rows[i], Features.distanceDouble(qemb, table, r.rows[i]).toFloat(), x, i * w)
            scores = models.score(req.ranker, x, n)
            order = (0 until n).sortedByDescending { scores[it] }.take(req.t).toIntArray()
        }
        val t5 = System.nanoTime()

        val results = order.map { i ->
            val row = r.rows[i]
            SearchResult(
                id = table.ids[row], lat = table.lat[row], lng = table.lng[row], price = table.price[row],
                accommodates = table.accommodates[row], minNights = table.minNights[row], maxNights = table.maxNights[row],
                roomType = table.roomType[row], distance = r.dist[i], score = scores?.get(i),
            )
        }
        val timings = StageTimings(
            queryTower = (t1 - t0) / 1000, probe = timing[0] / 1000, filter = timing[1] / 1000,
            distance = timing[2] / 1000, rank = (t5 - t4) / 1000, total = (System.nanoTime() - t0) / 1000,
        )
        if (record) stats.record(timings)
        return SearchResponse(mode.name.lowercase(), req.ranker, ctx.placeIdx, ctx.nights, r.nCandidates, r.rows.size, timings, results)
    }

    /** The Lucene filter for a request: box, guests, nights (unless flexible) and availability, plus optional clusters. */
    fun filterQuery(ctx: QueryContext, clusters: IntArray?, flexible: Boolean): Query {
        val b = BooleanQuery.Builder()
        fun f(q: Query) = b.add(q, BooleanClause.Occur.FILTER)
        if (clusters != null) f(IntField.newSetQuery(ListingIndex.F_CLUSTER, *clusters))
        f(LatLonPoint.newBoxQuery(ListingIndex.F_GEO, ctx.box.minLat, ctx.box.maxLat, ctx.box.minLng, ctx.box.maxLng))
        f(IntField.newRangeQuery(ListingIndex.F_ACC, ctx.guests, Int.MAX_VALUE))
        if (!flexible) {
            f(IntField.newRangeQuery(ListingIndex.F_MIN_N, Int.MIN_VALUE, ctx.nights))
            f(IntField.newRangeQuery(ListingIndex.F_MAX_N, ctx.nights, Int.MAX_VALUE))
        }
        f(AvailabilityQuery(table, ctx.checkinNight, ctx.checkoutNight, if (flexible) 3 else 0))
        return b.build()
    }

    /** Stages 2 to 4. [timingNs] receives probe, filter and distance nanoseconds. */
    fun retrieve(ctx: QueryContext, qemb: FloatArray, mode: Mode, k: Int, nprobe: Int, flexible: Boolean, timingNs: LongArray = LongArray(3)): Retrieved {
        val t0 = System.nanoTime()
        val clusters = if (mode == Mode.IVF) table.nearestCentroids(qemb, 0, nprobe) else null
        val t1 = System.nanoTime()
        val filter = filterQuery(ctx, clusters, flexible)
        val rows: IntArray = index.withSearcher { s ->
            if (mode == Mode.HNSW) {
                require(index.withVectors) { "hnsw mode needs an index built with vectors" }
                knnRows(s, KnnFloatVectorQuery(ListingIndex.F_VEC, qemb, k, filter), k)
            } else s.search(filter, RowCollectorManager)
        }
        val t2 = System.nanoTime()
        val inBox = rows.filter { ctx.box.contains(table.lat[it], table.lng[it]) }.toIntArray()
        val top = topK(qemb, inBox, k)
        val t3 = System.nanoTime()
        timingNs[0] = t1 - t0; timingNs[1] = t2 - t1; timingNs[2] = t3 - t2
        return Retrieved(top.first, top.second, rows.size)
    }

    private fun knnRows(s: IndexSearcher, q: KnnFloatVectorQuery, k: Int): IntArray {
        val docs = s.search(q, k).scoreDocs.map { it.doc }.sorted()
        val leaves = s.indexReader.leaves()
        var leaf: LeafReaderContext? = null
        var dv: NumericDocValues? = null
        val out = IntArray(docs.size)
        var n = 0
        for (doc in docs) {
            val li = ReaderUtil.subIndex(doc, leaves)
            if (leaf !== leaves[li]) {
                leaf = leaves[li]; dv = DocValues.getNumeric(leaf.reader(), ListingIndex.F_ROW)
            }
            if (dv!!.advanceExact(doc - leaf!!.docBase)) out[n++] = dv.longValue().toInt()
        }
        return out.copyOf(n)
    }

    /** Exact top [k] by distance (ascending) with a bounded max-heap. */
    fun topK(q: FloatArray, rows: IntArray, k: Int): Pair<IntArray, FloatArray> {
        val cap = minOf(k, rows.size)
        val hr = IntArray(cap)
        val hd = FloatArray(cap)
        var n = 0
        fun siftDown(start: Int) {
            var i = start
            while (true) {
                val l = 2 * i + 1
                if (l >= n) return
                val c = if (l + 1 < n && hd[l + 1] > hd[l]) l + 1 else l
                if (hd[c] <= hd[i]) return
                val tr = hr[i]; hr[i] = hr[c]; hr[c] = tr
                val td = hd[i]; hd[i] = hd[c]; hd[c] = td
                i = c
            }
        }
        for (row in rows) {
            val d = table.distance(q, row)
            if (n < cap) {
                hr[n] = row; hd[n] = d
                var i = n++
                while (i > 0) {
                    val p = (i - 1) / 2
                    if (hd[p] >= hd[i]) break
                    val tr = hr[i]; hr[i] = hr[p]; hr[p] = tr
                    val td = hd[i]; hd[i] = hd[p]; hd[p] = td
                    i = p
                }
            } else if (cap > 0 && (d < hd[0] || (d == hd[0] && row < hr[0]))) {
                hr[0] = row; hd[0] = d; siftDown(0)
            }
        }
        val idx = (0 until n).sortedWith(compareBy<Int>({ hd[it] }, { hr[it] }))
        return IntArray(n) { hr[idx[it]] } to FloatArray(n) { hd[idx[it]] }
    }
}

/** Collects the `row` doc value of every matching doc. */
private class RowCollector : SimpleCollector() {
    var rows = IntArray(256)
    var n = 0
    private var dv: NumericDocValues? = null

    override fun doSetNextReader(context: LeafReaderContext) {
        dv = DocValues.getNumeric(context.reader(), ListingIndex.F_ROW)
    }

    override fun collect(doc: Int) {
        val v = dv!!
        if (!v.advanceExact(doc)) return
        if (n == rows.size) rows = rows.copyOf(n * 2)
        rows[n++] = v.longValue().toInt()
    }

    override fun scoreMode() = ScoreMode.COMPLETE_NO_SCORES
}

private object RowCollectorManager : CollectorManager<RowCollector, IntArray> {
    override fun newCollector() = RowCollector()
    override fun reduce(collectors: Collection<RowCollector>): IntArray {
        val out = IntArray(collectors.sumOf { it.n })
        var o = 0
        for (c in collectors) {
            System.arraycopy(c.rows, 0, out, o, c.n); o += c.n
        }
        return out
    }
}
