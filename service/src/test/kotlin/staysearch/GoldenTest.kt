package staysearch

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Test
import staysearch.data.ListingTable
import staysearch.search.Features
import staysearch.search.Models
import staysearch.search.SearchRequest
import staysearch.util.Json
import kotlin.math.abs

/**
 * Feature golden files and ONNX parity, read from the bundle's golden/ directory (each test skips
 * when its file is absent). Features must agree within max(1e-5, 1e-6 x |value|).
 */
class GoldenTest {
    private fun close(expected: Double, actual: Double) = abs(expected - actual) <= maxOf(1e-5, 1e-6 * abs(expected))

    @Test
    fun `query features match the Python reference`() {
        val b = FakeBundle.require()
        val path = FakeBundle.golden("query_features.jsonl")
        assumeTrue(path != null, "no golden/query_features.jsonl")
        var n = 0
        for (row in Json.readLines(path!!)) {
            val req = Json.mapper.treeToValue(row["request"], SearchRequest::class.java)
            val ctx = Features.resolve(req, b)
            assertEquals(row["place_idx"].asInt(), ctx.placeIdx, "place for $req")
            val q = Features.queryFeatures(ctx, b.manifest.qFeatDim)
            val g = row["qfeat"]
            assertEquals(g.size(), q.size)
            for (i in q.indices) assertTrue(close(g[i].asDouble(), q[i].toDouble())) { "qfeat[$i] for $req: expected ${g[i]}, got ${q[i]}" }
            n++
        }
        assertTrue(n > 0)
    }

    @Test
    fun `ranker features match the Python reference`() {
        val b = FakeBundle.require()
        val path = FakeBundle.golden("ranker_features.jsonl")
        assumeTrue(path != null, "no golden/ranker_features.jsonl")
        val table = ListingTable.fromBundle(b, 0)
        val w = b.manifest.rankerFeatDim
        var n = 0
        for (row in Json.readLines(path!!)) {
            val req = Json.mapper.treeToValue(row["request"], SearchRequest::class.java)
            val ctx = Features.resolve(req, b)
            val qemb = FloatArray(b.manifest.dim) { row["qemb"][it].floatValue() }
            val r = requireNotNull(table.rowOf(row["listing_id"].asLong()))
            // rank (feature 14) depends on the candidate list, so the golden row carries it; gaps
            // (12, 13) come from the bundle's availability as loaded (no updates applied).
            val rank = requireNotNull(row["rank"]?.takeIf { it.isInt }) { "golden ranker row without an int rank: $row" }.asInt()
            val x = FloatArray(w)
            Features.rankerRow(ctx, table, r, Features.distanceDouble(qemb, table, r).toFloat(), rank, b.manifest.rankerRankK, x, 0)
            val g = row["x"]
            assertEquals(b.manifest.listingStaticDim + Features.N_CROSS, w, "ranker_feat_dim")
            assertEquals(w, g.size(), "golden x width")
            for (i in 0 until w) assertTrue(close(g[i].asDouble(), x[i].toDouble())) { "x[$i] for listing ${row["listing_id"]}, $req: expected ${g[i]}, got ${x[i]}" }
            n++
        }
        assertTrue(n > 0)
    }

    @Test
    fun `query tower in ONNX Runtime Java matches PyTorch within 1e-4`() {
        val b = FakeBundle.require()
        val path = FakeBundle.golden("parity_queries.jsonl")
        assumeTrue(path != null, "no golden/parity_queries.jsonl")
        val d = b.manifest.dim
        var maxDiff = 0.0
        Models(b).use { m ->
            for (row in Json.readLines(path!!)) {
                val qf = FloatArray(b.manifest.qFeatDim) { row["qfeat"][it].floatValue() }
                val out = m.embedQuery(row["place_idx"].asInt(), qf)
                for (j in 0 until d) maxDiff = maxOf(maxDiff, abs(out[j] - row["qemb"][j].asDouble()))
            }
        }
        assertTrue(maxDiff < 1e-4, "max abs diff $maxDiff")
    }
}
