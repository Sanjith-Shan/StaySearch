package staysearch

import org.apache.lucene.search.KnnFloatVectorQuery
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import staysearch.bundle.ListingRecord
import staysearch.bundle.Similarity
import staysearch.data.ListingTable
import staysearch.index.AvailabilityQuery
import staysearch.index.ListingIndex
import java.nio.file.Files
import kotlin.random.Random

/** Lucene's HNSW search pre-filtered by the availability query, with no bundle involved. */
class HnswFilterTest {
    private val dim = 16
    private val n = 3000
    private val horizon = 64

    private fun table(): ListingTable {
        val rnd = Random(5)
        val t = ListingTable(dim, 2, horizon, n, Similarity.L2, FloatArray(dim * 4) { rnd.nextFloat() })
        for (i in 0 until n) {
            // Listing i is open for nights [10, 20) iff i is even; every listing is open elsewhere.
            val avail = ByteArray(horizon / 8) { -1 }
            if (i % 2 == 1) avail[1] = 0 // nights 8..15 closed
            t.upsert(
                ListingRecord(1000L + i, "c", 0, 0, 10.0, 10.0, 4, 1, 30, 100.0, 0),
                FloatArray(dim) { rnd.nextFloat() }, FloatArray(2), avail, clusterId = 0,
            )
        }
        return t
    }

    @Test
    fun `knn with availability pre-filter returns k open listings and sees store changes without refresh`() {
        val t = table()
        val dir = Files.createTempDirectory("hnsw-filter-test")
        val index = ListingIndex(t, dir, withVectors = true, refreshMs = 0)
        try {
            index.buildAll()
            val q = FloatArray(dim) { 0.5f }
            val k = 20
            fun search(): List<Int> = index.withSearcher { s ->
                val docs = s.search(KnnFloatVectorQuery(ListingIndex.F_VEC, q, k, AvailabilityQuery(t, 10, 20)), k).scoreDocs
                val leaves = s.indexReader.leaves()
                docs.map { sd ->
                    val leaf = leaves[org.apache.lucene.index.ReaderUtil.subIndex(sd.doc, leaves)]
                    val dv = org.apache.lucene.index.DocValues.getNumeric(leaf.reader(), ListingIndex.F_ROW)
                    check(dv.advanceExact(sd.doc - leaf.docBase)); dv.longValue().toInt()
                }
            }
            val rows = search()
            assertEquals(k, rows.size)
            assertTrue(rows.all { it % 2 == 0 }, "every result must be open for the stay: $rows")
            // Recall against brute force over the open listings.
            val exact = (0 until n).filter { it % 2 == 0 }.sortedBy { t.distance(q, it) }.take(k).toSet()
            assertTrue(rows.count { it in exact } >= k * 0.8, "recall too low: ${rows.count { it in exact }} of $k")

            // Close a night for the best result: it must disappear at once, with no index refresh.
            val best = rows.first()
            t.store.setNight(best, 15, false)
            assertFalse(best in search())
            t.store.setNight(best, 15, true)
            assertTrue(best in search())
        } finally {
            index.closeAndDelete()
        }
    }
}
