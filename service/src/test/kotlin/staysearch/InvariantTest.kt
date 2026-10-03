@file:OptIn(io.kotest.common.ExperimentalKotest::class)

package staysearch

import io.kotest.property.Arb
import io.kotest.property.PropTestConfig
import io.kotest.property.arbitrary.boolean
import io.kotest.property.arbitrary.double
import io.kotest.property.arbitrary.element
import io.kotest.property.arbitrary.int
import io.kotest.property.arbitrary.orNull
import io.kotest.property.checkAll
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.AfterAll
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.BeforeAll
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.TestInstance
import staysearch.bundle.Box
import staysearch.cli.loadQueries
import staysearch.search.SearchRequest
import staysearch.updates.UpdateSource

/** The hard invariant on the synthetic bundle, in all three modes, with updates applied between queries. */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class InvariantTest {
    private lateinit var engine: Engine

    @BeforeAll
    fun open() {
        val b = FakeBundle.require()
        engine = Engine.open(b.dir, EngineConfig(refreshMs = 0), b)
    }

    @AfterAll
    fun close() {
        if (::engine.isInitialized) engine.close()
    }

    @Test
    fun `no violations over test queries in every mode while updates apply`() {
        val queries = loadQueries(engine.bundle, 150)
        UpdateSource(engine.bundle.updatesPath, loop = true).use { src ->
            var nonEmpty = 0
            var total = 0
            for ((i, q) in queries.withIndex()) {
                repeat(20) { src.next()?.let(engine.applier::apply) }
                if (i % 25 == 0) engine.index.refreshNow()
                for (mode in listOf("ivf", "hnsw", "exact")) for (flex in listOf(false, true)) {
                    val req = q.copy(mode = mode, flexible = flex, t = 50)
                    val resp = engine.pipeline.search(req)
                    val v = engine.invariants.check(req, resp)
                    assertTrue(v.isEmpty()) { "violations for $req: $v" }
                    total++
                    if (resp.results.isNotEmpty()) nonEmpty++
                }
            }
            // Guard against a vacuous pass: most test queries have eligible listings.
            assertTrue(nonEmpty > total * 0.8, "only $nonEmpty of $total searches returned results")
        }
        assertTrue(engine.applier.applied.get() > 0)
    }

    @Test
    fun `exact mode returns the full eligible set when k is large`() {
        val q = loadQueries(engine.bundle, 20).first().copy(mode = "exact", k = 100_000, t = 100_000, ranker = "none")
        val resp = engine.pipeline.search(q)
        // Brute force over the table: every live listing that satisfies the request.
        val ctx = staysearch.search.Features.resolve(q, engine.bundle)
        val t = engine.table
        val expected = (0 until t.size).filter { r ->
            !t.isRemoved(r) && ctx.box.contains(t.lat[r], t.lng[r]) && t.accommodates[r] >= ctx.guests &&
                ctx.nights in t.minNights[r]..t.maxNights[r] && t.store.allNightsOpenByBits(r, ctx.checkinNight, ctx.checkoutNight)
        }.map { t.ids[it] }.toSet()
        engine.index.refreshNow()
        val after = engine.pipeline.search(q)
        assertEquals(expected, after.results.map { it.id }.toSet())
        assertTrue(resp.results.isNotEmpty())
    }

    @Test
    fun `property - random requests never violate the invariant`() = runBlocking {
        val b = engine.bundle
        val h = b.manifest.horizonDays
        checkAll(
            PropTestConfig(iterations = 300),
            Arb.int(0 until b.places.size),
            Arb.int(-5..h + 5),
            Arb.int(1..20),
            Arb.int(1..10),
            Arb.element("ivf", "hnsw", "exact"),
            Arb.boolean(),
            Arb.double(0.005..0.2, includeNaNs = false).orNull(0.6),
        ) { place, checkinOffset, nights, guests, mode, flexible, boxHalf ->
            val p = b.places[place]
            val checkin = b.manifest.baseDate.plusDays(checkinOffset.toLong())
            val bbox = boxHalf?.let { Box(p.lat - boxHalf, p.lng - boxHalf, p.lat + boxHalf, p.lng + boxHalf) }
            val req = SearchRequest(
                placeIdx = if (bbox == null) place else null, bbox = bbox, checkin = checkin,
                checkout = checkin.plusDays(nights.toLong()), guests = guests, mode = mode, flexible = flexible,
                k = 200, t = 40, nprobe = 8,
            )
            val resp = engine.pipeline.search(req)
            val v = engine.invariants.check(req, resp)
            assertTrue(v.isEmpty()) { "violations for $req: $v" }
        }
        Unit
    }
}
