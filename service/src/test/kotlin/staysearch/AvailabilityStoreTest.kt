@file:OptIn(io.kotest.common.ExperimentalKotest::class)

package staysearch

import io.kotest.property.Arb
import io.kotest.property.PropTestConfig
import io.kotest.property.arbitrary.bind
import io.kotest.property.arbitrary.boolean
import io.kotest.property.arbitrary.int
import io.kotest.property.arbitrary.list
import io.kotest.property.checkAll
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import staysearch.avail.AvailabilityStore
import staysearch.avail.PerNightScan
import kotlin.random.Random

class AvailabilityStoreTest {
    private val horizon = 448

    /** Brute-force reference: a plain boolean per night. */
    private class Reference(val open: BooleanArray) {
        fun run(d: Int): Int {
            var n = 0
            while (d + n < open.size && open[d + n]) n++
            return n
        }

        fun available(a: Int, b: Int) = a >= 0 && b <= open.size && b > a && (a until b).all { open[it] }
    }

    private fun pack(open: BooleanArray): ByteArray {
        val out = ByteArray(open.size / 8)
        for (d in open.indices) if (open[d]) out[d ushr 3] = (out[d ushr 3].toInt() or (1 shl (d and 7))).toByte()
        return out
    }

    /** Calendar from alternating run lengths, starting open or closed. */
    private fun calendar(runs: List<Int>, startOpen: Boolean): BooleanArray {
        val open = BooleanArray(horizon)
        var d = 0
        var o = startOpen
        for (r in runs) {
            for (i in d until minOf(horizon, d + r)) open[i] = o
            d += r; o = !o
            if (d >= horizon) break
        }
        return open
    }

    private fun assertRunsMatch(store: AvailabilityStore, ref: Reference, row: Int) {
        for (d in 0 until horizon) {
            assertEquals(minOf(AvailabilityStore.RUN_CAP, ref.run(d)), store.runValue(row, d), "run[$d]")
            assertEquals(ref.open[d], store.isOpen(row, d), "bit $d")
        }
    }

    @Test
    fun `run lengths match brute force under random flips`() {
        val rnd = Random(1)
        val store = AvailabilityStore(horizon, 4)
        val ref = Array(4) { Reference(BooleanArray(horizon) { rnd.nextDouble() < 0.7 }) }
        for (r in 0 until 4) store.load(r, pack(ref[r].open))
        repeat(5000) {
            val row = rnd.nextInt(4)
            val night = rnd.nextInt(horizon)
            val open = rnd.nextBoolean()
            val flipped = store.setNight(row, night, open)
            assertEquals(ref[row].open[night] != open, flipped)
            ref[row].open[night] = open
            if (it % 50 == 0) assertRunsMatch(store, ref[row], row)
            repeat(20) {
                val a = rnd.nextInt(-2, horizon + 2)
                val b = a + rnd.nextInt(-1, 40)
                assertEquals(ref[row].available(a, b), store.isAvailable(row, a, b), "window [$a,$b)")
            }
        }
        for (r in 0 until 4) assertRunsMatch(store, ref[r], r)
    }

    @Test
    fun `stays longer than the run cap are exact`() {
        val store = AvailabilityStore(horizon, 1)
        val open = BooleanArray(horizon) { it in 10 until 400 }
        store.load(0, pack(open))
        assertEquals(255, store.runValue(0, 10))
        assertTrue(store.isAvailable(0, 10, 400))
        assertFalse(store.isAvailable(0, 10, 401))
        assertTrue(store.isAvailable(0, 100, 400))
        store.setNight(0, 300, false)
        assertFalse(store.isAvailable(0, 10, 400))
        assertTrue(store.isAvailable(0, 10, 300))
        assertTrue(store.isAvailable(0, 301, 400))
        assertTrue(store.allNightsOpenByBits(0, 10, 300))
    }

    @Test
    fun `flexible mask has 49 windows and agrees with the per-night scan`() {
        val rnd = Random(3)
        val store = AvailabilityStore(horizon, 50)
        val scan = PerNightScan(horizon, 50)
        for (r in 0 until 50) {
            val p = pack(BooleanArray(horizon) { rnd.nextDouble() < 0.8 })
            store.load(r, p); scan.load(r, p)
        }
        val allOpen = AvailabilityStore(horizon, 1).also { it.load(0, ByteArray(horizon / 8) { -1 }) }
        assertEquals(49, java.lang.Long.bitCount(allOpen.flexibleMask(0, 100, 110)))
        repeat(2000) {
            val r = rnd.nextInt(50)
            val a = rnd.nextInt(0, horizon)
            val b = a + rnd.nextInt(1, 15)
            val minN = rnd.nextInt(1, 4)
            val maxN = rnd.nextInt(minN, 30)
            assertEquals(scan.flexibleMask(r, a, b, 3, minN, maxN), store.flexibleMask(r, a, b, 3, minN, maxN))
            assertEquals(scan.isAvailable(r, a, b), store.isAvailable(r, a, b))
        }
    }

    @Test
    fun `property - any calendar and any flip sequence keeps runs exact`() = runBlocking {
        val flip = Arb.bind(Arb.int(0 until horizon), Arb.boolean()) { n, o -> n to o }
        checkAll(
            PropTestConfig(iterations = 150),
            Arb.list(Arb.int(1..300), 1..30),
            Arb.boolean(),
            Arb.list(flip, 0..200),
        ) { runs, startOpen, flips ->
            val ref = Reference(calendar(runs, startOpen))
            val store = AvailabilityStore(horizon, 1)
            store.load(0, pack(ref.open))
            assertRunsMatch(store, ref, 0)
            for ((night, open) in flips) {
                store.setNight(0, night, open)
                ref.open[night] = open
            }
            assertRunsMatch(store, ref, 0)
            for (a in 0 until horizon step 7) for (len in intArrayOf(1, 2, 5, 14, 30, 254, 255, 256, 300)) {
                assertEquals(ref.available(a, a + len), store.isAvailable(0, a, a + len), "[$a, ${a + len})")
            }
        }
        Unit
    }

    @Test
    fun `readers see consistent rows while a writer flips nights`() {
        val store = AvailabilityStore(horizon, 1)
        store.load(0, ByteArray(horizon / 8) { -1 })
        // The writer toggles night 200 only; for any reader, [100, 300) is available exactly when
        // night 200 is open, and the run at 100 is either 100 (closed) or 255-capped (open).
        val stop = java.util.concurrent.atomic.AtomicBoolean(false)
        val writer = Thread {
            var open = false
            while (!stop.get()) {
                store.setNight(0, 200, open); open = !open
            }
        }
        writer.start()
        val bad = java.util.concurrent.atomic.AtomicLong()
        val readers = (0 until 3).map {
            Thread {
                repeat(200_000) {
                    val run = store.runValue(0, 100)
                    if (run != 100 && run != 255) bad.incrementAndGet()
                    val mask = store.flexibleMask(0, 195, 205) // every window contains night 200
                    if (mask != 0L && java.lang.Long.bitCount(mask) != 49) bad.incrementAndGet()
                }
            }.apply { start() }
        }
        readers.forEach { it.join() }
        stop.set(true); writer.join()
        assertEquals(0, bad.get())
    }
}
