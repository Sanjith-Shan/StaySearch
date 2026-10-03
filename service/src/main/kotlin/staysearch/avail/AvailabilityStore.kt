package staysearch.avail

import java.lang.invoke.VarHandle
import java.util.concurrent.atomic.AtomicIntegerArray

/**
 * Per-listing availability: a bitset of [horizon] nights plus a run-length array, updated in place.
 *
 * Layout (flat arrays indexed by row, no per-listing objects):
 * - `bits`: [words] longs per row. Bit `d` (word `d / 64`, bit `d % 64`) is 1 iff night `d` is open.
 * - `run`: [horizon] bytes per row. `run[d]` is the number of consecutive open nights starting at
 *   night `d`, saturated at [RUN_CAP] (255, read as unsigned).
 *
 * Why bytes with a cap instead of shorts: a short per night costs 2 x horizon bytes per listing
 * (896 B at 448 nights), a byte halves that, and stays longer than 255 nights are rare. The cap stays
 * exact: a stored 255 means "at least 255 open nights from here", so the check continues at
 * `d + 255` for the remaining nights. With a 448-night horizon that is at most two lookups, so a stay
 * check is still O(1).
 *
 * A stay `[a, b)` (check-in night a, check-out morning b) is available iff `run[a] >= b - a`.
 *
 * Concurrency: one writer thread, many reader threads. Each row has a sequence counter (a seqlock)
 * in an [AtomicIntegerArray]. The writer makes it odd, fences, mutates the row's plain arrays in
 * place, then publishes an even value with a release store. Readers read the counter (acquire), read
 * the row, issue an acquire fence, and re-read the counter; if it changed or was odd they retry. That
 * gives every reader a consistent view of a row (the bitset and run array agree) and, through the
 * release/acquire pair, makes the writer's in-place stores visible without locks. Rows appended at
 * runtime are published by the caller (see [staysearch.data.ListingTable]); a row is only reachable
 * from a Lucene document indexed after the row was written, and the Lucene refresh that exposes the
 * document happens-after that write.
 */
class AvailabilityStore(val horizon: Int, val capacity: Int) {
    val words = (horizon + 63) / 64
    private val bits = LongArray(Math.multiplyExact(capacity, words))
    private val run = ByteArray(Math.multiplyExact(capacity, horizon))
    private val seq = AtomicIntegerArray(capacity)

    init {
        require(horizon in 1..65_535) { "horizon $horizon out of range" }
    }

    // ---------------------------------------------------------------- writer side (one thread)

    /** Replaces a row from the bundle's packed layout (`horizon / 8` bytes, LSB first). */
    fun load(row: Int, packed: ByteArray, offset: Int = 0) = write(row) {
        val base = row * words
        for (w in 0 until words) bits[base + w] = 0L
        for (d in 0 until horizon) {
            val byte = packed[offset + (d ushr 3)].toInt()
            if ((byte ushr (d and 7)) and 1 == 1) bits[base + (d ushr 6)] = bits[base + (d ushr 6)] or (1L shl (d and 63))
        }
        rebuildRun(row)
    }

    /** Marks every night of a row closed (used when a listing is removed). */
    fun clear(row: Int) = write(row) {
        val base = row * words
        for (w in 0 until words) bits[base + w] = 0L
        run.fill(0, row * horizon, (row + 1) * horizon)
    }

    /**
     * Sets one night open or closed. Fixes the bit and recomputes run[night] and then run[d] for
     * d < night while d is open, stopping early once a value is unchanged (everything to its left is
     * then unchanged too). Nights after [night] never change. Returns true if the bit flipped.
     */
    fun setNight(row: Int, night: Int, open: Boolean): Boolean {
        if (night !in 0 until horizon) return false
        if (isOpenUnsynced(row, night) == open) return false
        write(row) {
            val wi = row * words + (night ushr 6)
            val mask = 1L shl (night and 63)
            bits[wi] = if (open) bits[wi] or mask else bits[wi] and mask.inv()
            val rb = row * horizon
            var d = night
            while (d >= 0) {
                val next = if (d + 1 < horizon) (run[rb + d + 1].toInt() and 0xFF) else 0
                val v = if (isOpenUnsynced(row, d)) minOf(RUN_CAP, next + 1) else 0
                if (d != night && (run[rb + d].toInt() and 0xFF) == v) break
                run[rb + d] = v.toByte()
                if (v == 0 && d != night) break
                d--
            }
        }
        return true
    }

    private fun rebuildRun(row: Int) {
        val rb = row * horizon
        var next = 0
        for (d in horizon - 1 downTo 0) {
            next = if (isOpenUnsynced(row, d)) minOf(RUN_CAP, next + 1) else 0
            run[rb + d] = next.toByte()
        }
    }

    private inline fun write(row: Int, body: () -> Unit) {
        val s = seq.get(row)
        seq.set(row, s + 1) // odd: write in progress
        VarHandle.storeStoreFence() // the odd value must be visible before any data store
        try {
            body()
        } finally {
            seq.setRelease(row, s + 2) // even: publishes every data store above
        }
    }

    // ---------------------------------------------------------------- reader side (any thread)

    private fun isOpenUnsynced(row: Int, night: Int): Boolean =
        (bits[row * words + (night ushr 6)] ushr (night and 63)) and 1L == 1L

    private fun runAt(row: Int, d: Int): Int = run[row * horizon + d].toInt() and 0xFF

    private fun availableUnsynced(row: Int, checkin: Int, checkout: Int): Boolean {
        if (checkin < 0 || checkout > horizon || checkout <= checkin) return false
        var a = checkin
        var need = checkout - checkin
        while (true) {
            val r = runAt(row, a)
            if (r >= need) return true
            if (r < RUN_CAP) return false
            a += RUN_CAP
            need -= RUN_CAP
        }
    }

    /** Runs [read] under the row's seqlock until it sees a stable row. */
    private inline fun <T> read(row: Int, read: () -> T): T {
        while (true) {
            val s = seq.getAcquire(row)
            if (s and 1 == 1) {
                Thread.onSpinWait(); continue
            }
            val v = read()
            VarHandle.acquireFence() // data loads above complete before the re-check below
            if (seq.get(row) == s) return v
        }
    }

    /** O(1) check that every night in `[checkin, checkout)` is open (night offsets from base_date). */
    fun isAvailable(row: Int, checkin: Int, checkout: Int): Boolean = read(row) { availableUnsynced(row, checkin, checkout) }

    /** Reads one night from the bitset (independent of the run array). */
    fun isOpen(row: Int, night: Int): Boolean =
        night in 0 until horizon && read(row) { isOpenUnsynced(row, night) }

    /** Checks each night from the bitset one by one; used by the invariant checker as an independent path. */
    fun allNightsOpenByBits(row: Int, checkin: Int, checkout: Int): Boolean {
        if (checkin < 0 || checkout > horizon || checkout <= checkin) return false
        return read(row) { (checkin until checkout).all { isOpenUnsynced(row, it) } }
    }

    /**
     * Flexible-date check: all (checkin + i, checkout + j) pairs with i, j in [-flex, flex] (49 pairs
     * at flex = 3). A pair counts if it is a valid stay, its length is within [minNights, maxNights],
     * and it is open. Returns a bitmask with bit `(i + flex) * (2 flex + 1) + (j + flex)` set for each
     * open pair (fits in a Long for flex <= 3). One run lookup per pair, all under one seqlock read.
     */
    fun flexibleMask(row: Int, checkin: Int, checkout: Int, flex: Int = 3, minNights: Int = 1, maxNights: Int = Int.MAX_VALUE): Long {
        require(flex in 0..3)
        val side = 2 * flex + 1
        return read(row) {
            var mask = 0L
            for (i in -flex..flex) for (j in -flex..flex) {
                val a = checkin + i
                val b = checkout + j
                val n = b - a
                if (n < 1 || n < minNights || n > maxNights) continue
                if (availableUnsynced(row, a, b)) mask = mask or (1L shl ((i + flex) * side + (j + flex)))
            }
            mask
        }
    }

    /**
     * Open-night gaps around a stay `[checkin, checkout)`, read under one seqlock pass. Returns
     * `gapBefore shl 16 or gapAfter` (unpack with [gapBeforeOf] and [gapAfterOf]), each capped at [cap].
     * - gapBefore: consecutive open nights walking back from `checkin - 1`, stopping after night 0.
     * - gapAfter: consecutive open nights starting at `checkout` (the run value, which is exact up to 255).
     * Nights before night 0 or at or beyond the horizon count as closed.
     */
    fun gaps(row: Int, checkin: Int, checkout: Int, cap: Int): Int {
        require(cap in 0..RUN_CAP) { "gap cap $cap out of range" }
        return read(row) {
            var before = 0
            var d = checkin - 1
            while (before < cap && d >= 0 && d < horizon && isOpenUnsynced(row, d)) {
                before++; d--
            }
            val after = if (checkout in 0 until horizon) minOf(runAt(row, checkout), cap) else 0
            (before shl 16) or after
        }
    }

    /** The raw run value (saturated at 255) for tests and the listing endpoint. */
    fun runValue(row: Int, night: Int): Int = read(row) { runAt(row, night) }

    /** A copy of a row's bitset words. */
    fun bitsOf(row: Int): LongArray = read(row) { bits.copyOfRange(row * words, (row + 1) * words) }

    /** Number of open nights in `[from, to)`. */
    fun openCount(row: Int, from: Int = 0, to: Int = horizon): Int =
        read(row) { (from.coerceAtLeast(0) until to.coerceAtMost(horizon)).count { isOpenUnsynced(row, it) } }

    /** Bytes held by this store's arrays (bitsets, run arrays, seqlock counters). */
    val bytes: Long get() = bits.size * 8L + run.size.toLong() + capacity * 4L

    companion object {
        const val RUN_CAP = 255

        fun gapBeforeOf(packed: Int): Int = packed ushr 16
        fun gapAfterOf(packed: Int): Int = packed and 0xFFFF
    }
}

/**
 * The baseline for exp5: one boolean per night, a stay check scans every night of the window.
 * Not thread-safe; it exists only for benchmarks and tests.
 */
class PerNightScan(val horizon: Int, val capacity: Int) {
    private val open = BooleanArray(Math.multiplyExact(capacity, horizon))

    fun load(row: Int, packed: ByteArray, offset: Int = 0) {
        for (d in 0 until horizon) open[row * horizon + d] = (packed[offset + (d ushr 3)].toInt() ushr (d and 7)) and 1 == 1
    }

    fun setNight(row: Int, night: Int, value: Boolean) {
        open[row * horizon + night] = value
    }

    fun isAvailable(row: Int, checkin: Int, checkout: Int): Boolean {
        if (checkin < 0 || checkout > horizon || checkout <= checkin) return false
        val base = row * horizon
        for (d in checkin until checkout) if (!open[base + d]) return false
        return true
    }

    fun flexibleMask(row: Int, checkin: Int, checkout: Int, flex: Int = 3, minNights: Int = 1, maxNights: Int = Int.MAX_VALUE): Long {
        val side = 2 * flex + 1
        var mask = 0L
        for (i in -flex..flex) for (j in -flex..flex) {
            val a = checkin + i
            val b = checkout + j
            val n = b - a
            if (n < 1 || n < minNights || n > maxNights) continue
            if (isAvailable(row, a, b)) mask = mask or (1L shl ((i + flex) * side + (j + flex)))
        }
        return mask
    }
}
