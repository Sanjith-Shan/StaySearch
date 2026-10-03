package staysearch.updates

import com.fasterxml.jackson.databind.JsonNode
import staysearch.bundle.ListingRecord
import staysearch.data.ListingTable
import staysearch.index.ListingIndex
import staysearch.util.Json
import java.io.BufferedReader
import java.nio.file.Files
import java.nio.file.Path
import java.util.Base64
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.locks.LockSupport

/** One line of updates.jsonl. */
sealed interface UpdateEvent {
    data class Avail(val id: Long, val night: Int, val open: Boolean) : UpdateEvent
    data class Price(val id: Long, val price: Double) : UpdateEvent
    data class Remove(val id: Long) : UpdateEvent
    class New(val listing: ListingRecord, val emb: FloatArray, val static: FloatArray, val avail: ByteArray) : UpdateEvent

    companion object {
        fun parse(n: JsonNode): UpdateEvent = when (val t = n["t"].asText()) {
            "avail" -> Avail(n["id"].asLong(), n["night"].asInt(), n["open"].asBoolean())
            "price" -> Price(n["id"].asLong(), n["price"].asDouble())
            "remove" -> Remove(n["id"].asLong())
            "new" -> New(
                ListingRecord.parse(n["listing"]),
                FloatArray(n["emb"].size()) { n["emb"][it].floatValue() },
                FloatArray(n["static"].size()) { n["static"][it].floatValue() },
                Base64.getDecoder().decode(n["avail_b64"].asText()),
            )
            else -> throw IllegalArgumentException("unknown update type '$t'")
        }

        /** Counts "new" events with a cheap text scan, to size the listing table. */
        fun countNew(path: Path): Int {
            if (!Files.exists(path)) return 0
            var c = 0
            Files.newBufferedReader(path).useLines { lines -> lines.forEach { if (it.contains("\"t\": \"new\"") || it.contains("\"t\":\"new\"")) c++ } }
            return c
        }
    }
}

/** Streams updates.jsonl in file order; with [loop] it starts over at EOF and counts [wraps]. */
class UpdateSource(private val path: Path, private val loop: Boolean) : AutoCloseable {
    private var reader: BufferedReader = Files.newBufferedReader(path)
    var wraps = 0
        private set
    var exhausted = false
        private set

    fun next(): UpdateEvent? {
        while (true) {
            val line = reader.readLine()
            if (line == null) {
                if (!loop) {
                    exhausted = true; return null
                }
                reader.close(); reader = Files.newBufferedReader(path); wraps++
                continue
            }
            if (line.isNotBlank()) return UpdateEvent.parse(Json.mapper.readTree(line))
        }
    }

    override fun close() = reader.close()
}

/**
 * Applies events to the listing table, the availability store and the index. Only one thread may
 * call [apply].
 *
 * - avail: flips one night in place in the store (visible to searches at once, no index write).
 *   With [reindexOnAvail] (the index-placement experiment) the listing's document is also rewritten
 *   with updateDocument, as a design that keeps availability in the index would have to.
 * - price: updates the table and rewrites the document.
 * - remove: closes every night, forgets the id and deletes the document.
 * - new (or a known id again): upserts the row (cluster id = nearest centroid, the cold-start path)
 *   and writes the document.
 * Events for unknown ids are counted as skipped.
 */
class UpdateApplier(
    private val table: ListingTable,
    private val index: ListingIndex?,
    private val reindexOnAvail: Boolean,
) {
    val applied = AtomicLong()
    val skipped = AtomicLong()
    val indexWrites = AtomicLong()
    val byType = mapOf("avail" to AtomicLong(), "price" to AtomicLong(), "remove" to AtomicLong(), "new" to AtomicLong())

    fun apply(e: UpdateEvent) {
        when (e) {
            is UpdateEvent.Avail -> {
                val row = table.rowOf(e.id) ?: return skip()
                table.store.setNight(row, e.night, e.open)
                if (reindexOnAvail) write { index?.upsert(row) }
                byType["avail"]!!.incrementAndGet()
            }
            is UpdateEvent.Price -> {
                val row = table.rowOf(e.id) ?: return skip()
                table.price[row] = e.price
                write { index?.upsert(row) }
                byType["price"]!!.incrementAndGet()
            }
            is UpdateEvent.Remove -> {
                table.remove(e.id) ?: return skip()
                write { index?.delete(e.id) }
                byType["remove"]!!.incrementAndGet()
            }
            is UpdateEvent.New -> {
                val row = table.upsert(e.listing, e.emb, e.static, e.avail, clusterId = null)
                write { index?.upsert(row) }
                byType["new"]!!.incrementAndGet()
            }
        }
        applied.incrementAndGet()
    }

    private inline fun write(body: () -> Any?) {
        if (index != null) {
            body(); indexWrites.incrementAndGet()
        }
    }

    private fun skip() {
        skipped.incrementAndGet()
    }

    fun counters(): Map<String, Long> = mapOf(
        "applied" to applied.get(), "skipped" to skipped.get(), "index_writes" to indexWrites.get(),
    ) + byType.mapValues { it.value.get() }
}

/**
 * Replays an [UpdateSource] at a constant [rate] (events/s) on its own thread, open loop: event i is
 * due at start + i / rate whatever happened to earlier events, so when the applier falls behind the
 * backlog grows and is reported instead of the schedule silently stretching.
 */
class Replayer(
    private val source: UpdateSource,
    private val applier: UpdateApplier,
    private val rate: Double,
) : AutoCloseable {
    @Volatile
    private var running = false
    private var thread: Thread? = null
    private var startNs = 0L
    private var stopNs = 0L
    val done = AtomicLong()
    var error: Throwable? = null
        private set

    fun start() {
        if (rate <= 0) return
        running = true
        startNs = System.nanoTime()
        thread = Thread({ loop() }, "update-replay").apply { isDaemon = true; start() }
    }

    private fun loop() {
        try {
            val nsPerEvent = 1e9 / rate
            while (running) {
                val due = ((System.nanoTime() - startNs) / nsPerEvent).toLong() + 1
                var batch = 0
                while (done.get() < due && running && batch < 1024) {
                    val e = source.next() ?: run { running = false; null } ?: break
                    applier.apply(e)
                    done.incrementAndGet()
                    batch++
                }
                if (done.get() >= due) {
                    val nextDue = startNs + (done.get() * nsPerEvent).toLong()
                    val wait = nextDue - System.nanoTime()
                    if (wait > 0) LockSupport.parkNanos(minOf(wait, 1_000_000))
                }
            }
        } catch (t: Throwable) {
            error = t
            running = false
        } finally {
            stopNs = System.nanoTime()
        }
    }

    /** Events that were due by now (the schedule), for backlog = intended minus done. */
    fun intendedNow(): Long = if (startNs == 0L) 0 else ((System.nanoTime() - startNs) * rate / 1e9).toLong()

    fun stop(): Map<String, Any?> {
        val intended = intendedNow()
        running = false
        thread?.join(10_000)
        val elapsed = if (startNs == 0L) 0.0 else ((if (stopNs != 0L) stopNs else System.nanoTime()) - startNs) / 1e9
        return mapOf(
            "target_rate" to rate,
            "events_done" to done.get(),
            "events_intended" to intended,
            "backlog" to (intended - done.get()).coerceAtLeast(0),
            "achieved_rate" to if (elapsed > 0) done.get() / elapsed else 0.0,
            "elapsed_s" to elapsed,
            "wraps" to source.wraps,
            "exhausted" to source.exhausted,
            "error" to error?.toString(),
        )
    }

    override fun close() {
        running = false
        thread?.join(10_000)
    }
}
