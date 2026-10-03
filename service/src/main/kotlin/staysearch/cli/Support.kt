package staysearch.cli

import org.HdrHistogram.ConcurrentHistogram
import staysearch.bundle.Bundle
import staysearch.search.SearchRequest
import staysearch.util.Json
import staysearch.util.Machine
import java.nio.file.Path
import java.nio.file.Paths
import java.time.LocalDate
import java.util.concurrent.Executors
import java.util.concurrent.ThreadPoolExecutor
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.locks.LockSupport

/** `--key value` and bare `--flag` arguments. */
class Args(argv: List<String>) {
    private val values = HashMap<String, String>()

    init {
        var i = 0
        while (i < argv.size) {
            val a = argv[i]
            require(a.startsWith("--")) { "unexpected argument '$a'" }
            val eq = a.indexOf('=')
            if (eq > 0) {
                values[a.substring(2, eq)] = a.substring(eq + 1); i++
            } else if (i + 1 < argv.size && !argv[i + 1].startsWith("--")) {
                values[a.substring(2)] = argv[i + 1]; i += 2
            } else {
                values[a.substring(2)] = "true"; i++
            }
        }
    }

    fun str(k: String): String? = values[k]
    fun str(k: String, def: String): String = values[k] ?: def
    fun req(k: String): String = values[k] ?: throw IllegalArgumentException("--$k is required")
    fun path(k: String): Path = Paths.get(req(k))
    fun pathOrNull(k: String): Path? = values[k]?.let { Paths.get(it) }
    fun int(k: String, def: Int) = values[k]?.toInt() ?: def
    fun long(k: String, def: Long) = values[k]?.toLong() ?: def
    fun double(k: String, def: Double) = values[k]?.toDouble() ?: def
    fun flag(k: String, def: Boolean = false) = values[k]?.toBooleanStrict() ?: def
    fun list(k: String, def: String): List<String> = str(k, def).split(',').map { it.trim() }.filter { it.isNotEmpty() }
}

/** Reads queries_test.jsonl into requests, with the retrieval and ranking knobs applied. */
fun loadQueries(bundle: Bundle, limit: Int, base: SearchRequest.() -> SearchRequest = { this }): List<SearchRequest> =
    Json.readLines(bundle.queriesPath).take(if (limit <= 0) Int.MAX_VALUE else limit).map { n ->
        SearchRequest(
            placeIdx = n["place_idx"].asInt(),
            checkin = LocalDate.parse(n["checkin"].asText()),
            checkout = LocalDate.parse(n["checkout"].asText()),
            guests = n["guests"].asInt(),
            guestId = n["guest_id"]?.takeIf { !it.isNull }?.asLong(),
            searchDate = n["search_date"]?.takeIf { !it.isNull }?.let { LocalDate.parse(it.asText()) },
        ).base()
    }.toList()

/** Fields every results line carries. */
fun provenance(bundle: Bundle, repeat: Int, loadStart: Map<String, Any?>): Map<String, Any?> = mapOf(
    "machine" to Machine.info,
    "load_start" to loadStart,
    "load_end" to Machine.load(),
    "repeat" to repeat,
    "bundle_as_of" to bundle.manifest.asOf.toString(),
    "bundle_dir" to bundle.dir.toAbsolutePath().toString(),
    "bundle_synthetic" to bundle.manifest.synthetic,
    "n_listings" to bundle.manifest.nListings,
    "lucene_version" to Machine.luceneVersion,
)

/**
 * Open-loop arrivals at a constant rate. Arrival i is due at start + i / rate; it is handed to a
 * worker pool at that time regardless of how earlier arrivals are doing, and its latency is measured
 * from the due time (so queueing behind slow requests counts; no coordinated omission).
 */
class OpenLoop(private val rate: Double, private val threads: Int) {
    val latencyUs = ConcurrentHistogram(3)
    val errors = AtomicLong()
    val completed = AtomicLong()
    var issued = 0L
        private set
    var lastError: Throwable? = null

    /**
     * Runs for [durationS] seconds. [task] runs on a worker; arrivals within the first [warmupS]
     * seconds run but are not recorded (and [onWarmupEnd] is called once when recording starts).
     */
    fun run(durationS: Double, warmupS: Double = 0.0, onWarmupEnd: () -> Unit = {}, task: (Long, Boolean) -> Unit) {
        val pool = Executors.newFixedThreadPool(threads) { r -> Thread(r, "load").apply { isDaemon = true } } as ThreadPoolExecutor
        val interval = 1e9 / rate
        val start = System.nanoTime()
        val warmEnd = start + (warmupS * 1e9).toLong()
        val end = warmEnd + (durationS * 1e9).toLong()
        var warmed = warmupS <= 0
        if (warmed) onWarmupEnd()
        var i = 0L
        while (true) {
            val due = start + (i * interval).toLong()
            if (due >= end) break
            while (true) {
                val wait = due - System.nanoTime()
                if (wait <= 0) break
                LockSupport.parkNanos(wait)
            }
            val record = due >= warmEnd
            if (record && !warmed) {
                warmed = true; onWarmupEnd()
            }
            val idx = i
            if (record) issued++
            pool.execute {
                try {
                    task(idx, record)
                    if (record) {
                        latencyUs.recordValue(((System.nanoTime() - due) / 1000).coerceAtLeast(0))
                        completed.incrementAndGet()
                    }
                } catch (t: Throwable) {
                    if (record) errors.incrementAndGet()
                    lastError = t
                }
            }
            i++
        }
        pool.shutdown()
        if (!pool.awaitTermination(60, TimeUnit.SECONDS)) pool.shutdownNow()
    }
}
