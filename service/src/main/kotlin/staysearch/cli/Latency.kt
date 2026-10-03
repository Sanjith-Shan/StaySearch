package staysearch.cli

import org.HdrHistogram.ConcurrentHistogram
import staysearch.Engine
import staysearch.EngineConfig
import staysearch.bundle.Bundle
import staysearch.search.SearchRequest
import staysearch.search.StageStats
import staysearch.updates.Replayer
import staysearch.updates.UpdateApplier
import staysearch.updates.UpdateSource
import staysearch.util.Json
import staysearch.util.JsonlWriter
import staysearch.util.Machine
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.time.Duration
import java.util.concurrent.CompletableFuture
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.locks.LockSupport

/**
 * latency / exp9: open-loop load at constant arrival rates, latency measured from each request's
 * intended send time (no coordinated omission), with a warm-up phase that is run but not recorded.
 * Runs every rate without updates and with the update stream replaying at --update-rate.
 *
 * In-process by default (the generator calls the pipeline on a worker pool in the same JVM). With
 * --target http://host:port it sends POST /search over HTTP with async sends instead, and the
 * per-stage numbers come from the server's response.
 *
 * Options: --bundle DIR --out FILE [--rates 50,200,500] [--duration 30] [--warmup 10]
 * [--updates false,true] [--update-rate 1000] [--reindex-avail false] [--mode ivf] [--ranker dnn]
 * [--k 500] [--t 25] [--nprobe 16] [--threads 2 x cores] [--repeats 1] [--target URL]
 */
object Latency {
    fun run(a: Args): Int {
        val bundle = Bundle.load(a.path("bundle"))
        val rates = a.list("rates", "50,200,500").map { it.toDouble() }
        val withUpdates = a.list("updates", "false,true").map { it.toBooleanStrict() }
        val target = a.str("target")
        val queries = loadQueries(bundle, 0) {
            copy(mode = a.str("mode", "ivf"), ranker = a.str("ranker", "dnn"), k = a.int("k", 500), t = a.int("t", 25), nprobe = a.int("nprobe", 16))
        }
        val threads = a.int("threads", 2 * Runtime.getRuntime().availableProcessors())
        JsonlWriter(a.path("out")).use { w ->
            for (repeat in 1..a.int("repeats", 1)) for (upd in withUpdates) {
                val eng = if (target == null) Engine.open(bundle.dir, EngineConfig(withVectors = queries.first().mode == "hnsw"), bundle) else null
                try {
                    for (rate in rates) {
                        val row = runRate(a, bundle, eng, target, queries, rate, upd, threads, repeat)
                        w.write(row)
                        println(Json.write(row.filterKeys { it !in setOf("machine", "load_start", "load_end") }))
                    }
                } finally {
                    eng?.close()
                }
            }
        }
        return 0
    }

    private fun runRate(
        a: Args, bundle: Bundle, eng: Engine?, target: String?, queries: List<SearchRequest>, rate: Double,
        updates: Boolean, threads: Int, repeat: Int,
    ): Map<String, Any?> {
        System.gc()
        val loadStart = Machine.load()
        val stages = StageStats()
        val updateRate = if (updates) a.double("update-rate", 1000.0) else 0.0
        val replayer = if (updates && eng != null) {
            Replayer(UpdateSource(bundle.updatesPath, loop = true), UpdateApplier(eng.table, eng.index, a.flag("reindex-avail")), updateRate)
        } else null
        if (updates && eng == null) System.err.println("note: with --target, updates must run inside the server (serve --updates-rate)")
        replayer?.start()
        val duration = a.double("duration", 30.0)
        val warmup = a.double("warmup", 10.0)
        val summary: Map<String, Any?>
        if (target == null) {
            val loop = OpenLoop(rate, threads)
            loop.run(duration, warmup, onWarmupEnd = { stages.reset() }) { i, record ->
                val resp = eng!!.pipeline.search(queries[(i % queries.size).toInt()], record = false)
                if (record) stages.record(resp.timingsUs)
            }
            summary = mapOf(
                "issued" to loop.issued, "completed" to loop.completed.get(), "errors" to loop.errors.get(),
                "last_error" to loop.lastError?.toString(), "total_from_intended_us" to StageStats.summarize(loop.latencyUs), "generator_dispatch_lag_us" to StageStats.summarize(loop.dispatchLagUs),
            )
        } else {
            summary = runHttp(target, queries, rate, duration, warmup, stages)
        }
        val replay = replayer?.stop()
        return linkedMapOf<String, Any?>(
            "experiment" to "exp9_latency",
            "arrival_rate_qps" to rate,
            "duration_s" to duration,
            "warmup_s" to warmup,
            "with_updates" to updates,
            "update_rate" to updateRate,
            "replay" to replay,
            "mode" to queries.first().mode, "ranker" to queries.first().ranker, "k" to queries.first().k,
            "t" to queries.first().t, "nprobe" to queries.first().nprobe,
            "generator" to if (target == null) "in-process, same JVM, $threads worker threads" else "HTTP async to $target, same machine",
            "service_stages_us" to stages.summary(),
        ) + summary + provenance(bundle, repeat, loadStart)
    }

    private fun runHttp(target: String, queries: List<SearchRequest>, rate: Double, duration: Double, warmup: Double, stages: StageStats): Map<String, Any?> {
        val client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build()
        val uri = URI.create(target.trimEnd('/') + "/search")
        val hist = ConcurrentHistogram(3)
        val errors = AtomicLong()
        val completed = AtomicLong()
        val pending = ArrayList<CompletableFuture<*>>()
        val bodies = queries.map { Json.write(it) }
        val interval = 1e9 / rate
        val start = System.nanoTime()
        val warmEnd = start + (warmup * 1e9).toLong()
        val end = warmEnd + (duration * 1e9).toLong()
        var issued = 0L
        var i = 0L
        while (true) {
            val due = start + (i * interval).toLong()
            if (due >= end) break
            while (due - System.nanoTime() > 0) LockSupport.parkNanos(due - System.nanoTime())
            val record = due >= warmEnd
            if (record) issued++
            val req = HttpRequest.newBuilder(uri).header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(bodies[(i % bodies.size).toInt()])).build()
            pending += client.sendAsync(req, HttpResponse.BodyHandlers.ofString()).whenComplete { r, e ->
                if (!record) return@whenComplete
                if (e != null || r.statusCode() != 200) {
                    errors.incrementAndGet(); return@whenComplete
                }
                hist.recordValue((System.nanoTime() - due) / 1000)
                completed.incrementAndGet()
                val t = Json.mapper.readTree(r.body())["timings_us"]
                stages.record(staysearch.search.StageTimings(
                    t["query_tower"].asLong(), t["probe"].asLong(), t["filter"].asLong(),
                    t["distance"].asLong(), t["rank"].asLong(), t["total"].asLong(),
                ))
            }
            i++
        }
        runCatching { CompletableFuture.allOf(*pending.toTypedArray()).get(60, java.util.concurrent.TimeUnit.SECONDS) }
        return mapOf("issued" to issued, "completed" to completed.get(), "errors" to errors.get(), "total_from_intended_us" to StageStats.summarize(hist))
    }
}
