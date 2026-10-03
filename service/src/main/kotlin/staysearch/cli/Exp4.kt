package staysearch.cli

import org.HdrHistogram.ConcurrentHistogram
import staysearch.Engine
import staysearch.EngineConfig
import staysearch.bundle.Bundle
import staysearch.search.Features
import staysearch.search.Mode
import staysearch.search.Models
import staysearch.search.StageStats
import staysearch.updates.Replayer
import staysearch.updates.UpdateSource
import staysearch.util.Json
import staysearch.util.JsonlWriter
import staysearch.util.Machine
import java.nio.file.Files
import java.nio.file.Paths
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.DoubleAdder

/**
 * exp4: IVF-in-Lucene vs HNSW-in-Lucene on the same documents under the replayed update stream.
 *
 * For each repeat, mode and rate: build a fresh index from the bundle (IVF: no vectors in Lucene;
 * HNSW: KnnFloatVectorField), warm up with queries, then for --duration seconds replay updates at
 * the rate (every event rewrites the listing's document, as in the index-placement design) while
 * issuing test queries open-loop at --qps. Every --recall-every'th query is also run in exact mode
 * right after it, and recall@100 vs exact is computed on the retrieval output (before the ranker).
 *
 * Options: --bundle DIR --out FILE [--modes ivf,hnsw] [--rates 0,100,1000,10000] [--duration 60]
 * [--qps 20] [--query-threads 4] [--recall-every 5] [--warmup-queries 200] [--repeats 1]
 * [--k 500] [--nprobe 16] [--ranker dnn] [--refresh-ms 1000] [--lag-sample 100]
 * [--index-root DIR] [--no-loop]
 */
object Exp4 {
    fun run(a: Args): Int {
        val bundle = Bundle.load(a.path("bundle"))
        val out = a.path("out")
        val modes = a.list("modes", "ivf,hnsw").map(Mode::parse)
        val rates = a.list("rates", "0,100,1000,10000").map { it.toDouble() }
        val duration = a.double("duration", 60.0)
        val qps = a.double("qps", 20.0)
        val recallEvery = a.int("recall-every", 5)
        val k = a.int("k", 500)
        val nprobe = a.int("nprobe", 16)
        val ranker = a.str("ranker", "dnn")
        val loop = !a.flag("no-loop")
        val indexRoot = a.pathOrNull("index-root")?.also { Files.createDirectories(it) }
        val queries = loadQueries(bundle, 0) { copy(k = k, nprobe = nprobe, ranker = ranker) }
        require(queries.isNotEmpty()) { "no queries in ${bundle.queriesPath}" }
        val models = Models(bundle)
        JsonlWriter(out).use { w ->
            for (repeat in 1..a.int("repeats", 1)) for (mode in modes) for (rate in rates) {
                val row = runOne(a, bundle, models, queries, mode, rate, duration, qps, recallEvery, loop, indexRoot, repeat)
                w.write(row)
                println(Json.write(row.filterKeys { it !in setOf("machine", "load_start", "load_end", "query_stages") }))
            }
        }
        models.close()
        return 0
    }

    private fun runOne(
        a: Args, bundle: Bundle, models: Models, queries: List<staysearch.search.SearchRequest>, mode: Mode, rate: Double,
        duration: Double, qps: Double, recallEvery: Int, loop: Boolean, indexRoot: java.nio.file.Path?, repeat: Int,
    ): Map<String, Any?> {
        System.gc()
        val loadStart = Machine.load()
        val indexDir = indexRoot?.let { Files.createTempDirectory(it, "exp4-") }
        val cfg = EngineConfig(
            indexDir = indexDir, withVectors = mode == Mode.HNSW, refreshMs = a.long("refresh-ms", 1000),
            lagSampleEvery = a.int("lag-sample", 100), reindexOnAvail = true,
        )
        val eng = Engine.open(bundle.dir, cfg, bundle, models)
        try {
            val modeName = mode.name.lowercase()
            val heapAfterBuild = Machine.heapUsedAfterGc()
            val rssAfterBuild = Machine.rssBytes()
            val bytesAfterBuild = eng.index.bytesOnDisk

            // JIT warm-up, not recorded.
            repeat(a.int("warmup-queries", 200)) { i -> eng.pipeline.search(queries[i % queries.size].copy(mode = modeName), record = false) }
            eng.pipeline.stats.reset(); eng.index.lag.reset(); eng.index.merges.reset()
            val docsBefore = eng.index.docsWritten.get()

            val replayer = Replayer(UpdateSource(bundle.updatesPath, loop), eng.applier, rate)
            val recallSum = DoubleAdder()
            val recallN = AtomicLong()
            val serviceUs = ConcurrentHistogram(3)
            val stages = StageStats()
            val load = OpenLoop(qps, a.int("query-threads", 4))
            val t0 = System.nanoTime()
            replayer.start()
            load.run(duration) { i, _ ->
                val q = queries[(i % queries.size).toInt()].copy(mode = modeName)
                val resp = eng.pipeline.search(q)
                serviceUs.recordValue(resp.timingsUs.total)
                stages.record(resp.timingsUs)
                if (recallEvery > 0 && i % recallEvery == 0L) {
                    recallAt100(eng, q, mode)?.let { recallSum.add(it); recallN.incrementAndGet() }
                }
            }
            val replay = replayer.stop()
            val elapsed = (System.nanoTime() - t0) / 1e9
            // Let the last refresh land so pending lag samples resolve.
            Thread.sleep(cfg.refreshMs + 200)
            val lagH = eng.index.lag.histogramMicros
            val docsWritten = eng.index.docsWritten.get() - docsBefore

            return linkedMapOf<String, Any?>(
                "experiment" to "exp4",
                "mode" to modeName,
                "rate_target" to rate,
                "duration_s" to duration,
                "elapsed_s" to elapsed,
                "replay" to replay,
                "applier" to eng.applier.counters(),
                "index_writes_per_s" to docsWritten / elapsed,
                "jvm_heap_used_bytes" to Machine.heapUsedAfterGc(),
                "rss_bytes" to Machine.rssBytes(),
                "index_bytes" to eng.index.bytesOnDisk,
                "heap_after_build_bytes" to heapAfterBuild,
                "rss_after_build_bytes" to rssAfterBuild,
                "index_bytes_after_build" to bytesAfterBuild,
                "index_build_ms" to eng.buildMs,
                "availability_store_bytes" to eng.table.store.bytes,
                "refresh_lag" to mapOf(
                    "samples" to lagH.totalCount, "sample_every" to cfg.lagSampleEvery,
                    "p50_ms" to lagH.getValueAtPercentile(50.0) / 1000.0, "p99_ms" to lagH.getValueAtPercentile(99.0) / 1000.0,
                    "max_ms" to lagH.maxValue / 1000.0, "unresolved" to eng.index.lag.pendingCount(),
                ),
                "merge" to mapOf(
                    "total_ms" to eng.index.merges.mergeNanos.get() / 1e6, "count" to eng.index.merges.mergeCount.get(),
                    "max_ms" to eng.index.merges.maxMergeNanos.get() / 1e6,
                ),
                "recall_at_100_vs_exact" to if (recallN.get() > 0) recallSum.sum() / recallN.get() else null,
                "recall_queries" to recallN.get(),
                "queries" to mapOf(
                    "qps_target" to qps, "issued" to load.issued, "completed" to load.completed.get(), "errors" to load.errors.get(),
                    "last_error" to load.lastError?.toString(),
                    "latency_from_intended_us" to StageStats.summarize(load.latencyUs),
                    "service_time_us" to StageStats.summarize(serviceUs),
                ),
                "query_stages" to stages.summary(),
                "config" to mapOf(
                    "k" to queries.first().k, "nprobe" to queries.first().nprobe, "ranker" to queries.first().ranker,
                    "refresh_ms" to cfg.refreshMs, "hnsw" to if (mode == Mode.HNSW) "lucene defaults (M=16, beamWidth=100)" else null,
                    "updates_loop" to loop, "reindex_on_avail" to true,
                ),
            ) + provenance(bundle, repeat, loadStart)
        } finally {
            eng.close()
            indexDir?.let { staysearch.util.deleteRecursively(it) }
        }
    }

    /** |top100(mode) ∩ top100(exact)| / |top100(exact)| on the retrieval output, or null if exact is empty. */
    fun recallAt100(eng: Engine, q: staysearch.search.SearchRequest, mode: Mode): Double? {
        val ctx = Features.resolve(q, eng.bundle)
        val qemb = eng.models.embedQuery(ctx.placeIdx, Features.queryFeatures(ctx, eng.bundle.manifest.qFeatDim))
        val got = eng.pipeline.retrieve(ctx, qemb, mode, eng.pipeline.retrievalK(q), q.nprobe, q.flexible).rows.take(100).toSet()
        val exact = eng.pipeline.retrieve(ctx, qemb, Mode.EXACT, 100, q.nprobe, q.flexible).rows
        if (exact.isEmpty()) return null
        return exact.count { it in got }.toDouble() / exact.size
    }
}
