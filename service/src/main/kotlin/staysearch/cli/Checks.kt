package staysearch.cli

import staysearch.Engine
import staysearch.EngineConfig
import staysearch.bundle.Bundle
import staysearch.search.Models
import staysearch.updates.UpdateSource
import staysearch.util.Json
import staysearch.util.JsonlWriter
import staysearch.util.Machine
import kotlin.math.abs

/**
 * replay-invariant: runs queries from queries_test.jsonl in each mode, applying [updates-per-query]
 * update events between queries (single-threaded, so each check sees exactly the state the search
 * saw), and fails with exit code 1 on any invariant violation.
 *
 * Options: --bundle DIR, --n N (0 = all), --modes ivf,hnsw,exact, --updates-per-query 10,
 * --flexible-every 7 (every Nth query also runs flexible; 0 = never), --k 500, --t 25,
 * --nprobe 16, --ranker dnn, --refresh-every 50 (queries between forced NRT refreshes),
 * --out results/replay_invariant.jsonl
 */
object ReplayInvariant {
    fun run(a: Args): Int {
        val bundle = Bundle.load(a.path("bundle"))
        val loadStart = Machine.load()
        val modes = a.list("modes", "ivf,hnsw,exact")
        val perQuery = a.int("updates-per-query", 10)
        val flexEvery = a.int("flexible-every", 7)
        val refreshEvery = a.int("refresh-every", 50)
        val queries = loadQueries(bundle, a.int("n", 0)) {
            copy(k = a.int("k", 500), t = a.int("t", 25), nprobe = a.int("nprobe", 16), ranker = a.str("ranker", "dnn"))
        }
        Engine.open(bundle.dir, EngineConfig(withVectors = "hnsw" in modes, refreshMs = 0), bundle).use { eng ->
            val source = if (perQuery > 0 && java.nio.file.Files.exists(bundle.updatesPath)) UpdateSource(bundle.updatesPath, loop = true) else null
            var searches = 0L
            var results = 0L
            var violations = 0L
            val examples = ArrayList<Map<String, Any?>>()
            val t0 = System.nanoTime()
            for ((qi, q0) in queries.withIndex()) {
                source?.let { s -> repeat(perQuery) { s.next()?.let(eng.applier::apply) } }
                if (refreshEvery > 0 && qi % refreshEvery == 0) eng.index.refreshNow()
                val variants = buildList {
                    for (m in modes) add(q0.copy(mode = m))
                    if (flexEvery > 0 && qi % flexEvery == 0) add(q0.copy(mode = modes.first(), flexible = true))
                }
                for (q in variants) {
                    val resp = eng.pipeline.search(q)
                    val v = eng.invariants.check(q, resp)
                    searches++
                    results += resp.results.size
                    if (v.isNotEmpty()) {
                        violations += v.size
                        if (examples.size < 20) examples += mapOf("query_index" to qi, "request" to q, "violations" to v)
                    }
                }
            }
            source?.close()
            val summary = linkedMapOf<String, Any?>(
                "check" to "replay_invariant",
                "queries" to queries.size,
                "searches" to searches,
                "results_checked" to results,
                "violations" to violations,
                "modes" to modes,
                "updates_applied" to eng.applier.counters(),
                "elapsed_s" to (System.nanoTime() - t0) / 1e9,
                "examples" to examples,
            ) + provenance(bundle, 1, loadStart)
            a.pathOrNull("out")?.let { JsonlWriter(it).use { w -> w.write(summary) } }
            println(Json.write(summary - "machine" - "load_start" - "load_end"))
            if (violations > 0) System.err.println("FAILED: $violations invariant violations")
            return if (violations > 0) 1 else 0
        }
    }
}

/**
 * parity: runs golden/parity_queries.jsonl through ONNX Runtime Java and compares with the qemb
 * that PyTorch produced. Fails unless the max absolute difference is below --tol (1e-4).
 */
object Parity {
    fun run(a: Args): Int {
        val bundle = Bundle.load(a.path("bundle"))
        val tol = a.double("tol", 1e-4)
        val loadStart = Machine.load()
        val rows = Json.readLines(bundle.path("golden/parity_queries.jsonl")).toList()
        require(rows.isNotEmpty()) { "golden/parity_queries.jsonl is empty" }
        val dim = bundle.manifest.dim
        val qd = bundle.manifest.qFeatDim
        var maxDiff = 0.0
        var sumDiff = 0.0
        var count = 0L
        Models(bundle).use { models ->
            for (chunk in rows.chunked(256)) {
                val places = LongArray(chunk.size) { chunk[it]["place_idx"].asLong() }
                val qf = FloatArray(chunk.size * qd)
                chunk.forEachIndexed { i, r -> for (j in 0 until qd) qf[i * qd + j] = r["qfeat"][j].floatValue() }
                val out = models.embedQueries(places, qf)
                chunk.forEachIndexed { i, r ->
                    for (j in 0 until dim) {
                        val d = abs(out[i * dim + j].toDouble() - r["qemb"][j].asDouble())
                        maxDiff = maxOf(maxDiff, d); sumDiff += d; count++
                    }
                }
            }
        }
        val pass = maxDiff < tol
        val summary = linkedMapOf<String, Any?>(
            "check" to "onnx_parity", "queries" to rows.size, "max_abs_diff" to maxDiff,
            "mean_abs_diff" to sumDiff / count, "tolerance" to tol, "pass" to pass,
            "onnxruntime" to ai.onnxruntime.OrtEnvironment.getEnvironment().version,
        ) + provenance(bundle, 1, loadStart)
        a.pathOrNull("out")?.let { JsonlWriter(it).use { w -> w.write(summary) } }
        println(Json.write(summary - "machine" - "load_start" - "load_end"))
        return if (pass) 0 else 1
    }
}
