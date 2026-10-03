package staysearch.cli

import org.HdrHistogram.Histogram
import staysearch.Engine
import staysearch.EngineConfig
import staysearch.bundle.Bundle
import staysearch.data.ListingTable
import staysearch.search.Features
import staysearch.search.SearchRequest
import staysearch.search.StageStats
import staysearch.util.Json
import staysearch.util.JsonlWriter
import staysearch.util.Machine

/**
 * flexdates (M6 b): what plus or minus 3 days on check-in and on check-out (49 windows) buys and costs.
 *
 * Three parts, one JSON line each:
 * 1. `eligible`: for each of N test queries, the eligible set for the exact dates (live, inside the
 *    place's box, fits the guests, nights within min/max nights, every night open) vs the union over
 *    the 49 windows (a window counts only if it has at least 1 night and its length is within that
 *    listing's min/max nights). Brute force over the table, independent of Lucene. Reports the
 *    growth factor (flexible / exact) over queries with a non-empty exact set, and how many queries
 *    go from zero results to some.
 * 2. `latency`: one flexible search (a single pass whose availability filter is the 49-window mask)
 *    vs 49 separate exact-date searches merged by id (best ranker score, else smallest distance),
 *    sequential, same JVM, JIT warmed by untimed rounds of both. p50/p99 in microseconds.
 * 3. `check_ns`: nanoseconds per listing for one 49-window check (`flexibleMask`, all windows under
 *    one seqlock read) vs 49 single-window checks (`isAvailable` per window, each its own read),
 *    over every live listing for the query dates. A System.nanoTime loop (not JMH): untimed warm-up
 *    rounds, then timed rounds alternating the order of the two methods, median round reported.
 *    Both methods must agree on the number of open windows; the line says whether they did.
 *
 * Options: --bundle DIR --out FILE [--n 2000] [--mode ivf] [--ranker dnn] [--k 500] [--t 25]
 * [--nprobe 16] [--latency-n 200] [--warmup 50] [--check-queries 200] [--check-rounds 7] [--repeat 1]
 */
object FlexDates {
    const val FLEX = 3

    fun run(a: Args): Int {
        val bundle = Bundle.load(a.path("bundle"))
        val loadStart = Machine.load()
        val mode = a.str("mode", "ivf")
        val queries = loadQueries(bundle, a.int("n", 2000)) {
            copy(mode = mode, ranker = a.str("ranker", "dnn"), k = a.int("k", 500), t = a.int("t", 25), nprobe = a.int("nprobe", 16))
        }
        require(queries.isNotEmpty()) { "no queries in ${bundle.queriesPath}" }
        val repeat = a.int("repeat", 1)
        Engine.open(bundle.dir, EngineConfig(withVectors = mode == "hnsw", refreshMs = 0), bundle).use { eng ->
            JsonlWriter(a.path("out")).use { w ->
                fun emit(row: Map<String, Any?>) {
                    val full = linkedMapOf<String, Any?>("experiment" to "m6b_flexdates", "flex_days" to FLEX) + row + provenance(bundle, repeat, loadStart)
                    w.write(full)
                    println(Json.write(full.filterKeys { it !in setOf("machine", "load_start", "load_end") }))
                }
                emit(eligible(bundle, eng.table, queries))
                emit(latency(eng, queries, a.int("latency-n", 200), a.int("warmup", 50)))
                emit(checkNs(bundle, eng.table, queries, a.int("check-queries", 200), a.int("check-rounds", 7)))
            }
        }
        return 0
    }

    /** All (checkin, checkout) night pairs of the flexible search, including invalid ones (nights < 1). */
    fun windows(checkin: Int, checkout: Int): List<Pair<Int, Int>> =
        (-FLEX..FLEX).flatMap { i -> (-FLEX..FLEX).map { j -> (checkin + i) to (checkout + j) } }

    private fun percentile(sorted: DoubleArray, p: Double): Double? {
        if (sorted.isEmpty()) return null
        val idx = Math.ceil(p / 100.0 * sorted.size).toInt().coerceIn(1, sorted.size) - 1
        return sorted[idx]
    }

    // ------------------------------------------------------------------ part 1: eligible-set growth

    private fun eligible(bundle: Bundle, table: ListingTable, queries: List<SearchRequest>): Map<String, Any?> {
        val growth = ArrayList<Double>()
        val exactSizes = ArrayList<Double>()
        val flexSizes = ArrayList<Double>()
        var zeroExact = 0
        var rescued = 0
        var validWindows = 0L
        for (q in queries) {
            val ctx = Features.resolve(q, bundle)
            var exact = 0
            var flex = 0
            for (r in 0 until table.size) {
                if (table.isRemoved(r) || table.accommodates[r] < ctx.guests || !ctx.box.contains(table.lat[r], table.lng[r])) continue
                val minN = table.minNights[r]
                val maxN = table.maxNights[r]
                if (ctx.nights in minN..maxN && table.store.isAvailable(r, ctx.checkinNight, ctx.checkoutNight)) exact++
                if (table.store.flexibleMask(r, ctx.checkinNight, ctx.checkoutNight, FLEX, minN, maxN) != 0L) flex++
            }
            validWindows += windows(ctx.checkinNight, ctx.checkoutNight).count { (x, y) -> y - x >= 1 }
            check(flex >= exact) { "flexible set smaller than exact set for $q" }
            exactSizes += exact.toDouble(); flexSizes += flex.toDouble()
            if (exact == 0) {
                zeroExact++
                if (flex > 0) rescued++
            } else growth += flex.toDouble() / exact
        }
        val g = growth.toDoubleArray().also { it.sort() }
        val e = exactSizes.toDoubleArray().also { it.sort() }
        val f = flexSizes.toDoubleArray().also { it.sort() }
        return linkedMapOf(
            "part" to "eligible",
            "queries" to queries.size,
            "definition" to "live, in place box, accommodates >= guests, nights within min/max, all nights open; flexible = union over windows with nights >= 1 and within each listing's min/max",
            "mean_valid_windows_per_query" to validWindows.toDouble() / queries.size,
            "exact_size_p50" to percentile(e, 50.0), "exact_size_p90" to percentile(e, 90.0), "exact_size_mean" to e.average(),
            "flex_size_p50" to percentile(f, 50.0), "flex_size_p90" to percentile(f, 90.0), "flex_size_mean" to f.average(),
            "growth_queries" to g.size,
            "growth_p50" to percentile(g, 50.0),
            "growth_p90" to percentile(g, 90.0),
            "growth_mean" to if (g.isEmpty()) null else g.average(),
            "growth_max" to g.lastOrNull(),
            "queries_exact_empty" to zeroExact,
            "queries_exact_empty_flex_nonempty" to rescued,
        )
    }

    // ------------------------------------------------------------------ part 2: search latency

    /** 49 exact-date searches, merged: best score (or smallest distance) per id, top t. Returns the merged ids. */
    private fun separate(eng: Engine, q: SearchRequest): List<Long> {
        val best = HashMap<Long, Float>()
        val byScore = q.ranker != "none"
        for (i in -FLEX..FLEX) for (j in -FLEX..FLEX) {
            val ci = q.checkin.plusDays(i.toLong())
            val co = q.checkout.plusDays(j.toLong())
            if (!co.isAfter(ci)) continue
            val resp = eng.pipeline.search(q.copy(checkin = ci, checkout = co, flexible = false), record = false)
            for (r in resp.results) {
                val v = if (byScore) r.score ?: 0f else -r.distance
                val old = best[r.id]
                if (old == null || v > old) best[r.id] = v
            }
        }
        return best.entries.sortedWith(compareByDescending<Map.Entry<Long, Float>> { it.value }.thenBy { it.key }).take(q.t).map { it.key }
    }

    private fun latency(eng: Engine, all: List<SearchRequest>, n: Int, warmup: Int): Map<String, Any?> {
        val queries = all.take(n.coerceAtLeast(1))
        // Warm-up: untimed rounds of both paths so C2 has compiled the pipeline before anything is recorded.
        for (i in 0 until warmup) {
            val q = all[i % all.size]
            eng.pipeline.search(q.copy(flexible = true), record = false)
            separate(eng, q)
        }
        val one = Histogram(3)
        val many = Histogram(3)
        var searches49 = 0L
        var overlapSum = 0.0
        var overlapN = 0
        for ((i, q) in queries.withIndex()) {
            // Alternate which path goes first so neither always runs on a warmer cache.
            var flexIds: List<Long> = emptyList()
            var sepIds: List<Long> = emptyList()
            val runFlex = {
                val t0 = System.nanoTime()
                flexIds = eng.pipeline.search(q.copy(flexible = true), record = false).results.map { it.id }
                one.recordValue((System.nanoTime() - t0) / 1000)
            }
            val runSep = {
                val t0 = System.nanoTime()
                sepIds = separate(eng, q)
                many.recordValue((System.nanoTime() - t0) / 1000)
            }
            if (i % 2 == 0) { runFlex(); runSep() } else { runSep(); runFlex() }
            searches49 += windows(0, q.checkout.toEpochDay().toInt() - q.checkin.toEpochDay().toInt()).count { (x, y) -> y - x >= 1 }
            val union = (flexIds + sepIds).toSet().size
            if (union > 0) {
                overlapSum += flexIds.toSet().intersect(sepIds.toSet()).size.toDouble() / union; overlapN++
            }
        }
        val q0 = queries.first()
        return linkedMapOf(
            "part" to "latency",
            "queries" to queries.size,
            "warmup_queries" to warmup,
            "mode" to q0.mode, "ranker" to q0.ranker, "k" to q0.k, "t" to q0.t, "nprobe" to q0.nprobe,
            "generator" to "in-process, sequential, one thread, same JVM",
            "flexible_one_pass_us" to StageStats.summarize(one),
            "separate_49_us" to StageStats.summarize(many),
            "separate_searches_per_query_mean" to searches49.toDouble() / queries.size,
            "p50_ratio_separate_over_flexible" to many.getValueAtPercentile(50.0).toDouble() / one.getValueAtPercentile(50.0).coerceAtLeast(1),
            "p99_ratio_separate_over_flexible" to many.getValueAtPercentile(99.0).toDouble() / one.getValueAtPercentile(99.0).coerceAtLeast(1),
            "top_t_jaccard_mean" to if (overlapN == 0) null else overlapSum / overlapN,
            "note" to "separate_49 merges results by id (best ranker score, else smallest distance); the two paths rank different candidate sets so jaccard < 1 is expected",
        )
    }

    // ------------------------------------------------------------------ part 3: ns per listing

    private class Win(val ci: Int, val co: Int)

    private fun runMask(table: ListingTable, rows: IntArray, w: List<Win>): Long {
        var acc = 0L
        for (x in w) for (r in rows) acc += table.store.flexibleMask(r, x.ci, x.co, FLEX, table.minNights[r], table.maxNights[r]).countOneBits()
        return acc
    }

    private fun runSeparate(table: ListingTable, rows: IntArray, w: List<Win>): Long {
        var acc = 0L
        for (x in w) for (r in rows) {
            val minN = table.minNights[r]
            val maxN = table.maxNights[r]
            for (i in -FLEX..FLEX) for (j in -FLEX..FLEX) {
                val a = x.ci + i
                val b = x.co + j
                val n = b - a
                if (n < 1 || n < minN || n > maxN) continue
                if (table.store.isAvailable(r, a, b)) acc++
            }
        }
        return acc
    }

    private fun checkNs(bundle: Bundle, table: ListingTable, all: List<SearchRequest>, nq: Int, rounds: Int): Map<String, Any?> {
        val wins = all.take(nq.coerceAtLeast(1)).map { q -> Features.resolve(q, bundle).let { Win(it.checkinNight, it.checkoutNight) } }
        val rows = (0 until table.size).filter { !table.isRemoved(it) }.toIntArray()
        val evals = wins.size.toLong() * rows.size
        val warmWins = wins.take(maxOf(1, wins.size / 4))
        var sink = 0L
        repeat(3) { sink += runMask(table, rows, warmWins) + runSeparate(table, rows, warmWins) }
        val maskNs = ArrayList<Double>()
        val sepNs = ArrayList<Double>()
        var maskCount = -1L
        var sepCount = -1L
        repeat(rounds.coerceAtLeast(1)) { round ->
            fun timeMask() {
                val t0 = System.nanoTime(); val c = runMask(table, rows, wins); maskNs += (System.nanoTime() - t0).toDouble() / evals; maskCount = c
            }
            fun timeSep() {
                val t0 = System.nanoTime(); val c = runSeparate(table, rows, wins); sepNs += (System.nanoTime() - t0).toDouble() / evals; sepCount = c
            }
            if (round % 2 == 0) { timeMask(); timeSep() } else { timeSep(); timeMask() }
        }
        val m = maskNs.toDoubleArray().also { it.sort() }
        val s = sepNs.toDoubleArray().also { it.sort() }
        return linkedMapOf(
            "part" to "check_ns",
            "method" to "System.nanoTime loop (not JMH): 3 untimed warm-up rounds over a quarter of the windows, then $rounds timed rounds over all, order alternating, median round",
            "queries" to wins.size,
            "listings" to rows.size,
            "listing_checks_per_round" to evals,
            "mask49_ns_per_listing_median" to percentile(m, 50.0),
            "mask49_ns_per_listing_min" to m.first(),
            "separate49_ns_per_listing_median" to percentile(s, 50.0),
            "separate49_ns_per_listing_min" to s.first(),
            "speedup_median" to percentile(s, 50.0)!! / percentile(m, 50.0)!!,
            "open_windows_mask" to maskCount,
            "open_windows_separate" to sepCount,
            "methods_agree" to (maskCount == sepCount),
            "sink" to sink % 2,
        )
    }
}
