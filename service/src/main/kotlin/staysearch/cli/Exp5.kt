package staysearch.cli

import org.openjdk.jmh.runner.Runner
import org.openjdk.jmh.runner.options.OptionsBuilder
import org.openjdk.jmh.runner.options.TimeValue
import staysearch.bench.AvailabilityBench
import staysearch.util.Json
import staysearch.util.JsonlWriter
import staysearch.util.Machine
import java.nio.file.Paths

/**
 * exp5: runs the [AvailabilityBench] JMH benchmarks and writes one JSON line per benchmark.
 *
 * Options: --out FILE [--bundle DIR] (calendars from the bundle; synthetic otherwise)
 * [--forks 2] [--warmup-iterations 5] [--iterations 10] [--iteration-seconds 1] [--max-nights 7,14,28] [--synthetic-listings 20000]
 */
object Exp5 {
    fun run(a: Args): Int {
        val loadStart = Machine.load()
        val bundleDir = a.str("bundle")
        val asOf = bundleDir?.let { Json.mapper.readTree(Paths.get(it, "manifest.json").toFile())["as_of"].asText() }
        val synthetic = if (bundleDir == null) true
        else Json.mapper.readTree(Paths.get(bundleDir, "manifest.json").toFile())["synthetic"]?.asBoolean() ?: false
        val secs = a.int("iteration-seconds", 1)
        val opts = OptionsBuilder()
            .include(AvailabilityBench::class.java.name.replace(".", "\\.") + "\\.")
            .param("bundle", bundleDir?.let { Paths.get(it).toAbsolutePath().toString() } ?: "")
            .param("maxNights", *a.list("max-nights", "7,14,28").toTypedArray())
            .param("syntheticListings", a.str("synthetic-listings", "20000"))
            .forks(a.int("forks", 2))
            .warmupIterations(a.int("warmup-iterations", 5))
            .measurementIterations(a.int("iterations", 10))
            .warmupTime(TimeValue.seconds(secs.toLong()))
            .measurementTime(TimeValue.seconds(secs.toLong()))
            .jvmArgsAppend("-Xmx${a.str("fork-heap", "4g")}")
            .build()
        val results = Runner(opts).run()
        JsonlWriter(a.path("out")).use { w ->
            for (r in results) {
                val p = r.params
                val pr = r.primaryResult
                val row = linkedMapOf<String, Any?>(
                    "experiment" to "exp5_availability",
                    "benchmark" to p.benchmark.substringAfterLast('.'),
                    "method" to if (p.benchmark.contains("runLength")) "run_length" else "per_night_scan",
                    "windows" to if (p.benchmark.contains("Flexible49")) 49 else 1,
                    "ns_per_op" to pr.score,
                    "ns_per_op_error_99_9" to pr.scoreError.takeIf { it.isFinite() },
                    "unit" to pr.scoreUnit,
                    "samples" to pr.statistics.n,
                    "forks" to p.forks,
                    "warmup_iterations" to p.warmup.count,
                    "measurement_iterations" to p.measurement.count,
                    "max_nights" to p.getParam("maxNights").toInt(),
                    "calendars" to if (bundleDir == null) "synthetic (${p.getParam("syntheticListings")} listings)" else "bundle",
                    "jmh_version" to org.openjdk.jmh.util.Version.getPlainVersion(),
                    "machine" to Machine.info,
                    "load_start" to loadStart,
                    "load_end" to Machine.load(),
                    "repeat" to 1,
                    "bundle_as_of" to asOf,
                    "bundle_synthetic" to synthetic,
                )
                w.write(row)
                println(Json.write(row - "machine" - "load_start" - "load_end"))
            }
        }
        return 0
    }
}
