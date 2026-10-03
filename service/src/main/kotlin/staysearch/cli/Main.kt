package staysearch.cli

import kotlin.system.exitProcess

private val USAGE = """
    usage: staysearch <command> [--option value ...]

    commands:
      serve             HTTP service (/search, /health, /meta, /places, /sample-guests, /listing/{id}, /stats, static files)
      replay-invariant  run test queries (optionally with updates) and fail on any invariant violation
      parity            ONNX Runtime Java vs golden/parity_queries.jsonl
      exp4              IVF vs HNSW in Lucene under the replayed update stream
      exp5              JMH: run-length availability check vs per-night scan
      latency | exp9    open-loop latency at constant arrival rates, with and without updates
      flexdates         M6(b): +/-3 day flexible dates, eligible-set growth, latency and check cost vs 49 windows

    every command takes --bundle DIR; see README.md for the rest.
""".trimIndent()

fun main(argv: Array<String>) {
    if (argv.isEmpty() || argv[0] in setOf("-h", "--help", "help")) {
        println(USAGE); return
    }
    val args = Args(argv.drop(1))
    val code = try {
        when (argv[0]) {
            "serve" -> Serve.run(args)
            "replay-invariant" -> ReplayInvariant.run(args)
            "parity" -> Parity.run(args)
            "exp4" -> Exp4.run(args)
            "exp5" -> Exp5.run(args)
            "latency", "exp9" -> Latency.run(args)
            "flexdates" -> FlexDates.run(args)
            else -> {
                System.err.println("unknown command '${argv[0]}'\n$USAGE"); 2
            }
        }
    } catch (e: IllegalArgumentException) {
        System.err.println("error: ${e.message}")
        2
    }
    if (argv[0] != "serve") exitProcess(code)
}
