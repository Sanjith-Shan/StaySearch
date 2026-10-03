package staysearch

import staysearch.bundle.Bundle
import staysearch.data.ListingTable
import staysearch.index.ListingIndex
import staysearch.search.InvariantChecker
import staysearch.search.Models
import staysearch.search.SearchPipeline
import staysearch.updates.UpdateApplier
import staysearch.updates.UpdateEvent
import java.nio.file.Files
import java.nio.file.Path

data class EngineConfig(
    /** Where the Lucene index lives; null means a fresh temp directory, deleted on close. */
    val indexDir: Path? = null,
    /** Put vectors in the index (needed for hnsw mode). IVF and exact modes work either way. */
    val withVectors: Boolean = true,
    val refreshMs: Long = 1000,
    val ramBufferMb: Double = 64.0,
    /** Spare table rows for listings that arrive as "new" events; null sizes it from updates.jsonl. */
    val extraCapacity: Int? = null,
    val ortThreads: Int = 1,
    val lagSampleEvery: Int = 100,
    /** Rewrite the Lucene document on availability events too (the index-placement experiment). */
    val reindexOnAvail: Boolean = false,
)

/** A loaded bundle with its table, index, models and pipeline. One instance per index build. */
class Engine private constructor(
    val bundle: Bundle,
    val table: ListingTable,
    val index: ListingIndex,
    val models: Models,
    val config: EngineConfig,
    private val tempIndex: Boolean,
    private val ownsModels: Boolean,
    val buildMs: Long,
) : AutoCloseable {
    val pipeline = SearchPipeline(bundle, table, index, models)
    val invariants = InvariantChecker(bundle, table)
    val applier = UpdateApplier(table, index, config.reindexOnAvail)

    override fun close() {
        if (tempIndex) index.closeAndDelete() else index.close()
        if (ownsModels) models.close()
    }

    companion object {
        fun open(bundleDir: Path, config: EngineConfig = EngineConfig(), bundle: Bundle? = null, models: Models? = null): Engine {
            val b = bundle ?: Bundle.load(bundleDir)
            val extra = config.extraCapacity ?: (2 * UpdateEvent.countNew(b.updatesPath) + 1024)
            val table = ListingTable.fromBundle(b, extra)
            val temp = config.indexDir == null
            val dir = config.indexDir ?: Files.createTempDirectory("staysearch-index-")
            val index = ListingIndex(table, dir, config.withVectors, config.refreshMs, config.ramBufferMb, config.lagSampleEvery)
            val buildMs = index.buildAll()
            index.startRefresher()
            return Engine(b, table, index, models ?: Models(b, config.ortThreads), config, temp, models == null, buildMs)
        }
    }
}
