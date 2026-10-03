package staysearch.index

import org.HdrHistogram.ConcurrentHistogram
import org.apache.lucene.document.Document
import org.apache.lucene.document.DoubleField
import org.apache.lucene.document.Field
import org.apache.lucene.document.IntField
import org.apache.lucene.document.KnnFloatVectorField
import org.apache.lucene.document.LatLonPoint
import org.apache.lucene.document.NumericDocValuesField
import org.apache.lucene.document.StoredField
import org.apache.lucene.document.StringField
import org.apache.lucene.index.ConcurrentMergeScheduler
import org.apache.lucene.index.IndexWriter
import org.apache.lucene.index.IndexWriterConfig
import org.apache.lucene.index.MergePolicy
import org.apache.lucene.index.MergeScheduler
import org.apache.lucene.index.Term
import org.apache.lucene.index.VectorSimilarityFunction
import org.apache.lucene.search.IndexSearcher
import org.apache.lucene.search.ReferenceManager
import org.apache.lucene.search.SearcherFactory
import org.apache.lucene.search.SearcherManager
import org.apache.lucene.store.FSDirectory
import org.apache.lucene.index.IndexReader
import staysearch.bundle.Similarity
import staysearch.data.ListingTable
import staysearch.util.deleteRecursively
import staysearch.util.dirBytes
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledExecutorService
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong

/**
 * The Lucene index over the listing table: one document per live listing.
 *
 * Fields: `id` (keyword term, for updateDocument), `listing_id` (stored), `row` (doc value, the
 * docid to row map used by [AvailabilityQuery] and by result collection), `cluster` (IntField),
 * `geo` (LatLonPoint), `acc`, `min_n`, `max_n` (IntField), `price` (DoubleField), and in HNSW mode
 * `vec` (KnnFloatVectorField; EUCLIDEAN under l2, MAXIMUM_INNER_PRODUCT under dot).
 *
 * The same documents serve IVF mode (cluster id as a filter, vectors in the table) and HNSW mode
 * (vectors in the index). An index built [withVectors] = false can only serve IVF and exact modes.
 *
 * Writes come from one thread. A background thread calls [SearcherManager.maybeRefresh] every
 * [refreshMs]; [lag] measures update-to-visible time for sampled writes.
 */
class ListingIndex(
    private val table: ListingTable,
    val path: Path,
    val withVectors: Boolean,
    private val refreshMs: Long = 1000,
    ramBufferMb: Double = 64.0,
    lagSampleEvery: Int = 100,
) : AutoCloseable {
    val merges = TimedMergeScheduler()
    private val directory = FSDirectory.open(path.also { Files.createDirectories(it) })
    val writer: IndexWriter = IndexWriter(directory, IndexWriterConfig().apply {
        openMode = IndexWriterConfig.OpenMode.CREATE
        ramBufferSizeMB = ramBufferMb
        mergeScheduler = merges
    })
    val lag = VisibilityLag(writer, lagSampleEvery)
    val searcherManager = SearcherManager(writer, object : SearcherFactory() {
        override fun newSearcher(reader: IndexReader, previousReader: IndexReader?): IndexSearcher =
            IndexSearcher(reader).apply { queryCache = null } // AvailabilityQuery results must never be cached
    }).also { it.addListener(lag) }
    private var refresher: ScheduledExecutorService? = null
    val docsWritten = AtomicLong()

    private val vectorSimilarity = when (table.similarity) {
        Similarity.L2 -> VectorSimilarityFunction.EUCLIDEAN
        Similarity.DOT -> VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT
    }

    fun document(row: Int): Document = Document().apply {
        val id = table.ids[row]
        add(StringField(F_ID, id.toString(), Field.Store.NO))
        add(StoredField(F_LISTING_ID, id))
        add(NumericDocValuesField(F_ROW, row.toLong()))
        add(IntField(F_CLUSTER, table.cluster[row], Field.Store.NO))
        add(LatLonPoint(F_GEO, table.lat[row], table.lng[row]))
        add(IntField(F_ACC, table.accommodates[row], Field.Store.NO))
        add(IntField(F_MIN_N, table.minNights[row], Field.Store.NO))
        add(IntField(F_MAX_N, table.maxNights[row], Field.Store.NO))
        add(DoubleField(F_PRICE, table.price[row], Field.Store.NO))
        if (withVectors) {
            val v = table.emb.copyOfRange(row * table.dim, (row + 1) * table.dim)
            add(KnnFloatVectorField(F_VEC, v, vectorSimilarity))
        }
    }

    /** Indexes every live row of the table, commits and opens the first searcher. Returns elapsed ms. */
    fun buildAll(): Long {
        val t0 = System.nanoTime()
        for (row in 0 until table.size) if (!table.isRemoved(row)) writer.addDocument(document(row))
        docsWritten.addAndGet(table.size.toLong())
        writer.commit()
        searcherManager.maybeRefreshBlocking()
        return (System.nanoTime() - t0) / 1_000_000
    }

    /** Re-indexes one row (delete then add; in HNSW mode the vector is inserted into the graph again). */
    fun upsert(row: Int): Long {
        val seq = writer.updateDocument(Term(F_ID, table.ids[row].toString()), document(row))
        docsWritten.incrementAndGet()
        lag.onWrite(seq)
        return seq
    }

    fun delete(id: Long): Long {
        val seq = writer.deleteDocuments(Term(F_ID, id.toString()))
        lag.onWrite(seq)
        return seq
    }

    fun startRefresher() {
        if (refresher != null || refreshMs <= 0) return
        refresher = Executors.newSingleThreadScheduledExecutor { r -> Thread(r, "nrt-refresh").apply { isDaemon = true } }
            .also { it.scheduleWithFixedDelay({ runCatching { searcherManager.maybeRefresh() } }, refreshMs, refreshMs, TimeUnit.MILLISECONDS) }
    }

    fun refreshNow() = searcherManager.maybeRefreshBlocking()

    inline fun <T> withSearcher(body: (IndexSearcher) -> T): T {
        val s = searcherManager.acquire()
        try {
            return body(s)
        } finally {
            searcherManager.release(s)
        }
    }

    val bytesOnDisk: Long get() = dirBytes(path)

    override fun close() {
        refresher?.let { it.shutdownNow(); it.awaitTermination(5, TimeUnit.SECONDS) }
        runCatching { searcherManager.close() }
        try {
            writer.close()
        } finally {
            directory.close()
        }
    }

    /** Closes and deletes the index directory. */
    fun closeAndDelete() {
        try {
            close()
        } finally {
            deleteRecursively(path)
        }
    }

    companion object {
        const val F_ID = "id"
        const val F_LISTING_ID = "listing_id"
        const val F_ROW = "row"
        const val F_CLUSTER = "cluster"
        const val F_GEO = "geo"
        const val F_ACC = "acc"
        const val F_MIN_N = "min_n"
        const val F_MAX_N = "max_n"
        const val F_PRICE = "price"
        const val F_VEC = "vec"
    }
}

/** ConcurrentMergeScheduler that times every merge. */
class TimedMergeScheduler : ConcurrentMergeScheduler() {
    val mergeNanos = AtomicLong()
    val mergeCount = AtomicLong()
    val maxMergeNanos = AtomicLong()

    override fun doMerge(mergeSource: MergeScheduler.MergeSource, merge: MergePolicy.OneMerge) {
        val t0 = System.nanoTime()
        try {
            super.doMerge(mergeSource, merge)
        } finally {
            val dt = System.nanoTime() - t0
            mergeNanos.addAndGet(dt)
            mergeCount.incrementAndGet()
            maxMergeNanos.accumulateAndGet(dt, ::maxOf)
        }
    }

    fun reset() {
        mergeNanos.set(0); mergeCount.set(0); maxMergeNanos.set(0)
    }
}

/**
 * Update-to-visible lag, measured with sequence numbers. One in [sampleEvery] writes records
 * (sequence number, time applied). Before each refresh the listener captures the writer's highest
 * completed sequence number; every write at or below it is in the reader that refresh opens, so after
 * a successful refresh each pending sample up to the captured number is recorded as
 * now minus time applied.
 */
class VisibilityLag(private val writer: IndexWriter, private val sampleEvery: Int) : ReferenceManager.RefreshListener {
    val histogramMicros = ConcurrentHistogram(3)
    private val pending = ArrayDeque<LongArray>()
    private var writes = 0L

    @Volatile
    private var captured = -1L

    fun onWrite(seq: Long) {
        if (writes++ % sampleEvery != 0L) return
        synchronized(pending) { pending.addLast(longArrayOf(seq, System.nanoTime())) }
    }

    override fun beforeRefresh() {
        captured = writer.maxCompletedSequenceNumber
    }

    override fun afterRefresh(didRefresh: Boolean) {
        if (!didRefresh) return
        val now = System.nanoTime()
        synchronized(pending) {
            while (pending.isNotEmpty() && pending.first()[0] <= captured) {
                histogramMicros.recordValue(((now - pending.removeFirst()[1]) / 1000).coerceAtLeast(0))
            }
        }
    }

    fun pendingCount() = synchronized(pending) { pending.size }

    fun reset() {
        histogramMicros.reset()
        synchronized(pending) { pending.clear() }
    }
}
