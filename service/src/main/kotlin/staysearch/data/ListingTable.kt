package staysearch.data

import staysearch.avail.AvailabilityStore
import staysearch.bundle.Bundle
import staysearch.bundle.ListingRecord
import staysearch.bundle.Similarity
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger

/**
 * The serving copy of every listing, row-addressed (row = line number in the bundle, then appended
 * rows for listings that arrive in the update stream). Holds the attributes, the vectors (the IVF
 * path's "float arrays keyed by row"), the ranker's static rows and the availability store.
 *
 * Capacity is fixed at construction (n_listings plus room for new listings); appending past it fails.
 * Mutation happens only on the single update thread. Attribute fields other than price are written
 * once before a row becomes reachable. Price is a plain field: a reader may see the old or new quote
 * for a moment, which only affects ranking features and the displayed price.
 */
class ListingTable(
    val dim: Int,
    val staticDim: Int,
    val horizon: Int,
    val capacity: Int,
    val similarity: Similarity,
    val centroids: FloatArray,
) {
    val ids = LongArray(capacity)
    val lat = DoubleArray(capacity)
    val lng = DoubleArray(capacity)
    val accommodates = IntArray(capacity)
    val minNights = IntArray(capacity)
    val maxNights = IntArray(capacity)
    val roomType = IntArray(capacity)
    val placeCity = IntArray(capacity)
    val placeNbhd = IntArray(capacity)
    val cluster = IntArray(capacity)
    val price = DoubleArray(capacity)

    private val removedFlags = BooleanArray(capacity)
    val emb = FloatArray(Math.multiplyExact(capacity, dim))
    val static = FloatArray(Math.multiplyExact(capacity, staticDim))
    val store = AvailabilityStore(horizon, capacity)
    private val idToRow = ConcurrentHashMap<Long, Int>()
    private val sizeCounter = AtomicInteger(0)

    val nClusters = centroids.size / dim

    /** Rows allocated so far (bundle rows plus appended ones). */
    val size: Int get() = sizeCounter.get()

    fun rowOf(id: Long): Int? = idToRow[id]
    fun isRemoved(row: Int) = removedFlags[row]
    val liveCount: Int get() = idToRow.size

    fun record(row: Int) = ListingRecord(
        ids[row], "", placeCity[row], placeNbhd[row], lat[row], lng[row], accommodates[row],
        minNights[row], maxNights[row], price[row], roomType[row],
    )

    private fun setAttributes(row: Int, l: ListingRecord) {
        ids[row] = l.id; lat[row] = l.lat; lng[row] = l.lng
        accommodates[row] = l.accommodates; minNights[row] = l.minNights; maxNights[row] = l.maxNights
        roomType[row] = l.roomType; placeCity[row] = l.placeCity; placeNbhd[row] = l.placeNbhd
        price[row] = l.price
    }

    /**
     * Inserts or replaces a listing. A known id is overwritten in place (same row); an unknown id gets
     * a new row. [clusterId] null means "assign the nearest centroid" (the cold-start path).
     */
    fun upsert(l: ListingRecord, vector: FloatArray, staticRow: FloatArray, packedAvail: ByteArray, clusterId: Int? = null): Int {
        require(vector.size == dim) { "embedding has ${vector.size} floats, expected $dim" }
        require(staticRow.size == staticDim) { "static row has ${staticRow.size} floats, expected $staticDim" }
        val existing = idToRow[l.id]
        val row = existing ?: sizeCounter.get().also {
            check(it < capacity) { "listing table is full ($capacity rows); raise the capacity" }
        }
        setAttributes(row, l)
        System.arraycopy(vector, 0, emb, row * dim, dim)
        System.arraycopy(staticRow, 0, static, row * staticDim, staticDim)
        cluster[row] = clusterId ?: nearestCentroids(vector, 0, 1)[0]
        removedFlags[row] = false
        store.load(row, packedAvail)
        if (existing == null) {
            sizeCounter.incrementAndGet() // volatile write: publishes the row's arrays
            idToRow[l.id] = row
        }
        return row
    }

    /** Removes a listing: closes every night (so availability filtering drops it at once) and forgets the id. */
    fun remove(id: Long): Int? {
        val row = idToRow.remove(id) ?: return null
        removedFlags[row] = true
        store.clear(row)
        return row
    }

    /** Distance in the bundle's convention, smaller is better: squared L2, or minus the dot product. */
    fun distance(q: FloatArray, row: Int): Float = vecDistance(q, emb, row * dim, dim, similarity)

    /** The [nprobe] nearest centroids to [q] under the bundle's similarity, nearest first. */
    fun nearestCentroids(q: FloatArray, qOffset: Int = 0, nprobe: Int): IntArray {
        val k = nprobe.coerceIn(1, nClusters)
        val d = FloatArray(nClusters) { c ->
            vecDistance(q, centroids, c * dim, dim, similarity, qOffset)
        }
        return (0 until nClusters).sortedBy { d[it] }.take(k).toIntArray()
    }

    companion object {
        /** Builds the table from a bundle with [extraCapacity] spare rows for new listings. */
        fun fromBundle(b: Bundle, extraCapacity: Int): ListingTable {
            val m = b.manifest
            val t = ListingTable(m.dim, m.listingStaticDim, m.horizonDays, m.nListings + extraCapacity, m.similarity, b.centroids)
            val bpl = b.bytesPerListing
            for ((i, l) in b.listings.withIndex()) {
                t.setAttributes(i, l)
                t.cluster[i] = b.clusterIds[i]
                t.store.load(i, b.availability, i * bpl)
                t.idToRow[l.id] = i
            }
            System.arraycopy(b.embeddings, 0, t.emb, 0, b.embeddings.size)
            System.arraycopy(b.rankerStatic, 0, t.static, 0, b.rankerStatic.size)
            t.sizeCounter.set(m.nListings)
            return t
        }
    }
}

fun vecDistance(q: FloatArray, data: FloatArray, offset: Int, dim: Int, sim: Similarity, qOffset: Int = 0): Float {
    var acc = 0f
    if (sim == Similarity.L2) {
        for (i in 0 until dim) {
            val d = q[qOffset + i] - data[offset + i]
            acc += d * d
        }
        return acc
    }
    for (i in 0 until dim) acc += q[qOffset + i] * data[offset + i]
    return -acc
}
