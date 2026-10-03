package staysearch.search

import staysearch.avail.AvailabilityStore
import staysearch.bundle.Box
import staysearch.bundle.Bundle
import staysearch.bundle.GuestHistory
import staysearch.bundle.Similarity
import staysearch.data.ListingTable
import java.time.LocalDate
import java.time.temporal.ChronoUnit
import kotlin.math.asin
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.ln1p
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/** A search request (HTTP body of /search and the in-process API). Dates are ISO yyyy-mm-dd. */
data class SearchRequest(
    val placeIdx: Int? = null,
    val bbox: Box? = null,
    val checkin: LocalDate,
    val checkout: LocalDate,
    val guests: Int = 1,
    val guestId: Long? = null,
    val searchDate: LocalDate? = null,
    /** Retrieval top K. Null means the default: the manifest's ranker_rank_k (100) with a ranker, 500 without. */
    val k: Int? = null,
    val t: Int = 25,
    val mode: String = "ivf",
    val nprobe: Int = 16,
    val ranker: String = "dnn",
    val flexible: Boolean = false,
)

/**
 * Everything the features need about one request, resolved once.
 *
 * Search area: a place search uses the place's bbox, its (lat, lng) as the centre and its area_km2.
 * A bbox search uses the bbox, its centre and planar area, and the nearest city place (haversine
 * from the bbox centre) as the tower's place. If both are given, the bbox defines the area and the
 * given place is the tower's place. Feature 1 (whole city) is 1 only for a place search on a city.
 */
class QueryContext(
    val placeIdx: Int,
    val isCity: Boolean,
    val areaKm2: Double,
    val centreLat: Double,
    val centreLng: Double,
    val box: Box,
    val guests: Int,
    val nights: Int,
    val checkinNight: Int,
    val checkoutNight: Int,
    val checkinDate: LocalDate,
    val leadDays: Long,
    val guestBlock: FloatArray,
    val hasHistory: Boolean,
)

object Features {
    const val KM_PER_DEG_LAT = 111.32
    const val EARTH_RADIUS_KM = 6371.0
    const val N_CROSS = 15

    /** Gap features 12 and 13 cap the open-night gap at this many nights. */
    const val GAP_CAP = 28
    private val LN31 = ln(31.0)
    private val LN_GAP = ln1p(GAP_CAP.toDouble())
    private val LN366 = ln(366.0)

    fun haversineKm(lat1: Double, lng1: Double, lat2: Double, lng2: Double): Double {
        val p1 = Math.toRadians(lat1)
        val p2 = Math.toRadians(lat2)
        val dp = p2 - p1
        val dl = Math.toRadians(lng2 - lng1)
        val a = sin(dp / 2).let { it * it } + cos(p1) * cos(p2) * sin(dl / 2).let { it * it }
        return 2 * EARTH_RADIUS_KM * asin(min(1.0, sqrt(a)))
    }

    fun bboxAreaKm2(b: Box): Double {
        val h = (b.maxLat - b.minLat) * KM_PER_DEG_LAT
        val w = (b.maxLng - b.minLng) * KM_PER_DEG_LAT * cos(Math.toRadians(b.centreLat()))
        return kotlin.math.abs(h * w)
    }

    fun resolve(req: SearchRequest, bundle: Bundle): QueryContext {
        val places = bundle.places
        require(req.guests >= 1) { "guests must be >= 1" }
        require(req.checkout.isAfter(req.checkin)) { "checkout must be after checkin" }
        val placeIdx: Int
        val isCity: Boolean
        val area: Double
        val cLat: Double
        val cLng: Double
        val box: Box
        if (req.bbox == null) {
            val pi = requireNotNull(req.placeIdx) { "place_idx or bbox is required" }
            require(pi in places.indices) { "place_idx $pi out of range" }
            val p = places[pi]
            placeIdx = pi; isCity = p.isCity; area = p.areaKm2; cLat = p.lat; cLng = p.lng; box = p.box
        } else {
            box = req.bbox
            require(box.minLat <= box.maxLat && box.minLng <= box.maxLng) { "bbox min must be <= max" }
            cLat = box.centreLat(); cLng = box.centreLng()
            area = bboxAreaKm2(box)
            isCity = false
            placeIdx = req.placeIdx?.also { require(it in places.indices) { "place_idx $it out of range" } }
                ?: places.filter { it.isCity }.minByOrNull { haversineKm(cLat, cLng, it.lat, it.lng) }?.idx
                ?: error("bundle has no city places")
        }
        val base = bundle.manifest.baseDate
        val searchDate = req.searchDate ?: bundle.manifest.asOf
        val block = FloatArray(GuestHistory.WIDTH)
        val found = bundle.guests.copyInto(req.guestId, block, 0)
        return QueryContext(
            placeIdx = placeIdx, isCity = isCity, areaKm2 = area, centreLat = cLat, centreLng = cLng, box = box,
            guests = req.guests,
            nights = ChronoUnit.DAYS.between(req.checkin, req.checkout).toInt(),
            checkinNight = ChronoUnit.DAYS.between(base, req.checkin).toInt(),
            checkoutNight = ChronoUnit.DAYS.between(base, req.checkout).toInt(),
            checkinDate = req.checkin,
            leadDays = ChronoUnit.DAYS.between(searchDate, req.checkin),
            guestBlock = block,
            hasHistory = found && block[0] > 0.5f,
        )
    }

    /** Lead-time bucket: [<=1], [2,7], [8,14], [15,30], [31,60], [>=61]. */
    fun leadBucket(lead: Long): Int = when {
        lead <= 1 -> 0
        lead <= 7 -> 1
        lead <= 14 -> 2
        lead <= 30 -> 3
        lead <= 60 -> 4
        else -> 5
    }

    /** The query tower's qfeat vector (width q_feat_dim, 87 in the contract). */
    fun queryFeatures(ctx: QueryContext, width: Int = 87): FloatArray {
        val q = FloatArray(width)
        q[0] = (ln1p(ctx.areaKm2) / 10.0).toFloat()
        q[1] = if (ctx.isCity) 1f else 0f
        q[2] = (ctx.guests / 16.0).toFloat()
        q[3] = (ln1p(ctx.nights.toDouble()) / LN31).toFloat()
        q[4 + leadBucket(ctx.leadDays)] = 1f
        q[10 + ctx.checkinDate.dayOfWeek.value - 1] = 1f // Monday first
        System.arraycopy(ctx.guestBlock, 0, q, 17, min(GuestHistory.WIDTH, width - 17))
        return q
    }

    /**
     * Writes the ranker row for [row]: ranker_static then the 15 cross features, into [out] at [offset].
     * [distance] is the retrieval distance (squared L2, or minus the dot product). [rank] is the
     * candidate's 0-based position in the retrieval top K (distance ascending); feature 14 clamps it
     * to [0, rankK], where [rankK] is the K the rankers were trained with (manifest `ranker_rank_k`).
     * Gap features 12 and 13 read the availability store's current state.
     */
    fun rankerRow(ctx: QueryContext, table: ListingTable, row: Int, distance: Float, rank: Int, rankK: Int, out: FloatArray, offset: Int) {
        val sd = table.staticDim
        System.arraycopy(table.static, row * sd, out, offset, sd)
        val o = offset + sd
        val price = table.price[row]
        val g = ctx.guestBlock
        val h = ctx.hasHistory
        out[o + 0] = distance
        out[o + 1] = (haversineKm(table.lat[row], table.lng[row], ctx.centreLat, ctx.centreLng) / sqrt(1.0 + ctx.areaKm2)).toFloat()
        out[o + 2] = if (price > 0) (ln1p(price) - ln(ctx.guests.toDouble())).toFloat() else 0f
        out[o + 3] = ln1p(max(0, ctx.nights - table.minNights[row]).toDouble()).toFloat()
        out[o + 4] = (ctx.guests.toDouble() / table.accommodates[row]).toFloat()
        out[o + 5] = if (h) {
            var acc = 0.0
            val eo = row * table.dim
            for (i in 0 until table.dim) acc += g[6 + i].toDouble() * table.emb[eo + i].toDouble()
            acc.toFloat()
        } else 0f
        out[o + 6] = if (h && price > 0) ((ln1p(price) - 5.0) / 1.0 - g[1].toDouble()).toFloat() else 0f
        out[o + 7] = if (h) 1f else 0f
        out[o + 8] = (ln1p(max(0L, ctx.leadDays).toDouble()) / LN366).toFloat()
        out[o + 9] = (ln1p(ctx.nights.toDouble()) / LN31).toFloat()
        out[o + 10] = (ctx.guests / 16.0).toFloat()
        out[o + 11] = if (h) g[2 + table.roomType[row].coerceIn(0, 3)] else 0f
        val gaps = table.store.gaps(row, ctx.checkinNight, ctx.checkoutNight, GAP_CAP)
        out[o + 12] = (ln1p(AvailabilityStore.gapBeforeOf(gaps).toDouble()) / LN_GAP).toFloat()
        out[o + 13] = (ln1p(AvailabilityStore.gapAfterOf(gaps).toDouble()) / LN_GAP).toFloat()
        out[o + 14] = (ln1p(rank.coerceIn(0, rankK).toDouble()) / ln1p(rankK.toDouble())).toFloat()
    }

    /** Retrieval distance computed in double precision (for the golden test's tolerance). */
    fun distanceDouble(q: FloatArray, table: ListingTable, row: Int): Double {
        var acc = 0.0
        val eo = row * table.dim
        for (i in 0 until table.dim) {
            if (table.similarity == Similarity.L2) {
                val d = q[i].toDouble() - table.emb[eo + i].toDouble(); acc += d * d
            } else acc -= q[i].toDouble() * table.emb[eo + i].toDouble()
        }
        return acc
    }
}
