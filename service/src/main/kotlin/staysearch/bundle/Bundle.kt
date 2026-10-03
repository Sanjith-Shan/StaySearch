package staysearch.bundle

import com.fasterxml.jackson.databind.JsonNode
import staysearch.util.Json
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardOpenOption
import java.time.LocalDate

/** manifest.json, see BUNDLE_FORMAT.md. */
data class Manifest(
    val version: Int,
    val asOf: LocalDate,
    val baseDate: LocalDate,
    val horizonDays: Int,
    val nListings: Int,
    val dim: Int,
    val similarity: Similarity,
    val nClusters: Int,
    val nPlaces: Int,
    val qFeatDim: Int,
    val listingStaticDim: Int,
    val rankerFeatDim: Int,
    /** The retrieval K the rankers were trained with; cross feature 14 clamps the rank at it. */
    val rankerRankK: Int,
    val models: Map<String, String>,
    val cities: List<String>,
    val synthetic: Boolean,
) {
    companion object {
        fun parse(n: JsonNode) = Manifest(
            version = n["version"].asInt(),
            asOf = LocalDate.parse(n["as_of"].asText()),
            baseDate = LocalDate.parse(n["base_date"].asText()),
            horizonDays = n["horizon_days"].asInt(),
            nListings = n["n_listings"].asInt(),
            dim = n["dim"].asInt(),
            similarity = Similarity.parse(n["similarity"].asText()),
            nClusters = n["n_clusters"].asInt(),
            nPlaces = n["n_places"].asInt(),
            qFeatDim = n["q_feat_dim"].asInt(),
            listingStaticDim = n["listing_static_dim"].asInt(),
            rankerFeatDim = n["ranker_feat_dim"].asInt(),
            rankerRankK = (n["ranker_rank_k"]?.takeIf { !it.isNull }?.asInt() ?: 100).also { require(it >= 1) { "ranker_rank_k must be >= 1" } },
            models = n["models"]?.properties()?.associate { it.key to it.value.asText() } ?: emptyMap(),
            cities = n["cities"]?.map { it.asText() } ?: emptyList(),
            synthetic = n["synthetic"]?.asBoolean() ?: false,
        )
    }
}

enum class Similarity {
    L2, DOT;

    companion object {
        fun parse(s: String) = when (s.lowercase()) {
            "l2" -> L2
            "dot" -> DOT
            else -> error("unknown similarity '$s'")
        }
    }
}

/** One line of listings.jsonl (also the `listing` object of a "new" update event). */
data class ListingRecord(
    val id: Long,
    val city: String,
    val placeCity: Int,
    val placeNbhd: Int,
    val lat: Double,
    val lng: Double,
    val accommodates: Int,
    val minNights: Int,
    val maxNights: Int,
    val price: Double,
    val roomType: Int,
) {
    companion object {
        fun parse(n: JsonNode) = ListingRecord(
            id = n["id"].asLong(),
            city = n["city"]?.asText() ?: "",
            placeCity = n["place_city"]?.asInt() ?: -1,
            placeNbhd = n["place_nbhd"]?.asInt() ?: -1,
            lat = n["lat"].asDouble(),
            lng = n["lng"].asDouble(),
            accommodates = n["accommodates"].asInt(),
            minNights = n["min_nights"].asInt(),
            maxNights = n["max_nights"].asInt(),
            price = n["price"]?.asDouble() ?: 0.0,
            roomType = n["room_type"]?.asInt() ?: 0,
        )
    }
}

/** A lat/lng box (inclusive). */
data class Box(val minLat: Double, val minLng: Double, val maxLat: Double, val maxLng: Double) {
    fun centreLat() = (minLat + maxLat) / 2
    fun centreLng() = (minLng + maxLng) / 2
    fun contains(lat: Double, lng: Double) = lat in minLat..maxLat && lng in minLng..maxLng
}

/** One line of places.jsonl. */
data class Place(
    val idx: Int,
    val kind: String,
    val city: String,
    val name: String,
    val lat: Double,
    val lng: Double,
    val box: Box,
    val areaKm2: Double,
) {
    val isCity get() = kind == "city"

    companion object {
        fun parse(n: JsonNode) = Place(
            idx = n["idx"].asInt(),
            kind = n["kind"].asText(),
            city = n["city"]?.asText() ?: "",
            name = n["name"]?.asText() ?: "",
            lat = n["lat"].asDouble(),
            lng = n["lng"].asDouble(),
            box = Box(n["min_lat"].asDouble(), n["min_lng"].asDouble(), n["max_lat"].asDouble(), n["max_lng"].asDouble()),
            areaKm2 = n["area_km2"].asDouble(),
        )
    }
}

/** guest_history.i64 + guest_history.f32: sorted ids and a 70-float row per guest. */
class GuestHistory(val ids: LongArray, val feats: FloatArray) {
    val size get() = ids.size

    /** Copies the guest's 70 floats into [out] at [offset], or zeros if the guest is absent. Returns true if found. */
    fun copyInto(guestId: Long?, out: FloatArray, offset: Int): Boolean {
        val i = if (guestId == null) -1 else ids.binarySearch(guestId)
        if (i < 0) {
            out.fill(0f, offset, offset + WIDTH)
            return false
        }
        System.arraycopy(feats, i * WIDTH, out, offset, WIDTH)
        return true
    }

    companion object {
        const val WIDTH = 70
        val EMPTY = GuestHistory(LongArray(0), FloatArray(0))
    }
}

/**
 * Everything the offline pipeline hands over, read into plain arrays. Per-listing arrays have
 * exactly n_listings rows; the mutable serving copy lives in [staysearch.data.ListingTable].
 */
class Bundle(
    val dir: Path,
    val manifest: Manifest,
    val listings: List<ListingRecord>,
    val embeddings: FloatArray,
    val clusterIds: IntArray,
    val centroids: FloatArray,
    val availability: ByteArray,
    val rankerStatic: FloatArray,
    val places: List<Place>,
    val guests: GuestHistory,
) {
    val bytesPerListing get() = manifest.horizonDays / 8
    fun path(name: String): Path = dir.resolve(name)
    fun modelPath(key: String): Path? = manifest.models[key]?.let { dir.resolve(it) }?.takeIf { Files.exists(it) }
    val queriesPath: Path get() = dir.resolve("queries_test.jsonl")
    val updatesPath: Path get() = dir.resolve("updates.jsonl")

    companion object {
        fun load(dir: Path): Bundle {
            val m = Manifest.parse(Json.mapper.readTree(dir.resolve("manifest.json").toFile()))
            require(m.horizonDays % 8 == 0) { "horizon_days must be a multiple of 8, got ${m.horizonDays}" }
            val n = m.nListings
            val listings = Json.readLines(dir.resolve("listings.jsonl")).map(ListingRecord::parse).toList()
            require(listings.size == n) { "listings.jsonl has ${listings.size} rows, manifest says $n" }
            val emb = readFloats(dir.resolve("embeddings.f32"), n.toLong() * m.dim)
            val clusters = readInts(dir.resolve("cluster_ids.i32"), n.toLong())
            val cents = readFloats(dir.resolve("centroids.f32"), m.nClusters.toLong() * m.dim)
            val avail = readBytes(dir.resolve("availability.bin"), n.toLong() * (m.horizonDays / 8))
            val static = readFloats(dir.resolve("ranker_static.f32"), n.toLong() * m.listingStaticDim)
            val places = Json.readLines(dir.resolve("places.jsonl")).map(Place::parse).toList()
            require(places.size == m.nPlaces) { "places.jsonl has ${places.size} rows, manifest says ${m.nPlaces}" }
            places.forEachIndexed { i, p -> require(p.idx == i) { "place line $i has idx ${p.idx}" } }
            val gIds = dir.resolve("guest_history.i64")
            val guests = if (Files.exists(gIds)) {
                val g = Files.size(gIds) / 8
                val ids = readLongs(gIds, g)
                for (i in 1 until ids.size) require(ids[i - 1] < ids[i]) { "guest_history.i64 is not strictly ascending at $i" }
                GuestHistory(ids, readFloats(dir.resolve("guest_history.f32"), g * GuestHistory.WIDTH))
            } else GuestHistory.EMPTY
            return Bundle(dir, m, listings, emb, clusters, cents, avail, static, places, guests)
        }

        private fun map(path: Path, expectedBytes: Long): ByteBuffer {
            val size = Files.size(path)
            require(size == expectedBytes) { "$path has $size bytes, expected $expectedBytes" }
            FileChannel.open(path, StandardOpenOption.READ).use { ch ->
                return ch.map(FileChannel.MapMode.READ_ONLY, 0, size).order(ByteOrder.LITTLE_ENDIAN)
            }
        }

        fun readFloats(path: Path, count: Long): FloatArray =
            FloatArray(Math.toIntExact(count)).also { map(path, count * 4).asFloatBuffer().get(it) }

        fun readInts(path: Path, count: Long): IntArray =
            IntArray(Math.toIntExact(count)).also { map(path, count * 4).asIntBuffer().get(it) }

        fun readLongs(path: Path, count: Long): LongArray =
            LongArray(Math.toIntExact(count)).also { map(path, count * 8).asLongBuffer().get(it) }

        fun readBytes(path: Path, count: Long): ByteArray =
            ByteArray(Math.toIntExact(count)).also { map(path, count).get(it) }
    }
}
