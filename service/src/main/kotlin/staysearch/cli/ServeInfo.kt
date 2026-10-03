package staysearch.cli

import staysearch.bundle.Bundle
import staysearch.bundle.GuestHistory
import staysearch.util.Json

/** Payloads of the small read-only endpoints (`/meta`, `/sample-guests`), kept apart from Ktor so they are easy to test. */
object ServeInfo {
    /** Manifest fields a client needs to build a valid request (dates inside the calendar, cities, models). */
    fun meta(bundle: Bundle, rankers: Set<String>, hnsw: Boolean): Map<String, Any?> {
        val m = bundle.manifest
        return linkedMapOf(
            "as_of" to m.asOf.toString(),
            "base_date" to m.baseDate.toString(),
            "horizon_days" to m.horizonDays,
            "calendar_last_night" to m.baseDate.plusDays(m.horizonDays - 1L).toString(),
            "n_listings" to m.nListings,
            "n_places" to m.nPlaces,
            "n_clusters" to m.nClusters,
            "dim" to m.dim,
            "similarity" to m.similarity.name.lowercase(),
            "cities" to m.cities,
            "synthetic" to m.synthetic,
            "rankers" to rankers.sorted() + "none",
            "modes" to (if (hnsw) listOf("ivf", "hnsw", "exact") else listOf("ivf", "exact")),
            "defaults" to mapOf("k" to 500, "t" to 25, "nprobe" to 16, "mode" to "ivf", "ranker" to if ("dnn" in rankers) "dnn" else rankers.firstOrNull() ?: "none"),
            "flex_days" to FlexDates.FLEX,
        )
    }

    /**
     * Up to [n] distinct guests from queries_test.jsonl that have a history row, each with the test
     * query it came from (place, dates, guests), in file order.
     */
    fun sampleGuests(bundle: Bundle, n: Int = 20): List<Map<String, Any?>> {
        if (!java.nio.file.Files.exists(bundle.queriesPath)) return emptyList()
        val g = bundle.guests
        val seen = HashSet<Long>()
        val out = ArrayList<Map<String, Any?>>()
        for (q in Json.readLines(bundle.queriesPath)) {
            if (out.size >= n) break
            val id = q["guest_id"]?.takeIf { !it.isNull }?.asLong() ?: continue
            val i = g.ids.binarySearch(id)
            if (i < 0 || g.feats[i * GuestHistory.WIDTH] < 0.5f || !seen.add(id)) continue
            val place = q["place_idx"].asInt()
            out += linkedMapOf(
                "guest_id" to id,
                "place_idx" to place,
                "place_name" to bundle.places.getOrNull(place)?.name,
                "city" to bundle.places.getOrNull(place)?.city,
                "checkin" to q["checkin"].asText(),
                "checkout" to q["checkout"].asText(),
                "guests" to q["guests"].asInt(),
            )
        }
        return out
    }
}
