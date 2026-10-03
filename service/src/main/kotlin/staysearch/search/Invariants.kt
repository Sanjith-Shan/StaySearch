package staysearch.search

import staysearch.bundle.Bundle
import staysearch.data.ListingTable

data class Violation(val listingId: Long, val reason: String)

/**
 * The hard invariant: every returned listing is open for every requested night in the store's
 * current state, satisfies min and max nights, fits the guests, and lies inside the map box.
 *
 * Availability is verified night by night from the bitset, not through the run-length array, so a
 * bug in the run-length maintenance cannot hide itself. For a flexible search, at least one of the
 * 49 windows must satisfy every condition.
 */
class InvariantChecker(private val bundle: Bundle, private val table: ListingTable) {

    fun check(req: SearchRequest, resp: SearchResponse): List<Violation> {
        val ctx = Features.resolve(req, bundle)
        val out = ArrayList<Violation>()
        for (r in resp.results) {
            val row = table.rowOf(r.id)
            if (row == null) {
                out += Violation(r.id, "unknown or removed listing"); continue
            }
            if (!ctx.box.contains(table.lat[row], table.lng[row])) out += Violation(r.id, "outside box")
            if (table.accommodates[row] < ctx.guests) out += Violation(r.id, "accommodates ${table.accommodates[row]} < guests ${ctx.guests}")
            val windows = if (req.flexible) (-3..3).flatMap { i -> (-3..3).map { j -> (ctx.checkinNight + i) to (ctx.checkoutNight + j) } }
            else listOf(ctx.checkinNight to ctx.checkoutNight)
            val ok = windows.any { (a, b) -> windowOk(row, a, b) }
            if (!ok) out += Violation(r.id, if (req.flexible) "no open flexible window within min/max nights" else describe(row, ctx.checkinNight, ctx.checkoutNight))
        }
        return out
    }

    private fun windowOk(row: Int, a: Int, b: Int): Boolean {
        val n = b - a
        return n >= 1 && n >= table.minNights[row] && n <= table.maxNights[row] && table.store.allNightsOpenByBits(row, a, b)
    }

    private fun describe(row: Int, a: Int, b: Int): String {
        val n = b - a
        if (n < table.minNights[row]) return "nights $n < min_nights ${table.minNights[row]}"
        if (n > table.maxNights[row]) return "nights $n > max_nights ${table.maxNights[row]}"
        val closed = (a until b).filter { !table.store.isOpen(row, it) }
        return "closed nights $closed in [$a, $b)"
    }
}
