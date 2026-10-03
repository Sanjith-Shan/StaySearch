package staysearch

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import staysearch.bundle.GuestHistory
import staysearch.cli.FlexDates
import staysearch.cli.ServeInfo
import java.time.LocalDate

class ServeInfoTest {
    @Test
    fun `meta carries the calendar and the available models`() {
        val b = FakeBundle.require()
        val m = ServeInfo.meta(b, setOf("dnn", "gbdt"), hnsw = true)
        assertEquals(b.manifest.asOf.toString(), m["as_of"])
        assertEquals(b.manifest.horizonDays, m["horizon_days"])
        assertEquals(b.manifest.baseDate.plusDays(b.manifest.horizonDays - 1L).toString(), m["calendar_last_night"])
        assertEquals(listOf("dnn", "gbdt", "none"), m["rankers"])
        assertEquals(listOf("ivf", "exact"), ServeInfo.meta(b, emptySet(), hnsw = false)["modes"])
    }

    @Test
    fun `sample guests are distinct and all have history`() {
        val b = FakeBundle.require()
        val s = ServeInfo.sampleGuests(b, 20)
        assertTrue(s.isNotEmpty())
        assertTrue(s.size <= 20)
        val ids = s.map { it["guest_id"] as Long }
        assertEquals(ids.size, ids.toSet().size)
        for (id in ids) {
            val i = b.guests.ids.binarySearch(id)
            assertTrue(i >= 0 && b.guests.feats[i * GuestHistory.WIDTH] > 0.5f)
        }
        for (g in s) assertTrue(LocalDate.parse(g["checkout"] as String).isAfter(LocalDate.parse(g["checkin"] as String)))
    }

    @Test
    fun `flexible windows are the 49 shifted pairs`() {
        val w = FlexDates.windows(10, 12)
        assertEquals(49, w.size)
        assertEquals(49, w.toSet().size)
        // 2 nights: the 15 pairs where check-in moves later than check-out by 2 or more days have no nights.
        assertEquals(49 - 15, w.count { (a, b) -> b - a >= 1 })
    }
}
