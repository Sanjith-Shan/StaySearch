package staysearch.index

import org.apache.lucene.index.DocValues
import org.apache.lucene.index.LeafReaderContext
import org.apache.lucene.search.ConstantScoreScorer
import org.apache.lucene.search.ConstantScoreWeight
import org.apache.lucene.search.DocIdSetIterator
import org.apache.lucene.search.IndexSearcher
import org.apache.lucene.search.Query
import org.apache.lucene.search.QueryVisitor
import org.apache.lucene.search.ScoreMode
import org.apache.lucene.search.ScorerSupplier
import org.apache.lucene.search.TwoPhaseIterator
import org.apache.lucene.search.Weight
import staysearch.data.ListingTable

/**
 * Matches documents whose listing is open for the requested stay in the availability store's
 * *current* state, which lives outside the index. The document's row comes from the `row` numeric
 * doc value. With [flex] > 0 a document matches if any of the (2 flex + 1)^2 check-in/check-out
 * pairs is open and its length is within the listing's min/max nights.
 *
 * It is a two-phase iterator over all documents, so in a conjunction Lucene only calls it on docs
 * that already passed the cheaper point filters. It is never cacheable: the answer changes whenever
 * the store changes, without any change to the index.
 */
class AvailabilityQuery(
    private val table: ListingTable,
    val checkin: Int,
    val checkout: Int,
    val flex: Int = 0,
) : Query() {

    fun matchesRow(row: Int): Boolean {
        if (row < 0 || row >= table.size || table.isRemoved(row)) return false
        return if (flex == 0) table.store.isAvailable(row, checkin, checkout)
        else table.store.flexibleMask(row, checkin, checkout, flex, table.minNights[row], table.maxNights[row]) != 0L
    }

    override fun createWeight(searcher: IndexSearcher, scoreMode: ScoreMode, boost: Float): Weight =
        object : ConstantScoreWeight(this, boost) {
            override fun scorerSupplier(context: LeafReaderContext): ScorerSupplier {
                val rows = DocValues.getNumeric(context.reader(), ListingIndex.F_ROW)
                val approximation = DocIdSetIterator.all(context.reader().maxDoc())
                val twoPhase = object : TwoPhaseIterator(approximation) {
                    override fun matches(): Boolean {
                        val doc = approximation.docID()
                        return rows.advanceExact(doc) && matchesRow(rows.longValue().toInt())
                    }

                    override fun matchCost(): Float = if (flex == 0) 20f else 400f
                }
                val scorer = ConstantScoreScorer(score(), scoreMode, twoPhase)
                return DefaultScorerSupplier(scorer)
            }

            override fun isCacheable(ctx: LeafReaderContext): Boolean = false
        }

    override fun visit(visitor: QueryVisitor) = visitor.visitLeaf(this)
    override fun toString(field: String?) = "Availability[$checkin,$checkout)±$flex"
    override fun equals(other: Any?) = this === other
    override fun hashCode() = System.identityHashCode(this)
}
