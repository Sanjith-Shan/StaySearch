package staysearch.search

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import staysearch.bundle.Bundle
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * ONNX Runtime sessions, one per model, created once and shared by all request threads
 * (OrtSession.run is thread-safe).
 */
class Models(bundle: Bundle, intraOpThreads: Int = 1) : AutoCloseable {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val opts = OrtSession.SessionOptions().apply {
        setIntraOpNumThreads(intraOpThreads)
        setInterOpNumThreads(1)
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
    }
    val dim = bundle.manifest.dim
    val qFeatDim = bundle.manifest.qFeatDim
    val rankerWidth = bundle.manifest.rankerFeatDim.also {
        require(it == bundle.manifest.listingStaticDim + Features.N_CROSS) {
            "ranker_feat_dim $it != listing_static_dim ${bundle.manifest.listingStaticDim} + ${Features.N_CROSS} cross features"
        }
    }

    private val queryTower: OrtSession = env.createSession(
        requireNotNull(bundle.modelPath("query_tower")) { "query tower model missing" }.toString(), opts,
    )

    /** Rankers by request name: "dnn" -> models.ranker, "gbdt" -> models.ranker_gbdt (each optional). */
    private val rankers: Map<String, OrtSession> = buildMap {
        bundle.modelPath("ranker")?.let { put("dnn", env.createSession(it.toString(), opts)) }
        bundle.modelPath("ranker_gbdt")?.let { put("gbdt", env.createSession(it.toString(), opts)) }
    }

    val rankerNames: Set<String> get() = rankers.keys

    /** Runs the query tower on a batch: [places] has n entries, [qfeat] n x qFeatDim. Returns n x dim. */
    fun embedQueries(places: LongArray, qfeat: FloatArray): FloatArray {
        val n = places.size
        OnnxTensor.createTensor(env, LongBuffer.wrap(places), longArrayOf(n.toLong())).use { p ->
            OnnxTensor.createTensor(env, FloatBuffer.wrap(qfeat), longArrayOf(n.toLong(), qFeatDim.toLong())).use { q ->
                queryTower.run(mapOf("place_idx" to p, "qfeat" to q)).use { r ->
                    val out = (r.get("qemb").orElseGet { r.get(0) } as OnnxTensor).floatBuffer
                    return FloatArray(out.remaining()).also { out.get(it) }
                }
            }
        }
    }

    fun embedQuery(placeIdx: Int, qfeat: FloatArray): FloatArray = embedQueries(longArrayOf(placeIdx.toLong()), qfeat)

    /** Scores [rows] ranker rows ([x] is rows x rankerWidth). Accepts a [B] or [B, 1] output named "score". */
    fun score(ranker: String, x: FloatArray, rows: Int): FloatArray {
        val s = rankers[ranker] ?: throw IllegalArgumentException("ranker '$ranker' not in bundle (have ${rankers.keys})")
        if (rows == 0) return FloatArray(0)
        OnnxTensor.createTensor(env, FloatBuffer.wrap(x, 0, rows * rankerWidth), longArrayOf(rows.toLong(), rankerWidth.toLong())).use { t ->
            s.run(mapOf("x" to t)).use { r ->
                val out = (r.get("score").orElseGet { r.get(0) } as OnnxTensor).floatBuffer
                return FloatArray(out.remaining()).also { out.get(it) }
            }
        }
    }

    override fun close() {
        queryTower.close()
        rankers.values.forEach { it.close() }
        opts.close()
    }
}
