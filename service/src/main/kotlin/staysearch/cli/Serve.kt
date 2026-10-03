package staysearch.cli

import io.ktor.http.ContentType
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpMethod
import io.ktor.http.HttpStatusCode
import io.ktor.serialization.jackson.jackson
import io.ktor.server.application.Application
import io.ktor.server.application.install
import io.ktor.server.engine.embeddedServer
import io.ktor.server.http.content.staticFiles
import io.ktor.server.netty.Netty
import io.ktor.server.plugins.BadRequestException
import io.ktor.server.plugins.contentnegotiation.ContentNegotiation
import io.ktor.server.plugins.cors.routing.CORS
import io.ktor.server.request.receive
import io.ktor.server.response.respond
import io.ktor.server.routing.get
import io.ktor.server.routing.post
import io.ktor.server.routing.routing
import kotlinx.coroutines.runBlocking
import staysearch.Engine
import staysearch.EngineConfig
import staysearch.search.SearchRequest
import staysearch.updates.Replayer
import staysearch.updates.UpdateSource
import staysearch.util.Json
import java.io.File

/**
 * serve: the HTTP service.
 *
 * Options: --bundle DIR [--port N] (else $PORT, else an ephemeral port, printed) [--host 127.0.0.1]
 * [--static DIR] (served at /) [--index-dir DIR] (else a temp dir deleted on exit) [--no-vectors]
 * [--refresh-ms 1000] [--updates-rate R] (replay updates.jsonl in the background, events/s)
 * [--reindex-avail]
 */
object Serve {
    fun run(a: Args): Int {
        val port = a.str("port")?.toInt() ?: System.getenv("PORT")?.toIntOrNull() ?: 0
        val host = a.str("host", "127.0.0.1")
        val cfg = EngineConfig(
            indexDir = a.pathOrNull("index-dir"),
            withVectors = !a.flag("no-vectors"),
            refreshMs = a.long("refresh-ms", 1000),
            reindexOnAvail = a.flag("reindex-avail"),
        )
        val engine = Engine.open(a.path("bundle"), cfg)
        val rate = a.double("updates-rate", 0.0)
        val replayer = if (rate > 0) Replayer(UpdateSource(engine.bundle.updatesPath, loop = false), engine.applier, rate).also { it.start() } else null
        val staticDir = a.str("static")?.let(::File)
        val server = embeddedServer(Netty, port = port, host = host) { module(engine, replayer, staticDir) }
        Runtime.getRuntime().addShutdownHook(Thread {
            runCatching { server.stop(500, 2000) }
            runCatching { replayer?.close() }
            engine.close() // deletes a temp index directory
        })
        server.start(wait = false)
        val actual = runBlocking { server.engine.resolvedConnectors().first().port }
        println("staysearch serving ${engine.bundle.manifest.nListings} listings on http://$host:$actual")
        Thread.currentThread().join()
        return 0
    }

    fun Application.module(engine: Engine, replayer: Replayer?, staticDir: File?) {
        install(ContentNegotiation) { jackson { Json.configure(this) } }
        install(CORS) {
            anyHost()
            allowMethod(HttpMethod.Get)
            allowMethod(HttpMethod.Post)
            allowMethod(HttpMethod.Options)
            allowHeader(HttpHeaders.ContentType)
        }
        val bundle = engine.bundle
        val table = engine.table
        routing {
            get("/health") {
                call.respond(mapOf(
                    "status" to "ok",
                    "as_of" to bundle.manifest.asOf.toString(),
                    "n_listings_live" to table.liveCount,
                    "index_vectors" to engine.index.withVectors,
                    "updates" to engine.applier.counters(),
                    "replay_events_done" to replayer?.done?.get(),
                ))
            }
            get("/places") { call.respond(bundle.places) }
            get("/stats") { call.respond(engine.pipeline.stats.summary()) }
            get("/listing/{id}") {
                val id = call.parameters["id"]?.toLongOrNull() ?: throw BadRequestException("id must be a number")
                val row = table.rowOf(id)
                if (row == null) {
                    call.respond(HttpStatusCode.NotFound, mapOf("error" to "listing $id not found")); return@get
                }
                val h = table.horizon
                val nights = (0 until h).joinToString("") { if (table.store.isOpen(row, it)) "1" else "0" }
                call.respond(mapOf(
                    "id" to id, "lat" to table.lat[row], "lng" to table.lng[row], "price" to table.price[row],
                    "accommodates" to table.accommodates[row], "min_nights" to table.minNights[row], "max_nights" to table.maxNights[row],
                    "room_type" to table.roomType[row], "cluster" to table.cluster[row], "place_city" to table.placeCity[row],
                    "place_nbhd" to table.placeNbhd[row], "base_date" to bundle.manifest.baseDate.toString(),
                    "open_nights" to nights, "open_count" to nights.count { it == '1' },
                ))
            }
            post("/search") {
                val req = try {
                    call.receive<SearchRequest>()
                } catch (e: Exception) {
                    call.respond(HttpStatusCode.BadRequest, mapOf("error" to "bad request body: ${e.message}")); return@post
                }
                try {
                    call.respond(engine.pipeline.search(req))
                } catch (e: IllegalArgumentException) {
                    call.respond(HttpStatusCode.BadRequest, mapOf("error" to e.message))
                }
            }
            if (staticDir != null) staticFiles("/", staticDir) { default("index.html") }
            else get("/") { call.respond(mapOf("service" to "staysearch", "endpoints" to listOf("/search", "/health", "/places", "/listing/{id}", "/stats"))) }
        }
    }
}
