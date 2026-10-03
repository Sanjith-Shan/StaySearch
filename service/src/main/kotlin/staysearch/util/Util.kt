package staysearch.util

import com.fasterxml.jackson.databind.DeserializationFeature
import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.databind.PropertyNamingStrategies
import com.fasterxml.jackson.databind.SerializationFeature
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import java.io.BufferedWriter
import java.lang.management.ManagementFactory
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardOpenOption
import java.time.Instant
import java.util.concurrent.TimeUnit

object Json {
    /** Snake-case JSON everywhere, matching the bundle and the HTTP API. */
    val mapper: ObjectMapper = configure(jacksonObjectMapper())

    fun configure(m: ObjectMapper): ObjectMapper = m
        .setPropertyNamingStrategy(PropertyNamingStrategies.SNAKE_CASE)
        .registerModule(JavaTimeModule())
        .disable(SerializationFeature.WRITE_DATES_AS_TIMESTAMPS)
        .disable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES)

    fun readLines(path: Path): Sequence<JsonNode> = sequence {
        Files.newBufferedReader(path).use { r ->
            while (true) {
                val line = r.readLine() ?: break
                if (line.isNotBlank()) yield(mapper.readTree(line))
            }
        }
    }

    fun write(v: Any): String = mapper.writeValueAsString(v)
}

/** Appends JSON lines to a results file (creating parent directories). */
class JsonlWriter(path: Path) : AutoCloseable {
    private val w: BufferedWriter

    init {
        path.toAbsolutePath().parent?.let { Files.createDirectories(it) }
        w = Files.newBufferedWriter(path, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
    }

    fun write(row: Any) {
        w.write(Json.write(row)); w.newLine(); w.flush()
    }

    override fun close() = w.close()
}

/** Machine and load descriptors that every results line carries. */
object Machine {
    private fun sh(vararg cmd: String): String? = try {
        val p = ProcessBuilder(*cmd).redirectErrorStream(true).start()
        val out = p.inputStream.bufferedReader().readText().trim()
        if (p.waitFor(5, TimeUnit.SECONDS) && p.exitValue() == 0) out else null
    } catch (_: Exception) {
        null
    }

    val info: Map<String, Any?> by lazy {
        val mac = System.getProperty("os.name").lowercase().contains("mac")
        val cpu = if (mac) sh("sysctl", "-n", "machdep.cpu.brand_string")
        else sh("sh", "-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2")?.trim()
        val ram = if (mac) sh("sysctl", "-n", "hw.memsize")?.toLongOrNull()
        else sh("sh", "-c", "grep MemTotal /proc/meminfo | awk '{print \$2*1024}'")?.toLongOrNull()
        mapOf(
            "cpu" to cpu,
            "cores" to Runtime.getRuntime().availableProcessors(),
            "ram_bytes" to ram,
            "os" to "${System.getProperty("os.name")} ${System.getProperty("os.version")} ${System.getProperty("os.arch")}",
            "java" to System.getProperty("java.version"),
            "max_heap_bytes" to Runtime.getRuntime().maxMemory(),
        )
    }

    fun load(): Map<String, Any?> = mapOf(
        "loadavg_1m" to ManagementFactory.getOperatingSystemMXBean().systemLoadAverage,
        "uptime" to sh("uptime"),
        "at" to Instant.now().toString(),
    )

    /** Resident set size of this process in bytes, from `ps -o rss=` (kilobytes). */
    fun rssBytes(): Long? = sh("ps", "-o", "rss=", "-p", ProcessHandle.current().pid().toString())
        ?.trim()?.toLongOrNull()?.times(1024)

    /** Heap in use after a GC hint (best effort: System.gc twice with a short pause). */
    fun heapUsedAfterGc(): Long {
        repeat(2) { System.gc(); Thread.sleep(100) }
        val rt = Runtime.getRuntime()
        return rt.totalMemory() - rt.freeMemory()
    }

    val luceneVersion: String get() = org.apache.lucene.util.Version.LATEST.toString()
}

fun dirBytes(dir: Path): Long =
    if (!Files.exists(dir)) 0 else Files.walk(dir).use { s -> s.filter { Files.isRegularFile(it) }.mapToLong { f -> runCatching { Files.size(f) }.getOrDefault(0) }.sum() }

fun deleteRecursively(dir: Path) {
    if (!Files.exists(dir)) return
    Files.walk(dir).use { s -> s.sorted(Comparator.reverseOrder()).forEach { runCatching { Files.delete(it) } } }
}
