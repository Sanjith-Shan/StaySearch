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

    private val windows = System.getProperty("os.name").lowercase().contains("windows")
    private val osBean get() = ManagementFactory.getOperatingSystemMXBean() as? com.sun.management.OperatingSystemMXBean

    private fun ps(cmd: String): String? = sh("powershell", "-NoProfile", "-NonInteractive", "-Command", cmd)

    val info: Map<String, Any?> by lazy {
        val mac = System.getProperty("os.name").lowercase().contains("mac")
        val cpu = when {
            mac -> sh("sysctl", "-n", "machdep.cpu.brand_string")
            windows -> ps("(Get-CimInstance Win32_Processor | Select-Object -First 1).Name")
            else -> sh("sh", "-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2")?.trim()
        }
        val ram = osBean?.totalMemorySize
        mapOf(
            "cpu" to cpu,
            "cores" to Runtime.getRuntime().availableProcessors(),
            "ram_bytes" to ram,
            "os" to "${System.getProperty("os.name")} ${System.getProperty("os.version")} ${System.getProperty("os.arch")}",
            "java" to System.getProperty("java.version"),
            "max_heap_bytes" to Runtime.getRuntime().maxMemory(),
        )
    }

    /** Windows has no load average (the MXBean returns -1), so whole-machine CPU use is recorded too. */
    fun load(): Map<String, Any?> = mapOf(
        "loadavg_1m" to ManagementFactory.getOperatingSystemMXBean().systemLoadAverage,
        "system_cpu_load" to osBean?.cpuLoad,
        "uptime" to if (windows) null else sh("uptime"),
        "at" to Instant.now().toString(),
    )

    /** Resident set size of this process in bytes: `ps -o rss=` (kilobytes), or the working set on Windows. */
    fun rssBytes(): Long? {
        val pid = ProcessHandle.current().pid().toString()
        return if (windows) ps("(Get-Process -Id $pid).WorkingSet64")?.trim()?.toLongOrNull()
        else sh("ps", "-o", "rss=", "-p", pid)?.trim()?.toLongOrNull()?.times(1024)
    }

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
