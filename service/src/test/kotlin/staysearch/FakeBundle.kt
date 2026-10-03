package staysearch

import org.junit.jupiter.api.Assumptions.assumeTrue
import staysearch.bundle.Bundle
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths

/** The synthetic bundle written by tools/make_fake_bundle.py (Gradle's fakeBundle task). */
object FakeBundle {
    val dir: Path? = System.getProperty("staysearch.fakeBundle")?.let { Paths.get(it) }?.takeIf { Files.exists(it.resolve("manifest.json")) }

    val bundle: Bundle by lazy { Bundle.load(requireNotNull(dir)) }

    /** Skips the calling test when the bundle is absent. */
    fun require(): Bundle {
        assumeTrue(dir != null, "fake bundle not generated (run ./gradlew fakeBundle)")
        return bundle
    }

    fun golden(name: String): Path? = dir?.resolve("golden")?.resolve(name)?.takeIf { Files.exists(it) }
}
