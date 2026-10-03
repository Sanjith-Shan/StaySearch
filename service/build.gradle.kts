plugins {
    kotlin("jvm") version "2.4.20"
    application
}

group = "staysearch"
version = "0.1.0"

repositories {
    mavenCentral()
}

val luceneVersion = "10.5.1"
val ktorVersion = "3.6.0"
val jmhVersion = "1.37"
val jacksonVersion = "2.22.3"

dependencies {
    implementation("org.apache.lucene:lucene-core:$luceneVersion")
    implementation("com.microsoft.onnxruntime:onnxruntime:1.30.0")
    implementation("io.ktor:ktor-server-netty:$ktorVersion")
    implementation("io.ktor:ktor-server-content-negotiation:$ktorVersion")
    implementation("io.ktor:ktor-server-cors:$ktorVersion")
    implementation("io.ktor:ktor-serialization-jackson:$ktorVersion")
    implementation("com.fasterxml.jackson.module:jackson-module-kotlin:$jacksonVersion")
    implementation("com.fasterxml.jackson.datatype:jackson-datatype-jsr310:$jacksonVersion")
    implementation("org.hdrhistogram:HdrHistogram:2.2.2")
    implementation("org.slf4j:slf4j-simple:2.0.17")
    // JMH benchmarks live in src/main/java so that the `exp5` subcommand can run them in-process.
    implementation("org.openjdk.jmh:jmh-core:$jmhVersion")
    annotationProcessor("org.openjdk.jmh:jmh-generator-annprocess:$jmhVersion")

    testImplementation(platform("org.junit:junit-bom:5.14.4"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
    testImplementation("io.kotest:kotest-property:6.2.5")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.10.2")
}

kotlin {
    jvmToolchain(21)
}

java {
    toolchain { languageVersion.set(JavaLanguageVersion.of(21)) }
}

application {
    mainClass.set("staysearch.cli.MainKt")
    applicationDefaultJvmArgs = listOf("-Xmx6g", "--enable-native-access=ALL-UNNAMED", "--add-modules", "jdk.incubator.vector")
}

// A small synthetic bundle (tools/make_fake_bundle.py) that the tests run against.
// Python is taken from -Ppython=..., then $STAYSEARCH_PYTHON, then ../.venv/bin/python, then python3.
val fakeBundleDir = layout.buildDirectory.dir("fake-bundle")
val pythonExe: String = (findProperty("python") as String?)
    ?: System.getenv("STAYSEARCH_PYTHON")
    ?: rootDir.resolve("../.venv/bin/python").takeIf { it.exists() }?.absolutePath
    ?: "python3"

val fakeBundle = tasks.register<Exec>("fakeBundle") {
    description = "Generates the synthetic test bundle with tools/make_fake_bundle.py"
    group = "verification"
    val script = file("tools/make_fake_bundle.py")
    inputs.file(script)
    outputs.dir(fakeBundleDir)
    commandLine(pythonExe, script.absolutePath, "--out", fakeBundleDir.get().asFile.absolutePath)
}

tasks.test {
    dependsOn(fakeBundle)
    useJUnitPlatform()
    systemProperty("staysearch.fakeBundle", fakeBundleDir.get().asFile.absolutePath)
    maxHeapSize = "2g"
    jvmArgs("--enable-native-access=ALL-UNNAMED", "--add-modules", "jdk.incubator.vector")
    testLogging {
        events("passed", "skipped", "failed")
        exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
    }
}

tasks.named<JavaExec>("run") {
    workingDir = rootDir
}
