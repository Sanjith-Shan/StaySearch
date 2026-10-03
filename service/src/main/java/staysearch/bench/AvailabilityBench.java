package staysearch.bench;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;

import com.fasterxml.jackson.databind.JsonNode;

import staysearch.avail.AvailabilityStore;
import staysearch.avail.PerNightScan;
import staysearch.bundle.Bundle;
import staysearch.util.Json;

/**
 * exp5: nanoseconds per stay check, run-length array vs per-night scan, for one window and for the
 * 49-window flexible check (plus or minus 3 days on check-in and check-out).
 *
 * Calendars come from a bundle's availability.bin when {@code bundle} is set, otherwise from a
 * synthetic generator (alternating open and closed runs). Each invocation checks the next of 4096
 * precomputed random (listing, check-in, nights) windows, so the same windows feed both methods.
 * Written in Java so JMH's annotation processor runs without kapt.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
public class AvailabilityBench {
    private static final int WINDOWS = 4096;

    @Param({""})
    public String bundle;

    @Param({"20000"})
    public int syntheticListings;

    /** Longest stay length drawn for a window (uniform in 1..maxNights). */
    @Param({"14"})
    public int maxNights;

    private AvailabilityStore store;
    private PerNightScan scan;
    private final int[] rows = new int[WINDOWS];
    private final int[] checkin = new int[WINDOWS];
    private final int[] checkout = new int[WINDOWS];
    private int next;

    @Setup
    public void setup() throws Exception {
        int horizon;
        int n;
        byte[] packed;
        if (bundle != null && !bundle.isEmpty()) {
            Path dir = Paths.get(bundle);
            JsonNode m = Json.INSTANCE.getMapper().readTree(dir.resolve("manifest.json").toFile());
            horizon = m.get("horizon_days").asInt();
            n = m.get("n_listings").asInt();
            packed = Bundle.Companion.readBytes(dir.resolve("availability.bin"), (long) n * (horizon / 8));
        } else {
            horizon = 448;
            n = syntheticListings;
            packed = synthetic(n, horizon, new Random(11));
        }
        int bpl = horizon / 8;
        store = new AvailabilityStore(horizon, n);
        scan = new PerNightScan(horizon, n);
        for (int i = 0; i < n; i++) {
            store.load(i, packed, i * bpl);
            scan.load(i, packed, i * bpl);
        }
        Random r = new Random(42);
        for (int i = 0; i < WINDOWS; i++) {
            rows[i] = r.nextInt(n);
            int nights = 1 + r.nextInt(maxNights);
            checkin[i] = 3 + r.nextInt(horizon - nights - 6);
            checkout[i] = checkin[i] + nights;
        }
    }

    private static byte[] synthetic(int n, int horizon, Random r) {
        byte[] out = new byte[n * (horizon / 8)];
        for (int i = 0; i < n; i++) {
            boolean open = r.nextBoolean();
            int d = 0;
            while (d < horizon) {
                int len = 1 + (int) (-Math.log(1 - r.nextDouble()) * 6);
                for (int k = d; k < Math.min(horizon, d + len); k++) {
                    if (open) out[i * (horizon / 8) + (k >>> 3)] |= (byte) (1 << (k & 7));
                }
                d += len;
                open = !open;
            }
        }
        return out;
    }

    private int nextWindow() {
        return next++ & (WINDOWS - 1);
    }

    @Benchmark
    public boolean runLengthSingle() {
        int j = nextWindow();
        return store.isAvailable(rows[j], checkin[j], checkout[j]);
    }

    @Benchmark
    public boolean perNightScanSingle() {
        int j = nextWindow();
        return scan.isAvailable(rows[j], checkin[j], checkout[j]);
    }

    @Benchmark
    public long runLengthFlexible49() {
        int j = nextWindow();
        return store.flexibleMask(rows[j], checkin[j], checkout[j], 3, 1, Integer.MAX_VALUE);
    }

    @Benchmark
    public long perNightScanFlexible49() {
        int j = nextWindow();
        return scan.flexibleMask(rows[j], checkin[j], checkout[j], 3, 1, Integer.MAX_VALUE);
    }
}
