# StaySearch service (Kotlin)

One JVM: Apache Lucene 10 index (IVF cluster id as a filter, or Lucene HNSW on the same documents),
an in-place availability store exposed to Lucene as a custom query, the query tower and rankers in
ONNX Runtime, an update consumer, and the experiment harnesses. Not affiliated with Airbnb or Inside Airbnb.

## Build and test

```bash
export JAVA_HOME=/opt/homebrew/opt/openjdk@21
./gradlew build          # generates build/fake-bundle with ../.venv/bin/python, runs all tests
./gradlew installDist
B=build/install/staysearch-service/bin/staysearch-service
```

The fake bundle (`tools/make_fake_bundle.py --out build/fake-bundle`) is synthetic. Anything measured
on it (`bundle_synthetic: true` in the output) is a smoke test, never a result.

## Commands (real bundle in `../artifacts/bundle`)

| Command | What |
|---|---|
| `$B serve --bundle DIR [--port N] [--host 0.0.0.0] [--static ../ui] [--updates-rate 100]` | HTTP: `/search` (each response re-checked by the invariant checker: `invariant_ok`), `/health`, `/meta` (calendar, cities, models), `/places`, `/sample-guests` (test guests with history), `/listing/{id}`, `/stats`. With `--static ../ui` it also serves the map-and-list demo |
| `$B replay-invariant --bundle DIR --n 10000 --modes ivf,hnsw,exact --out ../results/m3_invariant.jsonl` | The hard invariant on every query; exits 1 on any violation |
| `$B parity --bundle DIR --out ../results/m3_parity.jsonl` | ONNX Runtime Java vs PyTorch, max abs diff < 1e-4 |
| `$B exp4 --bundle DIR --out ../results/m4_exp4.jsonl --modes ivf,hnsw --rates 0,100,1000,10000 --duration 60` | Index under updates |
| `$B exp5 --bundle DIR --out ../results/m3_exp5.jsonl` | JMH: availability check, run-length vs per-night scan, 49-window flexible check |
| `$B latency --bundle DIR --out ../results/m3_latency.jsonl --rates 50,200,500 --updates false,true` | Open-loop load at constant arrival rates, per-stage p50/p99 |
| `$B flexdates --bundle DIR --out ../results/m6_flexdates.jsonl [--n 2000 --mode ivf --latency-n 200]` | M6(b) flexible dates, see below |

**flexdates** measures plus or minus 3 days on check-in and on check-out (49 windows) on N test
queries and writes three lines. `eligible`: the exact-date eligible set (live, in the place's box,
fits the guests, nights within min/max, every night open) against the union over the 49 windows,
where a window counts only if it has at least 1 night and fits that listing's min/max nights; brute
force over the table, reported as the growth factor (median, p90) plus how many empty exact sets
become non-empty. `latency`: one flexible search (a single Lucene pass whose availability filter is
the 49-window mask) against 49 separate exact-date searches merged by id, sequential in one thread,
JIT warmed by untimed rounds of both, p50/p99 on the first `--latency-n` queries. `check_ns`: ns per
listing for one 49-window check against 49 single-window checks over every listing, timed with a
`System.nanoTime` loop (warm-up rounds, alternating order, median of `--check-rounds`), not JMH; the
line also says whether both methods counted the same open windows.

## Demo UI

```bash
$B serve --bundle ../artifacts/bundle --static ../ui   # prints the URL (ephemeral port unless --port or $PORT)
```

`../ui` is static HTML and JavaScript (no build step): city and neighbourhood pickers or "search this
map area", dates, guests, an optional guest id (with a sample-guest button), flexible dates, mode,
ranker and nprobe; results as map pins and a ranked list with per-stage server latency and the
invariant check. Map tiles come from OpenStreetMap and Leaflet from unpkg, so the browser needs network access.

Every results line carries `machine`, `load_start`, `load_end`, `repeat`, `bundle_as_of` and `lucene_version`.

See `BUNDLE_FORMAT.md` for the offline/online contract and `CONTRACT_NOTES.md` for the decisions taken under it.
