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
| `$B serve --bundle DIR [--port N] [--host 0.0.0.0] [--static ../ui] [--updates-rate 100]` | HTTP: `/search`, `/health`, `/places`, `/listing/{id}`, `/stats` |
| `$B replay-invariant --bundle DIR --n 10000 --modes ivf,hnsw,exact --out ../results/m3_invariant.jsonl` | The hard invariant on every query; exits 1 on any violation |
| `$B parity --bundle DIR --out ../results/m3_parity.jsonl` | ONNX Runtime Java vs PyTorch, max abs diff < 1e-4 |
| `$B exp4 --bundle DIR --out ../results/m4_exp4.jsonl --modes ivf,hnsw --rates 0,100,1000,10000 --duration 60` | Index under updates |
| `$B exp5 --bundle DIR --out ../results/m3_exp5.jsonl` | JMH: availability check, run-length vs per-night scan, 49-window flexible check |
| `$B latency --bundle DIR --out ../results/m3_latency.jsonl --rates 50,200,500 --updates false,true` | Open-loop load at constant arrival rates, per-stage p50/p99 |

Every results line carries `machine`, `load_start`, `load_end`, `repeat`, `bundle_as_of` and `lucene_version`.

See `BUNDLE_FORMAT.md` for the offline/online contract and `CONTRACT_NOTES.md` for the decisions taken under it.
