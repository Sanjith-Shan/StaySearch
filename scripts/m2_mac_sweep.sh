#!/bin/bash
# The Mac's share of the M2 sweep; the mini PC runs none/random/geo/price/mix_dot/mix_l2_noleak.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
while kill -0 80536 2>/dev/null; do sleep 10; done
run() { nice $PY -m staysearch.train_towers --epochs 10 --threads 4 "$@" >> logs/mac_towers.log 2>&1; }
run --name mined_l2 --sampler mined --similarity l2
run --name mix_l2_nohist --sampler mix --similarity l2 --no-history
echo SWEEP_DONE >> logs/mac_towers.log
