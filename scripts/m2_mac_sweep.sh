#!/bin/bash
# The Mac's half of the M2 sweep (the mini PC runs none/random/geo/price). Sequential, 4 threads.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
run() { nice $PY -m staysearch.train_towers --epochs 10 --threads 4 "$@" >> logs/mac_towers.log 2>&1; }
run --name mix_l2 --sampler mix --similarity l2
run --name mined_l2 --sampler mined --similarity l2
run --name mix_dot --sampler mix --similarity dot
run --name mix_l2_nohist --sampler mix --similarity l2 --no-history
run --name mix_l2_noleak --sampler mix --similarity l2 --drop-guests-nights
echo SWEEP_DONE >> logs/mac_towers.log
