#!/bin/bash

set -x

# Miscellaneous
python analytic.py
python force.py
python mode_freq.py

# Run envelope tracking benchmarks
python env.py --periods 20 --eta 0.9 --runname run_01
python env.py --periods 20 --eta 0.5 --runname run_02

# PCM: matched vs. mismatched
python pcm.py --r0 1.0100 --runname matched
python pcm.py --r0 0.6095 --runname mismatched

# PCM: scan tune depression ratio
python pcm.py --r0 0.6095 --eta 0.9 --runname run_01
python pcm.py --r0 0.6095 --eta 0.7 --runname run_02
python pcm.py --r0 0.6095 --eta 0.5 --runname run_03
python pcm.py --r0 0.6095 --eta 0.3 --runname run_04
python pcm.py --r0 0.6095 --eta 0.1 --runname run_05

# PIC
python pic.py --nparts 100_000 --sc-grid 128 --plot-smooth 1 --mismatch 0.6095 --runname mismatched

