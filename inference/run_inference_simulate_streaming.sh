#!/usr/bin/env bash
set -euo pipefail

python inference/inference_simulate_streaming.py \
  --src assets/target_examples/source.wav \
  --tgt assets/target_examples/british \
  --steps 16 \
  --cfg 2.0 \
  --device auto \
  --chunksize 9600 \
  --out inference/converted_simulated_streaming.wav
