#!/usr/bin/env bash
set -euo pipefail

python inference/inference_offline.py \
  --src assets/target_examples/source.wav \
  --tgt assets/target_examples/british \
  --steps 16 \
  --cfg 2.0 \
  --device auto \
  --out inference/converted_offline.wav
