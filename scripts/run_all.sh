#!/usr/bin/env bash
# Train + evaluate the 4 methods for every seed, then build the comparison report.
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=${SEEDS:-"0 1 2"}
for seed in $SEEDS; do
  for method in siamese protonet matching relation; do
    python scripts/train.py --method "$method" --seed "$seed"
  done
done
python scripts/compare.py
