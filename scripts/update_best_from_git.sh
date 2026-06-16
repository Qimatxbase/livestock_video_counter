#!/usr/bin/env bash
set -euo pipefail

git pull

MODEL_PATH="models/pigdetect_best.pt"
if [ ! -f "$MODEL_PATH" ]; then
  echo "Model file not found: $MODEL_PATH"
  exit 1
fi

echo "Best model is available at: $(pwd)/$MODEL_PATH"
