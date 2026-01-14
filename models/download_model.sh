#!/usr/bin/env bash
set -e

MODEL_DIR="$(dirname "$0")"
MODEL_PATH="$MODEL_DIR/arcface.onnx"

echo "Downloading ArcFace ONNX model..."

# 1. Correct link for 2026
# 2. Added --connect-timeout and --retry for better stability
curl -L --connect-timeout 60 --retry 5 -o "$MODEL_PATH" \
"https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx?download=true"

echo "Download finished."

echo "Verifying file type:"
file "$MODEL_PATH"