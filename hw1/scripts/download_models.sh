#!/usr/bin/env bash
# Download pretrained weights into the Hugging Face cache, retrying until each finishes.
# Safe to rerun: partial files resume and finished files are skipped.
# Usage: scripts/download_models.sh [repo_id ...]
set -u
cd "$(dirname "$0")/.."

export HF_HUB_DISABLE_XET=1

models=("$@")
if [ ${#models[@]} -eq 0 ]; then
  models=(
    m-a-p/MERT-v2-30s
    laion/larger_clap_music
    openai/whisper-large-v3-turbo
    Qwen/Qwen2-Audio-7B-Instruct
  )
fi

for m in "${models[@]}"; do
  for attempt in $(seq 1 50); do
    echo "=== $m attempt $attempt $(date +%T)"
    if .venv/bin/hf download "$m" \
        --exclude "*.msgpack" --exclude "*.h5" --exclude "*.onnx" \
        --exclude "flax_model*" --exclude "tf_model*" > /dev/null 2>&1; then
      echo "done $m $(date +%T)"
      break
    fi
    sleep 30
  done
done
