#!/usr/bin/env bash
# Qwen2-Audio plain-prompt scores on train and test (validation already done), for a train-fitted label-bias
# correction and the cross-validated fusion screen. Resumable (alm_qwen.py saves every 50 clips).
set -u
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for k in B A; do
    echo "$(date +%T) start alm_$k" >> runs/day_0929/status.txt
    timeout 5h uv run python scripts/alm_qwen.py --dataset $k --prompts plain --splits train test > runs/day_0929/alm_traintest_$k.log 2>&1
    echo "$(date +%T) alm_$k exit $?" >> runs/day_0929/status.txt
done
