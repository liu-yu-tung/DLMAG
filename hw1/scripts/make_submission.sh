#!/usr/bin/env bash
# Build the folder handed to the TA: inference code, the final training pipeline, checkpoints, README
# (submission/README.md) and an inference-only requirements.txt (the pins of requirements.txt, without -e .).
# Usage: scripts/make_submission.sh [out_dir]   (default dist/hw1_submission; rebuilt from scratch each run)
# An existing out_dir is replaced only under dist/; anywhere else it must not exist yet.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-dist/hw1_submission}
if [ -e "$OUT" ] && [[ "$OUT" != dist/?* ]]; then
    echo "$OUT exists and is outside dist/; refusing to replace it" >&2
    exit 1
fi
rm -rf "$OUT"
mkdir -p "$OUT"/{scripts,src/hw1/features,checkpoints,results}
cp scripts/{predict,alm_qwen,finetune_mert,extract_mert,extract_langid,train_final}.py "$OUT/scripts/"
cp src/hw1/{__init__,data,final,probe,metrics}.py "$OUT/src/hw1/"
cp src/hw1/features/{__init__,mert,langid,separation}.py "$OUT/src/hw1/features/"
cp results/final_validation.json "$OUT/results/"
cp --reflink=auto checkpoints/{A,B}.joblib checkpoints/ft_{A,B}_L12_s{0,1,2}.pt "$OUT/checkpoints/"
cp submission/README.md "$OUT/README.md"
{
    echo "# Inference only (scripts/predict.py). Python 3.12: pip install -r requirements.txt"
    grep -E '^[A-Za-z0-9_.-]+==' requirements.txt
} > "$OUT/requirements.txt"
(cd "$OUT" && sha256sum checkpoints/*) > "$OUT/checkpoints.sha256"
du -sh "$OUT"
