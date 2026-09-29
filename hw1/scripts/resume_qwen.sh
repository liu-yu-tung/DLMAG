#!/usr/bin/env bash
# Resume the Qwen2-Audio shards with curl -C - (this huggingface-hub gives every attempt a random temp name, so hf cannot resume).
# Each partial <oid>.<x>.incomplete is finished in place, renamed to <oid>, then "hf download" links the snapshot.
set -u
B=~/.cache/huggingface/hub/models--Qwen--Qwen2-Audio-7B-Instruct/blobs
declare -A F=(
  [383de5b5b06f7e7f276850a065d9164641c93009f79a377352a7c949d17c1d7a]="model-00001-of-00005.safetensors 3911340848"
  [610e59a23cdf1f78d7e3b42e69f2e1a08578f29726391a1772aecac95db3c59c]="model-00002-of-00005.safetensors 3980786080"
  [b9ea76e97226524a12b6acc5e896bebf01a01018fbcad20a72eb13c7950f6916]="model-00003-of-00005.safetensors 3980819456"
  [d68dee591619e26e3d0af23a07e181c70134fbc3aa33e24d872224607aae2c62]="model-00004-of-00005.safetensors 3643135696"
)
for oid in "${!F[@]}"; do
  read -r name size <<< "${F[$oid]}"
  part=$(ls "$B/$oid".*.incomplete 2>/dev/null | head -1)
  [ -e "$B/$oid" ] && continue
  for attempt in $(seq 1 100); do
    echo "$name attempt $attempt $(date +%T) have $(stat -c %s "$part")"
    curl -sL -C - -o "$part" "https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct/resolve/main/$name" 
    [ "$(stat -c %s "$part")" -eq "$size" ] && break
    sleep 20
  done
  [ "$(stat -c %s "$part")" -eq "$size" ] && mv "$part" "$B/$oid" && echo "done $name $(date +%T)"
done
