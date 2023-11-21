#!/usr/bin/env bash

LLAMA_SOURCE="huggyllama/llama-13b"
WEIGHT_DIFF_SOURCE="lmsys/vicuna-13b-delta-v1.1"
MODEL_NAME="$(basename $WEIGHT_DIFF_SOURCE)"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
CACHE_DIR="$SCRIPT_DIR/../cache/$MODEL_NAME"
HUB_FOLDER="$HOME/.cache/huggingface/hub"
PATCHED_PATH="$CACHE_DIR/patched"

error() {
  echo "$1" >/dev/stderr
  exit 1
}

if [[ -e "$PATCHED_PATH" ]]; then
  echo "model already setuped"
  exit 0
else
  echo "setting up model"
fi

mkdir "$CACHE_DIR" -p

download_file() {
  if ! [[ -f "$CACHE_DIR/$1" ]]; then
    FILENAME="$(basename $1)"
    curl -LsSfo "$CACHE_DIR/$FILENAME" "$1"
  fi
}

with_snapshot() {
  MODEL_FOLDER="$HUB_FOLDER"/models--$(sed "s/\\//--/g" <<<"$1")
  [[ -e "$MODEL_FOLDER" ]] || error "$MODEL_FOLDER does not exist. Is the latest transformers version installed?"
  SNAPSHOTS_FOLDER="$MODEL_FOLDER/snapshots"
  SNAPSHOTS=($(ls "$SNAPSHOTS_FOLDER"))
  echo "$SNAPSHOTS_FOLDER/${SNAPSHOTS[0]}"
}

FILE_ROOT_URL=https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/model

for file in apply_delta.py; do
  download_file "$FILE_ROOT_URL/$file"
done

python -c "
from huggingface_hub import snapshot_download

snapshot_download(repo_id=\"${LLAMA_SOURCE}\")
snapshot_download(repo_id=\"${WEIGHT_DIFF_SOURCE}\")
"

LLAMA_FOLDER="$(with_snapshot $LLAMA_SOURCE)"
VICUNA_FOLDER="$(with_snapshot $WEIGHT_DIFF_SOURCE)"

python "$CACHE_DIR/apply_delta.py" --base "$LLAMA_FOLDER" --target "$PATCHED_PATH" --delta "$VICUNA_FOLDER"
