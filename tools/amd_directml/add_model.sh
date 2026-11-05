#!/usr/bin/env bash
# Add an RVC .pth (+ optional .index) to a running VCClient v1.5.3 as an ONNX
# slot, which is what AMD/DirectML needs (.pth models run on the CPU there).
#
#   add_model.sh <slot> <model.pth> [model.index] [port]
#
# Steps: upload pth+index -> load slot -> export ONNX -> reload slot as onnx+index.
set -euo pipefail
SLOT=$1; PTH=$2; IDX=${3:-}; PORT=${4:-18888}
B="http://127.0.0.1:$PORT"

upload() { curl -sf -F "file=@$1" -F "filename=$(basename "$1")" "$B/upload_file" >/dev/null; }

load_slot() { # $1 = model file name, $2 = index file name or ""
  local files="[{\"name\":\"$1\",\"kind\":\"rvcModel\",\"dir\":\"\"}"
  [ -n "$2" ] && files="$files,{\"name\":\"$2\",\"kind\":\"rvcIndex\",\"dir\":\"\"}"
  files="$files]"
  curl -sf -F "slot=$SLOT" -F "isHalf=false" \
    -F "params={\"voiceChangerType\":\"RVC\",\"slot\":$SLOT,\"isSampleMode\":false,\"sampleId\":\"\",\"files\":$files,\"params\":{}}" \
    "$B/load_model" >/dev/null
}

echo "[1/4] uploading $(basename "$PTH")"
upload "$PTH"; [ -n "$IDX" ] && upload "$IDX"
load_slot "$(basename "$PTH")" "$([ -n "$IDX" ] && basename "$IDX")"

echo "[2/4] selecting slot $SLOT"
curl -sf -F "key=modelSlotIndex" -F "val=$SLOT" "$B/update_settings" >/dev/null

echo "[3/4] exporting ONNX (takes a little while)"
# reply: {"status":"ok","path":"/tmp/<name>_simple.onnx","filename":"<name>_simple.onnx"}
REPLY=$(curl -s -m 600 "$B/onnx" || true)
[ -z "$REPLY" ] && REPLY=$(curl -s -m 600 "$B/onnx")
ONNX_URL=$(echo "$REPLY" | sed -n 's/.*"path":"\([^"]*\)".*/\1/p')
[ -z "$ONNX_URL" ] && { echo "onnx export failed: $REPLY"; exit 1; }
WORK=$(mktemp -d)
ONNX="$WORK/$(basename "$ONNX_URL")"
curl -sf -o "$ONNX" "$B$ONNX_URL"
echo "      -> $(basename "$ONNX") ($(du -m "$ONNX" | cut -f1) MB)"
cp "$ONNX" "$(dirname "$PTH")/"  # keep a copy next to the .pth

echo "[4/4] reloading slot $SLOT as ONNX"
upload "$ONNX"; [ -n "$IDX" ] && upload "$IDX"
load_slot "$(basename "$ONNX")" "$([ -n "$IDX" ] && basename "$IDX")"
curl -sf -F "key=modelSlotIndex" -F "val=$SLOT" "$B/update_settings" >/dev/null
echo "done: slot $SLOT"
