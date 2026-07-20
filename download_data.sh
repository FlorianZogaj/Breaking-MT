#!/bin/bash

WMT_URL="https://github.com/wmt-conference/wmt25-general-mt/raw/refs/heads/main/data/wmt25-genmt-humeval.jsonl"
DATA_DIR="data"
WMT_FILE_PATH="$DATA_DIR/wmt25-genmt-humeval.jsonl"

mkdir -p "$DATA_DIR"

if curl -L "$WMT_URL" -o "$WMT_FILE_PATH"; then
    echo "Downloaded WMT data."
else
    echo "ERROR: Failed to download WMT data."
    rm -f "$WMT_FILE_PATH"
    exit 1
fi
