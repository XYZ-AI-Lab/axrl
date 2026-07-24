#!/usr/bin/bash

set -euo pipefail

# Following https://github.com/PeterGriffinJin/Search-R1/blob/main/docs/retriever.md


export AXRL_DATA_DIR="${AXRL_DATA_DIR:-${HOME}/axrl-data/datasets}"
echo "${AXRL_DATA_DIR}"
save_path="${AXRL_DATA_DIR}/search_r1_data"
echo "Using data directory: $save_path"
mkdir -p "$save_path"

# download  data if not existed
if [ -f "${save_path}/wiki-18.jsonl" ]; then
    echo "Data already exists in $save_path, skipping download."
else
    hf download PeterJinGo/wiki-18-bm25-index --repo-type dataset --local-dir "$save_path"
    hf download PeterJinGo/wiki-18-corpus --repo-type dataset --local-dir "$save_path"
    gzip -d "${save_path}/wiki-18.jsonl.gz"
fi

if [[ -z "${AXRL_SEARCH_PORT:-}" ]]; then
    echo "ERROR: AXRL_SEARCH_PORT is not set. Please export it before running this script."
    exit 1
fi

# start retriever server
index_file=$save_path/bm25
corpus_file=$save_path/wiki-18.jsonl
retriever_name=bm25

python axis_recipe/search_r1/retrieval_server.py \
    --index_path "$index_file" \
    --corpus_path "$corpus_file" \
    --topk 3 \
    --retriever_name $retriever_name &

echo "Retrieval server starting on 0.0.0.0:${AXRL_SEARCH_PORT} (waiting 60s for init)..."

sleep 60s
