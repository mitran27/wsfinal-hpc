#!/bin/bash
# ONCE, on a login node (it has internet): unpack the data, build the venv, download the backbones.
#   bash /scratch/$USER/wsfinal-hpc/jobs/setup_env.sh
set -e
source "$(dirname "$0")/env.sh"
mkdir -p $W/configs $W/logs $HF_HOME $SINGULARITY_TMPDIR
echo "1/4 data (the data branch: course collection, BM25 candidates, splits, queries, qrels)"
git -C $R fetch -q origin data
git -C $R ls-tree --name-only origin/data | grep '^bundle' | sort | while read p; do git -C $R show "origin/data:$p"; done | tar xz -C $W
ls $W/data $W/work
echo "2/4 venv on top of the overlay's Python (keeps its torch; adds the pinned transformers stack)"
singularity exec --overlay $OVL:ro $SIF bash -c "
  python -m venv --system-site-packages $W/venv &&
  $W/venv/bin/pip install -q --upgrade pip &&
  $W/venv/bin/pip install -q 'transformers==5.24.0' 'safetensors>=0.4' 'huggingface_hub>=0.25' scipy"
echo "3/4 backbones into $HF_HOME (compute nodes then run offline)"
py - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('convaiinnovations/laya', revision='1c5edc17a7acd8701df6fc341c0d179f1c62c982',
                  allow_patterns=['encoder/config.json', 'tokenizer/*', 'model.safetensors'])
snapshot_download('answerdotai/ModernBERT-large', allow_patterns=['*.json', '*.safetensors', '*.txt'])
print('backbones cached')
PY
echo "4/4 check"
py -c 'import torch, transformers, rerank; print("torch", torch.__version__, "transformers", transformers.__version__, "rerank ok")'
echo "SETUP OK. Next: bash $R/jobs/submit.sh 0"
