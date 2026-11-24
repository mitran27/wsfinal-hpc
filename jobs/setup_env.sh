#!/bin/bash
# ONCE, on a login node (it has internet): unpack the data, build the venv, download the backbones.
#   bash /scratch/$USER/wsfinal-hpc/jobs/setup_env.sh
set -e
source "$(dirname "$0")/env.sh"
mkdir -p $W/configs $W/logs $HF_HOME $SINGULARITY_TMPDIR
echo "1/4 data: MS MARCO text from the official site + our ids (ids/), then a checksum against the course file"
D=$W/download; mkdir -p $D $W/data/msmarco
U=https://msmarco.z22.web.core.windows.net/msmarcoranking
for f in collection.tar.gz queries.tar.gz qrels.train.tsv msmarco-test2019-queries.tsv.gz msmarco-test2020-queries.tsv.gz; do
  [ -s $D/$f ] || curl -fsSL --retry 3 -o $D/$f $U/$f
done
tar xzf $D/queries.tar.gz -C $D
cp $D/queries.train.tsv $D/qrels.train.tsv $W/data/msmarco/
cp $D/queries.dev.tsv $W/data/queries.dev.tsv
gunzip -c $D/msmarco-test2019-queries.tsv.gz $D/msmarco-test2020-queries.tsv.gz > $W/data/queries.eval.tsv
cp $R/ids/qrels.dev.tsv $R/ids/qrels.eval.one.tsv $R/ids/qrels.eval.two.tsv $W/data/
if ! (cd $W/data && sha256sum -c --quiet $R/ids/collection_subset.sha256 2>/dev/null); then
  gunzip -c $R/ids/subset_pids.txt.gz > $D/subset_pids.txt
  tar xzf $D/collection.tar.gz -O | awk -F'\t' 'NR==FNR {keep[$1]; next} ($1 in keep)' $D/subset_pids.txt - > $W/data/collection_subset.tsv
  (cd $W/data && sha256sum -c $R/ids/collection_subset.sha256) || { echo "collection does not match the course file"; exit 1; }
fi
tar xzf $R/ids/work.tgz -C $W
ls $W/data $W/work
echo "2/4 venv from the overlay's Python with torch + the pinned transformers stack (≈ 3 GB, a few minutes)"
[ -f $OVL ] || { echo "no overlay at $OVL"; exit 1; }
box 'echo "overlay python: $(command -v python || command -v python3)"'
# torch: the version the earlier cv4 jobs used from /scratch/$USER/pylib (known to work with these nodes' drivers),
# installed into our own venv so none of pylib's older libraries shadow ours.
TV=$(ls -d /scratch/$USER/pylib/torch-*.dist-info 2>/dev/null | head -1 | sed -E 's/.*torch-([^-]+)\.dist-info/\1/')
TORCH="torch${TV:+==$TV}"; echo "torch to install: $TORCH"
export PIP_CACHE_DIR=/scratch/$USER/pip-cache
rm -rf $W/venv
box "P=\$(command -v python || command -v python3); \$P -m venv $W/venv"
box "$W/venv/bin/pip install -q --upgrade pip && $W/venv/bin/pip install -q '$TORCH' && \
     $W/venv/bin/pip install -q 'transformers==5.24.0' 'safetensors>=0.4' 'huggingface_hub>=0.25' numpy scipy"
box "$W/venv/bin/python -c 'import torch, transformers; print(\"venv torch\", torch.__version__, \"cuda build\", torch.version.cuda, \"transformers\", transformers.__version__)'"
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
if [ -f ~/.ssh/wsfinal_hpc ]; then
  git -C $R remote set-url --push origin git@github-wsfinal:mitran27/wsfinal-hpc.git
  echo "results will be pushed with the deploy key"
else
  echo "NOTE: no ~/.ssh/wsfinal_hpc key, so jobs cannot push results to GitHub (they stay in /scratch/$USER/wsfinal-results)"
fi
date > $W/.setup_ok
echo "SETUP OK. Next: bash $R/jobs/submit.sh 0"
