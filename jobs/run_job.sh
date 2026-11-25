#!/bin/bash
# One LayaRanker run (a row of runs.tsv, chosen by $TAG): LoRA fine-tune, then end-to-end reranking of the listed splits.
# Everything the run produces is copied to /scratch/$USER/wsfinal/out/<tag>/ — download that folder from Open OnDemand.
source "$(dirname "$0")/env.sh"
export HF_HUB_OFFLINE=1
row=$(grep -vE '^#|^\s*$' $R/runs.tsv | awk -F'|' -v t="$TAG" '{g=$2; gsub(/ /,"",g)} g==t')
[ -n "$row" ] || { echo "no row $TAG in runs.tsv"; exit 1; }
IFS='|' read -r stage tag variant seed hours sets splits <<< "$row"
variant=$(echo $variant); seed=$(echo $seed); splits=$(echo $splits)
A100=$(grep -oE '^# A100_SETS=.*' $R/runs.tsv | cut -d= -f2-)
ARGS=""; for s in $A100 $sets; do ARGS="$ARGS --set $s"; done
EXTRA=""; LIMIT=""; [ "$TAG" = lr-smoke ] && { EXTRA="--max-train-queries 400"; LIMIT="--limit 10"; }
cp $R/code/configs/pilot.toml $W/configs/pilot.toml
# All-eligible training pool (stage 3): BM25 candidates for 58,201 queries, ids only.
[[ "$sets" == *train58* && ! -s $W/work/candidates/train58.tsv ]] && { cat $R/ids/train58.tsv.part-* > $W/work/candidates/train58.tsv.$$ && mv $W/work/candidates/train58.tsv.$$ $W/work/candidates/train58.tsv; }   # parts < 100 MB (GitHub limit), joined atomically: jobs may start together
# Full-corpus training pool (stage 4): BM25 over all 8.8M MS MARCO passages for every judged training query, ids only,
# plus the full collection text, unpacked from the official archive that setup downloaded.
if [[ "$sets" == *train500* ]]; then
  [ -s $W/work/candidates/train500.tsv ] || { cat $R/ids/train500.tsv.part-* > $W/work/candidates/train500.tsv.$$ && mv $W/work/candidates/train500.tsv.$$ $W/work/candidates/train500.tsv; }
  [ -s $W/data/msmarco/collection.tsv ] || { tar xzf $W/download/collection.tar.gz -O > $W/data/msmarco/collection.tsv.$$ && mv $W/data/msmarco/collection.tsv.$$ $W/data/msmarco/collection.tsv; }
  [ "$(wc -l < $W/data/msmarco/collection.tsv)" = 8841823 ] || { echo "full collection incomplete"; exit 4; }
fi
M=$W/work/models/$TAG; O=$W/out/$TAG; mkdir -p $M $O; LOG=$O/train.log; start=$(date +%s)
echo "TAG=$TAG variant=$variant seed=$seed sets=[$A100 $sets] splits=[$splits] start $(date)" | tee $LOG
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee -a $LOG
# Never train on the CPU by accident: stop at once if this torch build cannot use the GPU.
py -c 'import torch; ok = torch.cuda.is_available(); print("torch", torch.__version__, "cuda build", torch.version.cuda, "GPU visible:", ok); raise SystemExit(0 if ok else 3)' 2>&1 | tee -a $LOG
[ ${PIPESTATUS[0]} = 0 ] || { echo "STOPPED: torch cannot see the GPU (driver/CUDA mismatch); no training was run" | tee -a $LOG; exit 3; }
rerank $ARGS finetune $variant --seed $seed --tag $TAG $EXTRA 2>&1 | grep -v -i "warn" | tee -a $LOG
RC=${PIPESTATUS[0]}
if [ $RC = 0 ]; then
  rerank $ARGS live $M/best.pt $splits $LIMIT 2>&1 | grep -v -i "warn" | tee -a $LOG
  RC=${PIPESTATUS[0]}
fi
hours=$(echo "($(date +%s) - $start) / 3600" | bc -l)
printf '{"tag": "%s", "a100_hours": %.3f, "slurm_job": "%s", "rc": %d}\n' $TAG $hours "$SLURM_JOB_ID" $RC > $O/job.json
cp $M/best.pt $M/adapter.pt $M/best.json $M/log.jsonl $O/ 2>/dev/null
mkdir -p $O/runs $O/set_logits $O/timings
cp $W/work/runs/$TAG-live.*.run $O/runs/ 2>/dev/null
cp $W/work/results/set_logits/$TAG-live.*.json $O/set_logits/ 2>/dev/null
cp $W/work/results/timings/$TAG-live.*.json $O/timings/ 2>/dev/null
[ $RC = 0 ] && cp $O/job.json $M/job.json
echo "end $(date) rc=$RC a100_hours=$(printf %.2f $hours)  ->  download $O" | tee -a $LOG
