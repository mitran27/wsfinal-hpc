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
M=$W/work/models/$TAG; O=$W/out/$TAG; mkdir -p $M $O; LOG=$O/train.log; start=$(date +%s)
echo "TAG=$TAG variant=$variant seed=$seed sets=[$A100 $sets] splits=[$splits] start $(date)" | tee $LOG
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a $LOG
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
