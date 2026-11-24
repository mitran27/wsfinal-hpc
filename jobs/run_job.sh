#!/bin/bash
# One LayaRanker run (a row of runs.tsv, chosen by $TAG): LoRA fine-tune, then end-to-end reranking of the listed splits.
# Reports SUBMITTED / STARTED / RUNNING (every 20 min) / DONE / FAILED to the `results` branch.
source "$(dirname "$0")/env.sh"
export HF_HUB_OFFLINE=1
pub() { bash $R/jobs/publish_results.sh $TAG "$1" "$2" || echo "WARNING: publish $1 failed"; }
row=$(grep -vE '^#|^\s*$' $R/runs.tsv | awk -F'|' -v t="$TAG" '{g=$2; gsub(/ /,"",g)} g==t')
[ -n "$row" ] || { echo "no row $TAG in runs.tsv"; pub FAILED "no row in runs.tsv"; exit 1; }
IFS='|' read -r stage tag variant seed hours sets splits <<< "$row"
variant=$(echo $variant); seed=$(echo $seed); splits=$(echo $splits)
A100=$(grep -oE '^# A100_SETS=.*' $R/runs.tsv | cut -d= -f2-)
ARGS=""; for s in $A100 $sets; do ARGS="$ARGS --set $s"; done
EXTRA=""; LIMIT=""; [ "$TAG" = lr-smoke ] && { EXTRA="--max-train-queries 400"; LIMIT="--limit 10"; }
cp $R/code/configs/pilot.toml $W/configs/pilot.toml
M=$W/work/models/$TAG; mkdir -p $M; LOG=$M/train.log; start=$(date +%s)
echo "TAG=$TAG variant=$variant seed=$seed sets=[$A100 $sets] splits=[$splits] start $(date)" | tee $LOG
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a $LOG
pub STARTED
( while sleep 1200; do bash $R/jobs/publish_results.sh $TAG RUNNING >/dev/null 2>&1; done ) & HB=$!
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 30 > $M/gpu_util.csv 2>/dev/null & GU=$!
rerank $ARGS finetune $variant --seed $seed --tag $TAG $EXTRA 2>&1 | grep -v -i "warn" | tee -a $LOG
RC=${PIPESTATUS[0]}
if [ $RC = 0 ]; then
  rerank $ARGS live $M/best.pt $splits $LIMIT 2>&1 | grep -v -i "warn" | tee -a $LOG
  RC=${PIPESTATUS[0]}
fi
kill $HB $GU 2>/dev/null
hours=$(echo "($(date +%s) - $start) / 3600" | bc -l)
printf '{"tag": "%s", "a100_hours": %.3f, "slurm_job": "%s", "rc": %d}\n' $TAG $hours "$SLURM_JOB_ID" $RC > $M/job.json
echo "end $(date) rc=$RC a100_hours=$(printf %.2f $hours)" | tee -a $LOG
git -C $R pull -q --rebase 2>/dev/null
if [ $RC = 0 ]; then pub DONE; else rm -f $M/job.json; pub FAILED "exit code $RC"; fi
