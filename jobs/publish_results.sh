#!/bin/bash
# usage: bash jobs/publish_results.sh TAG [STATE] [NOTE]     STATE = SUBMITTED | STARTED | RUNNING | DONE | FAILED
# Commits status (+ results when DONE/FAILED) to the `results` branch (a worktree of this repo) and pushes it.
TAG=$1; STATE=${2:-DONE}; NOTE=$3
source "$(dirname "$0")/env.sh"
M=$W/work/models/$TAG; T=/scratch/$USER/wsfinal-results; LOCK=/scratch/$USER/wsfinal-results.lock
for i in $(seq 1 90); do                                   # one publisher at a time
  mkdir $LOCK 2>/dev/null && break
  [ -n "$(find $LOCK -maxdepth 0 -mmin +15 2>/dev/null)" ] && rmdir $LOCK 2>/dev/null
  sleep 10
done
trap 'rmdir $LOCK 2>/dev/null' EXIT
[ -d $T ] && [ ! -e $T/.git ] && rm -rf $T
cd $R || exit 1
if [ ! -e $T/.git ]; then
  git worktree prune
  if git fetch -q origin results 2>/dev/null; then git worktree add -q $T -B results origin/results
  else git worktree add -q --detach $T && cd $T && git checkout -q --orphan results && git rm -rqf . && cd $R; fi
fi
cd $T || exit 1
git pull -q --rebase origin results 2>/dev/null || true
D=$T/$TAG; mkdir -p $D $T/status
if [ -f $M/train.log ]; then
  grep -E "^TAG=|training queries|step [0-9]+/|^end |Error|Traceback|OutOfMemory|work/runs" $M/train.log | tail -n 60 > $D/summary.txt
  tail -n 40 $M/train.log > $D/log_tail.txt
  [ -s $M/gpu_util.csv ] && awk -F', ' '{gsub(/ %/,"",$2); s+=$2; n++} END {if (n) printf "GPU utilisation: mean %.0f%% over %d samples\n", s/n, n}' $M/gpu_util.csv >> $D/summary.txt
fi
if [ "$STATE" = DONE ] || [ "$STATE" = FAILED ]; then
  for f in $M/best.pt $M/adapter.pt $M/best.json $M/log.jsonl $M/job.json; do
    [ -f "$f" ] && [ $(stat -c%s "$f") -lt 95000000 ] && cp "$f" $D/
  done
  mkdir -p $D/runs $D/set_logits $D/timings; cp $W/work/runs/$TAG-live.*.run $D/runs/ 2>/dev/null
  cp $W/work/results/set_logits/$TAG-live.*.json $D/set_logits/ 2>/dev/null
  cp $W/work/results/timings/$TAG-live.*.json $D/timings/ 2>/dev/null
fi
LAST=$(grep -E "step [0-9]+/" $M/train.log 2>/dev/null | tail -n 1 | sed -E 's/^\[[^]]*\] //' | cut -c1-80)
echo "$STATE | $TAG | $(date '+%F %H:%M') | job ${SLURM_JOB_ID:-manual} on $(hostname -s) | $LAST $NOTE" > $T/status/$TAG.txt
used=$(sacct -X -n -P -S 2026-11-24 -u $USER -o JobName,ElapsedRaw 2>/dev/null | awk -F'|' '$1 ~ /^lr-/ {s+=$2} END {printf "%.2f", s/3600}')
{ echo "# LayaRanker A100 status ($(date '+%F %H:%M %Z'))"; echo; echo "A100 hours used by lr-* jobs: $used"; echo
  echo '```'; cat $T/status/*.txt | sort -t'|' -k2; echo '```'; } > $T/STATUS.md
git add -A; git commit -qm "$STATE $TAG $(date +%F_%H%M)" || true
for i in 1 2 3 4; do
  git push -q origin results 2>/dev/null && { echo "published $STATE $TAG"; exit 0; }
  git pull -q --rebase origin results 2>/dev/null; sleep 15
done
echo "WARNING: could not push $STATE $TAG (committed locally in $T)"; exit 1
