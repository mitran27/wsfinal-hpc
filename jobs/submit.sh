#!/bin/bash
# Submit every run of the given stages (runs.tsv) that is not done or queued, within the A100 budget.
#   cd /scratch/$USER/wsfinal-hpc && git pull && bash jobs/submit.sh 0      (smoke, ≈ 10 min)
#   ... bash jobs/submit.sh 1   /   bash jobs/submit.sh 2
# Budget: hours used by finished lr-* jobs + time limits of queued/running lr-* jobs + the new limit ≤ BUDGET.
source "$(dirname "$0")/env.sh"
BUDGET=$(grep -oE '^# BUDGET_HOURS=[0-9.]+' $R/runs.tsv | cut -d= -f2)
[ $# -ge 1 ] || { echo "usage: submit.sh <stage> [stage...]"; exit 1; }
[ -f $W/.setup_ok ] || { echo "setup has not finished: run  bash $R/jobs/setup_env.sh  first"; exit 1; }
used=$(sacct -X -n -P -S 2026-11-24 -u $USER -o JobName,ElapsedRaw | awk -F'|' '$1 ~ /^lr-/ {s+=$2} END {printf "%.2f", s/3600}')
reserved=$(squeue -h -u $USER -o '%j %l' | awk '$1 ~ /^lr-/ {n=split($2,t,":"); s+=(n==3? t[1]+t[2]/60 : t[1]/60)} END {printf "%.2f", s+0}')
total=$(echo "$used + $reserved" | bc)
echo "A100 hours: used $used, reserved by queued/running jobs $reserved, budget $BUDGET"
queued=$(squeue -h -u $USER -o '%j')
grep -vE '^#|^\s*$' $R/runs.tsv | while IFS='|' read -r stage tag variant seed hours sets splits; do
  stage=$(echo $stage); tag=$(echo $tag); hours=$(echo $hours)
  [[ " $* " == *" $stage "* ]] || continue
  [ -f $W/work/models/$tag/job.json ] && { echo "$tag: already done"; continue; }
  grep -qx "$tag" <<< "$queued" && { echo "$tag: already queued"; continue; }
  new=$(echo "$total + $hours" | bc)
  if (( $(echo "$new > $BUDGET" | bc) )); then echo "$tag: SKIPPED, would reach $new of $BUDGET A100 hours"; continue; fi
  limit=$(printf '%d:%02d:00' "${hours%.*}" "$(echo "($hours - ${hours%.*}) * 60 / 1" | bc)")
  ID=$(sbatch --parsable -J $tag --time=$limit --export=ALL,TAG=$tag $R/jobs/laya.sbatch) && echo "$tag -> job $ID ($limit)"
  total=$new
done
