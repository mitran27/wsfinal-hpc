#!/bin/bash
# On the Mac: pull the results branch and install finished runs into work/ (models, run files, [SET] logits, timings).
#   bash experiments/A100/fetch.sh && bash experiments/A100/evaluate.sh
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"; T="$HERE/.results"
cd "$HERE" && git fetch -q origin results
[ -e "$T/.git" ] || git worktree add -q "$T" origin/results
git -C "$T" checkout -q --detach origin/results
sed -n '1,40p' "$T/STATUS.md"
for d in "$T"/lr-*/; do
  tag=$(basename "$d"); [ -f "$d/job.json" ] || continue
  mkdir -p "$REPO/work/models/$tag" "$REPO/work/runs" "$REPO/work/results/set_logits" "$REPO/work/results/timings"
  cp "$d"/{best.pt,adapter.pt,best.json,log.jsonl,job.json} "$REPO/work/models/$tag/" 2>/dev/null || true
  cp "$d"/runs/*.run "$REPO/work/runs/" 2>/dev/null || true
  cp "$d"/set_logits/*.json "$REPO/work/results/set_logits/" 2>/dev/null || true
  cp "$d"/timings/*.json "$REPO/work/results/timings/" 2>/dev/null || true
  echo "installed $tag"
done
