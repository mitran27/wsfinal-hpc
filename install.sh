#!/bin/bash
# On the Mac: install run folders downloaded from Open OnDemand (/scratch/$USER/wsfinal/out/<tag>/, folder or .zip) into work/.
#   bash experiments/A100/install.sh ~/Downloads/lr-smoke.zip ~/Downloads/lr-core-s13 ...
#   then: bash experiments/A100/evaluate.sh
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
for src in "$@"; do
  d="$src"
  if [[ "$src" == *.zip ]]; then d=$(mktemp -d); unzip -q "$src" -d "$d"; d=$(dirname "$(find "$d" -name job.json | head -1)"); fi
  tag=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['tag'])" "$d/job.json")
  mkdir -p "$REPO/work/models/$tag" "$REPO/work/runs" "$REPO/work/results/set_logits" "$REPO/work/results/timings"
  cp "$d"/{best.pt,adapter.pt,best.json,log.jsonl,job.json,train.log} "$REPO/work/models/$tag/" 2>/dev/null || true
  cp "$d"/runs/*.run "$REPO/work/runs/" 2>/dev/null || true
  cp "$d"/set_logits/*.json "$REPO/work/results/set_logits/" 2>/dev/null || true
  cp "$d"/timings/*.json "$REPO/work/results/timings/" 2>/dev/null || true
  echo "installed $tag ($(grep -o '"a100_hours": [0-9.]*' "$d/job.json"))"
done
