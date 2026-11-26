#!/bin/bash
# On the Mac: install run folders downloaded from Open OnDemand into work/.
# Accepts the whole out/ folder (or out.zip / out.tgz) or single run folders; every folder with a job.json is installed.
#   bash experiments/A100/install.sh ~/Downloads/out.zip
#   then: bash experiments/A100/evaluate.sh
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
for src in "$@"; do
  d="$src"
  case "$src" in
    *.zip) d=$(mktemp -d); unzip -q "$src" -d "$d" ;;
    *.tgz|*.tar.gz) d=$(mktemp -d); tar xzf "$src" -C "$d" ;;
    *.tar) d=$(mktemp -d); tar xf "$src" -C "$d" ;;
  esac
  find "$d" -name job.json -not -path '*/__MACOSX/*' | while read -r j; do
    r=$(dirname "$j")
    read -r tag rc < <(python3 -c "import json,sys; j=json.load(open(sys.argv[1])); print(j['tag'], j['rc'])" "$j")
    [ "$rc" = 0 ] || { echo "skipped $tag (rc=$rc, failed run)"; continue; }
    mkdir -p "$REPO/work/models/$tag" "$REPO/work/runs" "$REPO/work/results/set_logits" "$REPO/work/results/timings"
    cp "$r"/{best.pt,adapter.pt,best.json,log.jsonl,job.json,train.log} "$REPO/work/models/$tag/" 2>/dev/null || true
    cp "$r"/runs/*.run "$REPO/work/runs/" 2>/dev/null || true
    cp "$r"/set_logits/*.json "$REPO/work/results/set_logits/" 2>/dev/null || true
    cp "$r"/timings/*.json "$REPO/work/results/timings/" 2>/dev/null || true
    echo "installed $tag ($(grep -o '"a100_hours": [0-9.]*' "$j"))"
  done
done
