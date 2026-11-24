#!/bin/bash
# On the Mac: copy the current LayaRanker code into this mailbox repo and push it. Jobs pull it when they start.
#   bash experiments/A100/push.sh "message"
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
rsync -a --delete --exclude '__pycache__' "$REPO/src/rerank/" "$HERE/code/src/rerank/"
mkdir -p "$HERE/code/configs" && cp "$REPO/configs/pilot.toml" "$HERE/code/configs/pilot.toml"
cd "$HERE" && git add -A && { git commit -qm "${1:-update code and runs}" && git push -q origin main && echo "pushed: $(git log --oneline -1)"; } || echo "nothing to push"
