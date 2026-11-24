#!/bin/bash
# On the Mac, CPU only: score every fetched LayaRanker run against BM25 (paired bootstrap) and print one table.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE/../.."
R=".venv/bin/python -I -m rerank"
for s in dev dl19 dl20; do
  runs=$(ls work/runs/lr-*-live.$s.run 2>/dev/null | grep -v smoke || true)
  [ -n "$runs" ] || continue
  $R evaluate $s work/runs/bm25.$s.run $runs --baseline work/runs/bm25.$s.run > /dev/null
done
.venv/bin/python -I - <<'PY'
import json, pathlib
res = pathlib.Path("work/results")
m = {s: json.loads((res / f"metrics.{s}.json").read_text())["metrics"] for s in ("dev", "dl19", "dl20")
     if (res / f"metrics.{s}.json").exists()}
names = sorted({k.split("-live.")[0] for t in m.values() for k in t if k.startswith("lr-")})
print(f"{'run':22} {'val MRR@10':>10} {'dev MRR@10':>10} {'DL19 nDCG@10':>12} {'DL20 nDCG@10':>12}")
for n in ["bm25"] + names:
    best = pathlib.Path("work/models") / n / "best.json"
    val = json.loads(best.read_text())["mrr@10"] if best.exists() else None
    get = lambda s, k: m.get(s, {}).get(f"{n}.{s}.run" if n == "bm25" else f"{n}-live.{s}.run", {}).get(k)
    f = lambda v: "—" if v is None else f"{v:.4f}"
    print(f"{n:22} {f(val):>10} {f(get('dev', 'mrr@10')):>10} {f(get('dl19', 'ndcg@10')):>12} {f(get('dl20', 'ndcg@10')):>12}")
PY
