# wsfinal-hpc: LayaRanker A100 jobs on NYU Torch

Clone once, set up once, then dispatch jobs. Each job writes everything it produces to
`/scratch/$USER/wsfinal/out/<tag>/`; download that folder from the Open OnDemand file browser.
No MS MARCO text is stored here: `ids/` has only ids, and setup downloads the text from the official site
(checked against the course file's checksum).

Account `cs_gy_6643-2026fa`, partition `c12m85-a100-1`, one A100 per job. `jobs/submit.sh` refuses any job
that would take total A100 hours past `BUDGET_HOURS` in `runs.tsv` (20 for Phase 1).

```bash
git clone https://github.com/mitran27/wsfinal-hpc.git /scratch/$USER/wsfinal-hpc     # once
bash /scratch/$USER/wsfinal-hpc/jobs/setup_env.sh                                     # once
cd /scratch/$USER/wsfinal-hpc && git pull && bash jobs/submit.sh 0                    # 0 smoke, 1 core, 2 seeds
squeue -u $USER                                                                       # progress
```

On the Mac: `bash experiments/A100/install.sh ~/Downloads/out.tgz` (the whole out/ folder, or single run folders/zips) then `bash experiments/A100/evaluate.sh`.
Code changes: `bash experiments/A100/push.sh "msg"` (jobs pull the latest code when they start).

| Run | Question |
|---|---|
| `lr-smoke` | Does it work on Torch, and how many seconds per step? |
| `lr-pointwise` | Laya + LoRA scoring each passage alone |
| `lr-candidate` | + attention across the candidate pool |
| `lr-ffn` | Same parameters, no cross-candidate attention |
| `lr-core` | Candidate attention + [SET] + ordinal head (LayaRanker stage-1 core) |
| `lr-core-mbl` | The core on plain ModernBERT-large |
| seeds 14, 15 | Pointwise vs core, three seeds each |
