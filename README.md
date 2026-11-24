# wsfinal-hpc: LayaRanker A100 mailbox for NYU Torch

No persistent agent and no SSH from the Mac. Claude pushes code and `runs.tsv` here; you submit from the Open OnDemand shell;
every job pulls this repo when it starts and publishes its status and results to the `results` branch (deploy key).
This repo holds no MS MARCO text (licensed for non-commercial research): `ids/` has only ids, and `jobs/setup_env.sh`
downloads the text from the official MS MARCO site and checks the rebuilt 1M-passage collection against a checksum.

Account `cs_gy_6643-2026fa`, partition `c12m85-a100-1`, one A100 per job. Budget: `BUDGET_HOURS` in `runs.tsv`
(20 A100-hours for Phase 1). `jobs/submit.sh` refuses any job that would exceed it.

## On Torch (Open OnDemand shell)

Once:
```bash
git clone https://github.com/mitran27/wsfinal-hpc.git /scratch/$USER/wsfinal-hpc
bash /scratch/$USER/wsfinal-hpc/jobs/setup_env.sh        # data (≈ 1 GB download), venv, backbones; login node, ≈ 15 min
```
Each wave (Claude says which stage):
```bash
cd /scratch/$USER/wsfinal-hpc && git pull && bash jobs/submit.sh 0      # 0 = smoke, 1 = core comparisons, 2 = seeds
```
Status: the `results` branch's `STATUS.md` (Claude reads it), or `squeue -u $USER`.

## On the Mac

`bash experiments/A100/push.sh "msg"` sends code changes; `bash experiments/A100/fetch.sh && bash experiments/A100/evaluate.sh`
installs finished runs into `work/` and scores them on CPU.

## Phase 1 runs (`runs.tsv`)

| Run | Question |
|---|---|
| `lr-smoke` | Does everything work on Torch, and how many seconds per step (re-sizes every later estimate)? |
| `lr-pointwise` | Laya + LoRA scoring each passage alone |
| `lr-candidate` | + attention across the candidate pool |
| `lr-ffn` | Same parameters, no cross-candidate attention: attention or capacity? |
| `lr-core` | Candidate attention + [SET] + ordinal head: the LayaRanker stage-1 core |
| `lr-core-mbl` | The core on plain ModernBERT-large: does Laya's decision pre-training help? |
| seeds 14, 15 | Pointwise vs core with three seeds each |
