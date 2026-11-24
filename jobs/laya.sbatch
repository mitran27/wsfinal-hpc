#!/bin/bash
#SBATCH --account=cs_gy_6643-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/%u/wsfinal/logs/%x_%j.out
#SBATCH --comment="gpu_mps=yes"
# Submitted by jobs/submit.sh with --job-name=<tag> --time=<limit> --export=ALL,TAG=<tag>.
# This file is frozen at submit time; it pulls the repo when the job STARTS and runs jobs/run_job.sh,
# so code and runs.tsv can still be changed on GitHub until then.
R=/scratch/$USER/wsfinal-hpc
cd $R && git pull -q --rebase || echo "WARNING: git pull failed, using local copy"
echo "repo at $(git -C $R log --oneline -1)"
exec bash $R/jobs/run_job.sh
