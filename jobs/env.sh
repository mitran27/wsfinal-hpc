# Sourced by every script: paths, and `py` = Python inside the CUDA container with our venv.
R=/scratch/$USER/wsfinal-hpc          # this repo (code, runs.tsv, jobs)
W=/scratch/$USER/wsfinal              # work area: data/, work/, configs/, venv/, logs/
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVL=/scratch/$USER/overlays/overlay.ext3     # the existing Python overlay (mounted read-only; our packages live in $W/venv)
export HF_HOME=/scratch/$USER/hf-cache PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export SINGULARITY_TMPDIR=/scratch/$USER/singularity-tmp SINGULARITY_CACHEDIR=/scratch/$USER/singularity-cache
py() { singularity exec --nv --pwd $W --overlay $OVL:ro $SIF env PYTHONPATH=$R/code/src $W/venv/bin/python "$@"; }
rerank() { py -I -m rerank --config $W/configs/pilot.toml "$@"; }
