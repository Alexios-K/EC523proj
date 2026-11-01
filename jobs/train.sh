#!/bin/bash -l
# Train the submission model on one L40S. Resubmitting with the same RUN
# resumes from runs/$RUN/last.pt.
#
#   qsub -N dccrn_seed0 -v RUN=dccrn_seed0,SEED=0 jobs/train.sh
#
# Arguments after the script name are passed to train.py, e.g.
#   qsub -N dccrn_comp -v RUN=dccrn_comp_seed0 jobs/train.sh --loss sisnr+comp
#
#$ -P ec523kb
#$ -l h_rt=12:00:00
#$ -l gpus=1
#$ -l gpu_type=L40S
#$ -pe omp 8
#$ -j y
#$ -cwd
#$ -o runs/

set -euo pipefail
RUN=${RUN:?set RUN, e.g. qsub -v RUN=dccrn_seed0 jobs/train.sh}
SEED=${SEED:-0}

module load miniconda academic-ml/spring-2026
conda activate spring-2026-pyt
source .venv/bin/activate

echo "host $(hostname)  job ${JOB_ID:-local}  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
python train/train.py --run "runs/$RUN" --seed "$SEED" --workers "${NSLOTS:-8}" "$@"
