#!/bin/bash -l
# Score a checkpoint on the dev split with the official runner and evaluator:
# noisy input (no enhancement), the model on noisy input, and the model on
# clean input (clean preservation), then the leaderboard formula.
#
#   qsub -N eval_dccrn -v RUN=dccrn_seed0 jobs/eval_dev.sh
#
#$ -P ec523kb
#$ -l h_rt=02:00:00
#$ -l gpus=1
#$ -l gpu_type=L40S
#$ -pe omp 4
#$ -j y
#$ -cwd
#$ -o runs/

set -eo pipefail  # not -u: conda activation scripts read unset variables
RUN=${RUN:?set RUN}
ROOT=$PWD
OUT=$ROOT/runs/$RUN/dev_eval
CKPT=${CKPT:-$ROOT/runs/$RUN/model.safetensors}

module load miniconda academic-ml/spring-2026
conda activate spring-2026-pyt
source .venv/bin/activate
cd cli

# The noisy-input row depends only on the data, so compute it once.
NOISY_REPORT=$ROOT/runs/noisy_input_dev_report.json
[ -f "$NOISY_REPORT" ] || python evaluate.py --clean ../data/dev/clean \
    --enhanced ../data/dev/noisy --report "$NOISY_REPORT"

rm -rf "$OUT"
python enhance.py --checkpoint "$CKPT" --input ../data/dev/noisy \
    --output "$OUT/enhanced_noisy" --device cuda
python enhance.py --checkpoint "$CKPT" --input ../data/dev/clean \
    --output "$OUT/enhanced_clean" --device cuda
python evaluate.py --clean ../data/dev/clean --enhanced "$OUT/enhanced_noisy" \
    --report "$OUT/noisy_report.json"
python evaluate.py --clean ../data/dev/clean --enhanced "$OUT/enhanced_clean" \
    --report "$OUT/clean_report.json"

python -"$NOISY_REPORT" "$OUT/noisy_report.json" "$OUT/clean_report.json" <<'EOF'
import json, sys
names = ["noisy input", "model (noisy in)", "model (clean in)"]
print(f"{'':18s} {'PESQ':>6s} {'STOI':>7s} {'SI-SDR':>7s} {'A(D)':>7s}")
for name, path in zip(names, sys.argv[1:]):
    r = json.load(open(path))
    print(f"{name:18s} {r['pesq']:6.3f} {r['stoi']:7.4f} {r['si_sdr']:7.2f} {r['quality']:7.4f}")
EOF
echo "== leaderboard formula on dev (model) =="
python score.py --noisy-report "$OUT/noisy_report.json" --clean-report "$OUT/clean_report.json"
