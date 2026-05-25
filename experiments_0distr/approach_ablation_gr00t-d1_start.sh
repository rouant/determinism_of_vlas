#!/bin/bash
### INIT
source /opt/conda/etc/profile.d/conda.sh
conda activate gr00t-B

## SETTINGS
SEED=4200
MODEL="gr00t"
DEGREE=1

### START
python -u approach_ablation_run.py -s "${SEED}" -m "${MODEL}" -d "${DEGREE}" > "approach_ablation_${MODEL}_degree-${DEGREE}_seed-${SEED}.metrics" # 0 to 5
### RESUME
python -u approach_ablation_resume.py -s "${SEED}" -m "${MODEL}" -d "${DEGREE}" >> "approach_ablation_${MODEL}_degree-${DEGREE}_seed-${SEED}.metrics" # 5 to 10
python -u approach_ablation_resume.py -s "${SEED}" -m "${MODEL}" -d "${DEGREE}" >> "approach_ablation_${MODEL}_degree-${DEGREE}_seed-${SEED}.metrics" # 10 to 15
python -u approach_ablation_resume.py -s "${SEED}" -m "${MODEL}" -d "${DEGREE}" >> "approach_ablation_${MODEL}_degree-${DEGREE}_seed-${SEED}.metrics" # 15 to 20
