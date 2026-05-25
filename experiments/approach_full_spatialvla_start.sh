#!/bin/bash
### INIT
source /opt/conda/etc/profile.d/conda.sh
conda activate pymoo-B

## SETTINGS
SEED=5000
MODEL="spatialvla"

### START
python -u approach_full_run.py -s "${SEED}" -m "${MODEL}" > "approach_full_${MODEL}_seed-${SEED}.metrics" # 0 to 5
### RESUME
python -u approach_full_resume.py -s "${SEED}" -m "${MODEL}" >> "approach_full_${MODEL}_seed-${SEED}.metrics" # 5 to 10
python -u approach_full_resume.py -s "${SEED}" -m "${MODEL}" >> "approach_full_${MODEL}_seed-${SEED}.metrics" # 10 to 15
python -u approach_full_resume.py -s "${SEED}" -m "${MODEL}" >> "approach_full_${MODEL}_seed-${SEED}.metrics" # 15 to 20
