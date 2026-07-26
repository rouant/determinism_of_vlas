#!/bin/bash
### INIT
source /opt/conda/etc/profile.d/conda.sh
conda activate gr00t-B

## SETTINGS
SEED=4201
MODEL="gr00t"


### START
python -u approach_random_run.py -s "${SEED}" -m "${MODEL}" > "approach_random_additional_0_${MODEL}_seed-${SEED}.metrics" # 0 to 5
### RESUME
python -u approach_random_resume.py -s "${SEED}" -m "${MODEL}" >> "approach_random_additional_0_${MODEL}_seed-${SEED}.metrics" # 5 to 10
python -u approach_random_resume.py -s "${SEED}" -m "${MODEL}" >> "approach_random_additional_0_${MODEL}_seed-${SEED}.metrics" # 10 to 15
python -u approach_random_resume.py -s "${SEED}" -m "${MODEL}" >> "approach_random_additional_0_${MODEL}_seed-${SEED}.metrics" # 15 to 20
