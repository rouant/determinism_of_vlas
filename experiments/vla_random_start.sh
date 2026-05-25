#!/bin/bash
### INIT
source /opt/conda/etc/profile.d/conda.sh
conda activate pi0-B
### START
python -u vla_random_run.py #> test_random_3simil_2obj.txt
### RESUME
python -u vla_random_resume.py #>> test_random_3simil_2obj.txt
python -u vla_random_resume.py #>> test_random_3simil_2obj.txt
python -u vla_random_resume.py #>> test_random_3simil_2obj.txt