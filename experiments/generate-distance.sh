#!/bin/bash

# 10 digits number
SEED=1234666666

# action corresponding to supported tasks in simplerenv: "grasp", "move", "put-on", "put-in"
ACTION="grasp"

GRANULARITY=5
# size of the dataset to generate
SIZE=$(( (GRANULARITY+1)**2))

FILENAME="t-${ACTION}_n-${SIZE}_o-m3_s-${SEED}.json"
PROMPT_LOCATION="./"
DATA_LOCATION="../data/"


PROMPT="pick coke can"

# begin prompt file
echo "[" > "${PROMPT_LOCATION}${FILENAME}"

for i in $(seq 1 1 $SIZE); do
    echo "  \"${PROMPT}\"," >> "${PROMPT_LOCATION}${FILENAME}"
done
# remove trailin comma
truncate -s -2 "${PROMPT_LOCATION}${FILENAME}"

# end prompt file
echo "]" >> "${PROMPT_LOCATION}${FILENAME}"


X_MIN=-0.65
X_MAX=-0.05
Y_MIN=-0.1
Y_MAX=0.5

# begin data file
echo "{" > "${DATA_LOCATION}${FILENAME}"

idx=0
X_INCR=$(awk "BEGIN { print ($X_MAX - $X_MIN) / $GRANULARITY }")
Y_INCR=$(awk "BEGIN { print ($Y_MAX - $Y_MIN) / $GRANULARITY }")

for X in $(seq $X_MIN $X_INCR $X_MAX); do
    for Y in $(seq $Y_MIN $Y_INCR $Y_MAX); do
        echo "  \"${idx}\": {
    \"model_id\": \"coke_can\", 
    \"obj_init_options\": {
      \"init_xy\": [${X}, ${Y}], 
      \"orientation\": [0.5, 0.5, 0.5, 0.5]
    }
  }, 
" >> "${DATA_LOCATION}${FILENAME}"
        ((idx++))
    done
done
# remove trailin comma
truncate -s -2 "${DATA_LOCATION}${FILENAME}"

# end data file
echo "  \"seed\": ${SEED}, 
  \"num\": ${SIZE}
}" >> "${DATA_LOCATION}${FILENAME}"