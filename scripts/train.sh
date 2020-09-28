#!/bin/sh

# Assumes running inside docker container
SCRIPT_NAME=$0
SCRIPT_DIR=$(dirname "$0")
. $SCRIPT_DIR/env.sh

t2t-trainer \
  --generate_data \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS_SET \
  --eval_steps=$EVAL_STEPS \
  --train_steps=$TRAIN_STEPS \
  --local_eval_frequency=$EVAL_FREQ \
  --hparams=$TRAIN_HPARAMS \
  --output_dir=$TRAIN_DIR
