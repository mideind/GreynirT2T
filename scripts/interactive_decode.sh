#!/bin/sh

# Assumes running inside docker container
SCRIPT_NAME=$0
SCRIPT_DIR=$(dirname "$0")
. $SCRIPT_DIR/env.sh

t2t-decoder \
    --t2t_usr_dir=$USR_DIR \
    --data_dir=$DATA_DIR \
    --problem=$PROBLEM \
    --tmp_dir=$TMP_DIR \
    --model=$MODEL \
    --hparams_set=$HPARAMS_SET \
    --hparams=$TRAIN_HPARAMS \
    --output_dir=$TRAIN_DIR \
    --decode_interactive \
    --decode_hparams="$DECODE_HPARAMS"
