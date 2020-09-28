#!/usr/bin/env bash

# Assumes running inside docker container
SCRIPT_NAME=$0
SCRIPT_DIR=$(dirname "$0")
. $SCRIPT_DIR/env.sh

mkdir -p $EXPORT_DIR

# Select checkpoint number here
CHECKPOINT_PATH=$TRAIN_DIR/model.ckpt-

t2t-exporter \
    --t2t_usr_dir=$USR_DIR \
    --model=$MODEL \
    --hparams_set=$HPARAMS_SET \
    --problem=$PROBLEM \
    --data_dir=$DATA_DIR \
    --output_dir=$TRAIN_DIR \
    --hparams=$HPARAMS \
    --checkpoint_path="$CHECKPOINT_PATH" \
    --export_dir="$EXPORT_DIR" \
    --decode_hparams=$EXPORT_DECODE_HPARAMS

