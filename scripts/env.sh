#!/bin/sh
# Assumes running inside docker container
EXTERNAL_NN_DIR=/users/home/hbs24
ABS_PATH=""

LANG1=en
LANG2=is

PROBLEM=translate_enis16k_v4_tagged_bt_noised_mix3
MODEL=transformer
HPARAMS_SET=transformer_base_single_gpu

BASE_DIR=/scratch/hbs24/
EXPERIMENT_NAME=$PROBLEM
DATA_DIR=$BASE_DIR/t2t_data
TRAIN_DIR=$BASE_DIR/t2t_train/$EXPERIMENT_NAME
USR_DIR=$HOME/t2t
TMP_DIR=$BASE_DIR/t2t_tmp

DATASETS=$HOME/data/

EXPORT_DIR=$HOME/translations/$EXPERIMENT_NAME

BEAM_SIZE=4
ALPHA=0.7
EXTRA_LENGTH=64
DECODE_HPARAMS="alpha=$ALPHA,beam_size=$BEAM_SIZE,extra_length=$EXTRA_LENGTH"

# In case we want different hparams for exported model
EXPORT_DECODE_HPARAMS="alpha=$ALPHA,beam_size=$BEAM_SIZE,extra_length=$EXTRA_LENGTH"

SAMPLING_TEMP=0.5
SAMPLING_METHOD=random
SAMPLING_TOPK=5
SAMPLING_HP="sampling_temp=$SAMPLING_TEMP,sampling_method=$SAMPLING_METHOD,sampling_keep_top_k=$SAMPLING_TOPK"

TRAIN_HPARAMS="batch_size=1700,eval_drop_long_sequences=True,max_length=200"
# In order to simulate larger batch sizes, this can be used instead
#MULTISTEP_HP=",optimizer=multistep_adam,optimizer_multistep_accumulate_steps=3"
#TRAIN_HPARAMS="batch_size=1700,eval_drop_long_sequences=True,max_length=200,$MULTISTEP_HP"

TRAIN_STEPS=1125000
EVAL_STEPS=2000
EVAL_FREQ=25000

NUM_LAST_CHECKPOINTS=8
MIN_STEPS=5000
KEEP_CHECKPOINTS_MAX=20
