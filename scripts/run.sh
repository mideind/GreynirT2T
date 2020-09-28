##!/usr/bin/env bash

NN_DIR="/home/haukur"
T2T_VERSION=1.14.1
IMAGE="t2t-${T2T_VERSION}"

#docker run --runtime=nvidia \
docker run \
       --name "translate_enis16k_v4-run" \
       --rm \
       --interactive \
       --tty \
       --shm-size=1g \
       --ulimit memlock=-1 \
       --volume $NN_DIR/t2t_datagen:/t2t_tmp \
       --volume $NN_DIR/t2t_train:/t2t_train \
       --volume $NN_DIR/t2t_data:/t2t_data \
       --volume $NN_DIR/models:/models \
       --volume $NN_DIR/data:/data/ \
       --volume $NN_DIR/t2t_usr:/t2t_usr \
       --volume $NN_DIR/scripts:/scripts \
       --env HOST_PERMS="$(id -u):$(id -g)" \
       $IMAGE /scripts/enis-v4/train.sh
