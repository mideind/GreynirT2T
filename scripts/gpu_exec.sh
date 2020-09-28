#!/usr/bin/env bash

T2T_VERSION=1.14.1
IMAGE="t2t-${T2T_VERSION}"
BASENAME=$(basename $(dirname $0))

# base directory where the run files will be stored
NN_DIR=

COMMAND_NAME=$(basename $1)
COMMAND=$1
shift

docker run --runtime=nvidia \
       --name "$IMAGE-exec-$COMMAND_NAME" \
       --rm \
       --interactive \
       --tty \
       --shm-size=1g \
       --ulimit memlock=-1 \
       --volume $NN_DIR/train_runs:/t2t_train \
       --volume $NN_DIR/compiled_data:/t2t_data \
       --volume $NN_DIR/models:/models \
       --volume $NN_DIR/data:/data/ \
       --volume $NN_DIR/t2t_usr:/t2t_usr \
       --volume $NN_DIR/scripts:/scripts \
       --env HOST_PERMS="$(id -u):$(id -g)" \
       $IMAGE $COMMAND $@
