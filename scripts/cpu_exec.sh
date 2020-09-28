#!/usr/bin/env bash

T2T_VERSION=1.14.1
IMAGE="t2t-${T2T_VERSION}"
BASENAME=$(basename $(dirname $0))

# base directory where the run files will be stored
NN_DIR=

COMMAND=$1
COMMAND_NAME=$(basename $1)
shift

docker run \
       --name "$IMAGE-exec-$COMMAND_NAME" \
       --rm \
       --interactive \
       --tty \
       --shm-size=1g \
       --ulimit memlock=-1 \
       --volume $NN_DIR/t2t_train:/t2t_train \
       --volume $NN_DIR/t2t_data:/t2t_data \
       --volume $NN_DIR/models:/models \
       --volume $NN_DIR/data:/data \
       --volume $NN_DIR/t2t_usr:/t2t_usr \
       --volume $NN_DIR/scripts:/scripts \
       --env HOST_PERMS="$(id -u):$(id -g)" \
       $IMAGE $COMMAND $@
