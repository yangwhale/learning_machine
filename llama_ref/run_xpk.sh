#!/usr/bin/env bash

set -eo

##### Runs SPMD training as an xpk job on a GKE cluster #####
#
# To run this, a _source_ install of xpk is required to access the latest TPU.
#
# Example: pip install git+https://github.com/AI-Hypercomputer/xpk.git@main
#

# Always build a new image. This is fast when cached.
./buildpush.sh

# You can override these by setting corresponding environment variables.
#: "${CLUSTER_NAME:=bodaborg-v6e-256}"
: "${CLUSTER_NAME:=bodaborg-v6e-256-donotdelete}"
# "${CLUSTER_NAME:=abhinavsing-in-mem}"
: "${DOCKER_URL:=gcr.io/tpu-pytorch/llama3:latest}"
: "${NUM_SLICES:=2}"
: "${TPU_TYPE:=v6e-256}"
: "${ZONE:=us-east5-c}"
#: "${PROJECT_ID:=tpu-prod-env-automated}"
: "${PROJECT_ID:=tpu-prod-env-one-vm}"

DATETIMESTR=$(date +%Y%m%d-%H%M%S)
COMMAND="python run_xpk.py --batch_size=512 --model_type=405B --seqlen=8192 --use_custom_offload=True --use_custom_mesh=False --model_impl=scan --tp=1 --unroll_layers=2"

xpk workload create \
    --cluster ${CLUSTER_NAME} \
    --docker-image ${DOCKER_URL} \
    --workload "${USER}-xpk-v6e-256-$NUM_SLICES-${DATETIMESTR}" \
    --tpu-type=${TPU_TYPE} \
    --num-slices=${NUM_SLICES} \
    --zone $ZONE \
    --project $PROJECT_ID \
    --enable-debug-logs \
    --command "$COMMAND"
    #$--on-demand \
