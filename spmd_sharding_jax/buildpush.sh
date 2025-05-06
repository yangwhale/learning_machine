#!/usr/bin/env bash

if groups "$USER" | grep -qw "docker"; then
    SUDO=""
else
    SUDO="sudo"
fi

echo "build"
$SUDO docker build --network=host -t spmd_demo_lsiyuan .
echo "tag"
$SUDO docker tag spmd_demo_lsiyuan gcr.io/tpu-pytorch/spmd_demo_lsiyuan:latest
echo "upload"
$SUDO docker push gcr.io/tpu-pytorch/spmd_demo_lsiyuan:latest
