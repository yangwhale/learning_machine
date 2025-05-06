#!/usr/bin/env bash

if groups "$USER" | grep -qw "docker"; then
    SUDO=""
else
    SUDO="sudo"
fi

$SUDO docker build --network=host -t spmd_demo .
$SUDO docker tag spmd_demo gcr.io/tpu-pytorch/spmd_demo:latest
$SUDO docker push gcr.io/tpu-pytorch/spmd_demo:latest
