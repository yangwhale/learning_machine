#!/usr/bin/env bash

if groups "$USER" | grep -qw "docker"; then
    SUDO=""
else
    SUDO="sudo"
fi

$SUDO docker build --network=host -t llama3 .
$SUDO docker tag llama3 gcr.io/tpu-pytorch/llama3:latest
$SUDO docker push gcr.io/tpu-pytorch/llama3:latest
