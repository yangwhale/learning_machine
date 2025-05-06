# syntax=docker/dockerfile:experimental
# Use Python 3.10 as the base image
FROM python:3.10-slim-bullseye

# Install system dependencies
RUN apt-get update && apt-get upgrade -y
RUN apt-get update && apt-get install -y curl gnupg

# Add the Google Cloud SDK package repository
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

# Add the Cloud Storage FUSE distribution URL as a package source
RUN echo "deb https://packages.cloud.google.com/apt gcsfuse-bullseye main" | tee /etc/apt/sources.list.d/gcsfuse.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

# Install the Google Cloud SDK and GCS fuse
RUN apt-get update && apt-get install -y google-cloud-sdk git fuse gcsfuse && gcsfuse -v

# Set the default Python version to 3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.10 1

RUN pip install fire
RUN pip install -U --pre jax jaxlib libtpu requests -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
RUN pip install flax optax

RUN mkdir /workdir
WORKDIR /workdir
COPY ./*.py ./
COPY ./entrypoint.sh ./
# ENV LIBTPU_INIT_ARGS "--xla_tpu_scoped_vmem_limit_kib=98304 --xla_enable_async_all_gather=true --xla_tpu_overlap_compute_collective_tc=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true"
# ENV LIBTPU_INIT_ARGS "--xla_tpu_enable_flash_attention=false --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true --xla_tpu_scoped_vmem_limit_kib=81920"
# ENV LIBTPU_INIT_ARGS "--xla_tpu_enable_flash_attention=false  --xla_tpu_enable_async_collective_fusion=true  --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true  --xla_tpu_enable_async_collective_fusion_multiple_steps=true  --xla_tpu_overlap_compute_collective_tc=true  --xla_enable_async_all_gather=true  --xla_tpu_scoped_vmem_limit_kib=98304  --xla_latency_hiding_scheduler_rerun=1  --xla_tpu_prefer_async_allgather_to_allreduce=true"
ENTRYPOINT ["bash", "entrypoint.sh"]



