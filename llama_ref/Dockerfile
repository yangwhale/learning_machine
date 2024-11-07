# syntax=docker/dockerfile:experimental
# Use Python 3.10 as the base image
FROM python:3.10-slim-bullseye

# Install system dependencies
RUN apt-get update && apt-get upgrade -y
RUN apt-get update && apt-get install -y curl gnupg

# Add the Google Cloud SDK package repository
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

# Install the Google Cloud SDK
RUN apt-get update && apt-get install -y google-cloud-sdk git

# Set the default Python version to 3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.10 1


RUN git clone https://github.com/pytorch/xla.git
WORKDIR xla/experimental/torch_xla2
RUN pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
RUN pip install optax fire
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install -e .[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html

RUN mkdir /llama
WORKDIR /llama
COPY ./*.py ./
ENV LIBTPU_INIT_ARGS "--xla_tpu_scoped_vmem_limit_kib=98304 --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_tpu_decompose_all_gather_einsum=true --xla_tpu_decompose_einsum_reduce_scatter=true --xla_tpu_use_enhanced_launch_barrier=true --xla_tpu_spmd_rng_bit_generator_unsafe=true"
ENTRYPOINT ["python", "run.py"]
CMD ["--lr=0.001", "--model_type=8B"]



