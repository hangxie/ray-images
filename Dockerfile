# Upstream versions: Ray pins the vLLM version it supports, so bump these together
ARG RAY_VERSION=2.55.1
ARG VLLM_VERSION=0.19.1

# Use the slim CPU Ray image (no "-extra", no CUDA toolkit) as the foundation. torch ships its own
# CUDA runtime as pip wheels and the driver (libcuda) is mounted by the NVIDIA container toolkit.
FROM rayproject/ray:${RAY_VERSION}-py312-cpu

ARG RAY_VERSION
ARG VLLM_VERSION
ARG VERSION=dev

LABEL org.opencontainers.image.source="https://github.com/hangxie/ray-vllm" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.base.name="rayproject/ray:${RAY_VERSION}-py312-cpu" \
      io.github.hangxie.ray-vllm.ray-version="${RAY_VERSION}" \
      io.github.hangxie.ray-vllm.vllm-version="${VLLM_VERSION}"

# The CPU base doesn't set these; without "compute" the container toolkit may not mount libcuda
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

USER root
# gcc + libc6-dev: Triton compiles its launcher stubs with the host C compiler at runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends git-lfs gcc libc6-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* /tmp/* /var/tmp/*

USER ray
# Install vLLM (with the Run:ai Model Streamer for s3:// weights) + Scipy and verify in a single layer;
# check that gcc can build against the headers Triton's stubs include; clean pip/tmp caches afterwards
RUN pip install --upgrade pip \
    && pip install "vllm[runai]==${VLLM_VERSION}" "scipy>=1.14.0" \
    && python3 -c "import vllm, runai_model_streamer, runai_model_streamer_s3; print('vLLM version:', vllm.__version__)" \
    && printf '#include <Python.h>\n#include <dlfcn.h>\n#include <stdio.h>\n#include "cuda.h"\n' \
       | gcc -fsyntax-only -x c - \
         -I"$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["include"])')" \
         -I"$(python3 -c 'import os, triton; print(os.path.join(os.path.dirname(triton.__file__), "backends/nvidia/include"))')" \
    && rm -rf /home/ray/.cache /tmp/* /var/tmp/*
