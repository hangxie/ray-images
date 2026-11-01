# Upstream versions: Ray pins the vLLM version it supports, so bump these together
ARG RAY_VERSION=2.55.1
ARG CUDA_TAG=cu124
ARG VLLM_VERSION=0.19.1

# Use the slim Ray image (no "-extra") as the foundation
FROM rayproject/ray:${RAY_VERSION}-py312-${CUDA_TAG}

ARG RAY_VERSION
ARG CUDA_TAG
ARG VLLM_VERSION
ARG VERSION=dev

LABEL org.opencontainers.image.source="https://github.com/hangxie/ray-vllm" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.base.name="rayproject/ray:${RAY_VERSION}-py312-${CUDA_TAG}" \
      io.github.hangxie.ray-vllm.ray-version="${RAY_VERSION}" \
      io.github.hangxie.ray-vllm.cuda-tag="${CUDA_TAG}" \
      io.github.hangxie.ray-vllm.vllm-version="${VLLM_VERSION}"

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends git-lfs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* /tmp/* /var/tmp/*

USER ray
# Install vLLM (with the Run:ai Model Streamer for s3:// weights) + Scipy and verify in a single layer;
# clean pip/tmp caches afterwards
RUN pip install --upgrade pip \
    && pip install "vllm[runai]==${VLLM_VERSION}" "scipy>=1.14.0" \
    && python3 -c "import vllm, runai_model_streamer, runai_model_streamer_s3; print('vLLM version:', vllm.__version__)" \
    && rm -rf /home/ray/.cache /tmp/* /var/tmp/*
