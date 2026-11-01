# Use the slim Ray image (no "-extra") as the foundation
FROM rayproject/ray:2.55.1-py312-cu124

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
    && pip install "vllm[runai]==0.19.1" "scipy>=1.14.0" \
    && python3 -c "import vllm, runai_model_streamer, runai_model_streamer_s3; print('vLLM version:', vllm.__version__)" \
    && rm -rf /home/ray/.cache /tmp/* /var/tmp/*
