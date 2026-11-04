# Upstream versions: Ray pins the vLLM version it supports, so bump these together
ARG RAY_VERSION=2.59.0
ARG VLLM_VERSION=0.27.0

# Use the slim CPU Ray image (no "-extra", no CUDA toolkit) as the foundation. torch ships its own
# CUDA runtime as pip wheels and the driver (libcuda) is mounted by the NVIDIA container toolkit.
FROM rayproject/ray:${RAY_VERSION}-py312-cpu

ARG RAY_VERSION
ARG VLLM_VERSION

LABEL org.opencontainers.image.source="https://github.com/hangxie/ray-images" \
      org.opencontainers.image.base.name="rayproject/ray:${RAY_VERSION}-py312-cpu" \
      io.github.hangxie.ray-images.ray-version="${RAY_VERSION}" \
      io.github.hangxie.ray-images.vllm-version="${VLLM_VERSION}"

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
# resolve Ray's Prometheus exporter too, so vLLM keeps its OpenTelemetry dependencies compatible.
# Check package consistency and Triton's compiler headers; clean pip/tmp caches afterwards.
RUN pip install --upgrade pip \
    && pip install "vllm[runai]==${VLLM_VERSION}" "scipy>=1.14.0" opentelemetry-exporter-prometheus \
    && pip check \
    && python3 -c "import vllm, runai_model_streamer, runai_model_streamer_s3; print('vLLM version:', vllm.__version__)" \
    && printf '#include <Python.h>\n#include <dlfcn.h>\n#include <stdio.h>\n#include "cuda.h"\n' \
       | gcc -fsyntax-only -x c - \
         -I"$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["include"])')" \
         -I"$(python3 -c 'import os, triton; print(os.path.join(os.path.dirname(triton.__file__), "backends/nvidia/include"))')" \
    && rm -rf /home/ray/.cache /tmp/* /var/tmp/*

# FlashInfer otherwise JIT-compiles the sampler and needs nvcc, which the CPU base lacks.
# Keep the JIT cache on the same release as flashinfer-python and match torch's CUDA runtime.
# Omit the large optional cubin bundle; other cubins can be downloaded on demand.
# Drop cached modules none of FLASHINFER_KEEP_ARCHS can run (about 0.9 GB for 8.6); add an arch or use "all" before
# deploying to other GPUs. Pruned in this layer, since a later one could not shrink it.
ARG FLASHINFER_VERSION=0.6.16.post3
ARG FLASHINFER_KEEP_ARCHS=8.6
COPY --chown=ray:users scripts/prune_flashinfer_cache.py /tmp/prune_flashinfer_cache.py
RUN pip install "flashinfer-python==${FLASHINFER_VERSION}" \
    && pip install "flashinfer-jit-cache==${FLASHINFER_VERSION}+cu130" --index-url https://flashinfer.ai/whl/cu130 \
    && pip check \
    && python3 /tmp/prune_flashinfer_cache.py ${FLASHINFER_KEEP_ARCHS} \
    && FLASHINFER_CUDA_ARCH_LIST=8.6 FLASHINFER_DISABLE_JIT=1 python3 -c "from flashinfer.jit.sampling import gen_sampling_module; spec = gen_sampling_module(); assert spec.is_aot, 'Prebuilt FlashInfer sampler is missing'; print('Prebuilt FlashInfer sampler:', spec.aot_path)" \
    && rm -rf /home/ray/.cache /tmp/* /var/tmp/*

# The base ships Brotli 1.1, but huggingface_hub's HTTP client passes output_buffer_limit to brotli's Decompressor,
# which needs 1.2: without it `hf download` fails on brotli-encoded files such as a tokenizer's vocab.json.
RUN pip install "brotli>=1.2" \
    && pip check \
    && rm -rf /home/ray/.cache /tmp/* /var/tmp/*

# Qwen3-TTS runs in its own virtualenv because qwen-tts pins transformers==4.57.3, which vLLM rules out. The venv
# inherits the system packages (Ray, torch, CUDA libraries) and adds only what qwen-tts needs on top; torch/torchaudio
# are constrained to the system versions so no second copy is installed. TTS replicas run its interpreter through
# runtime_env py_executable (tts/app/serve_tts.py). Kept after the vLLM layers so a TTS change rebuilds only this.
ARG QWEN_TTS_VERSION=0.1.1
LABEL io.github.hangxie.ray-images.qwen-tts-version="${QWEN_TTS_VERSION}"
COPY --chown=ray:users tts/check_requirements.py /tmp/check_requirements.py
RUN python3 -m venv --system-site-packages /home/ray/tts-venv \
    && pip list --format=freeze | grep -E '^(torch|torchaudio)==' > /tmp/constraints.txt \
    && /home/ray/tts-venv/bin/pip install --constraint /tmp/constraints.txt \
       "qwen-tts==${QWEN_TTS_VERSION}" "scipy>=1.14.0" soundfile \
    && /home/ray/tts-venv/bin/python /tmp/check_requirements.py "qwen-tts==${QWEN_TTS_VERSION}" scipy soundfile \
    && pip check \
    && rm -rf /home/ray/.cache /tmp/* /var/tmp/*

# The Serve app, importable as serve_tts:app: the head builds it with the system Python, replicas run it in the venv.
# pyarrow (from the base image, unused here) is blocked for the checks because its native library segfaults under
# amd64 emulation on Apple Silicon; ray.serve and qwen_tts import fine without it.
COPY --chown=ray:users tts/app/ /home/ray/tts/
ENV PYTHONPATH=/home/ray/tts${PYTHONPATH:+:${PYTHONPATH}}
RUN python3 -c "import sys; sys.modules['pyarrow'] = None; import serve_tts, transformers; print('Serve app:', serve_tts.app, 'system transformers', transformers.__version__)" \
    && /home/ray/tts-venv/bin/python -c "import sys; sys.modules['pyarrow'] = None; import qwen_tts, ray, torch, transformers, serve_tts; assert transformers.__version__ == '4.57.3', transformers.__version__; print('TTS venv: Ray', ray.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)"

# Last, so a new version only changes image metadata and keeps the build cache
ARG VERSION=dev
LABEL org.opencontainers.image.version="${VERSION}"
