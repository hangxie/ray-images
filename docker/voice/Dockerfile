# hangxie/ray-voice: Ray with the TTS, STT, voice chat and meeting notes Serve apps. Built from the repository root
# (`make build-voice`). No vLLM, so qwen-tts and its transformers 4.57.3 install straight into the system Python.
ARG RAY_VERSION=2.59.0

# Use the slim CPU Ray image (no "-extra", no CUDA toolkit) as the foundation. torch ships its own
# CUDA runtime as pip wheels and the driver (libcuda) is mounted by the NVIDIA container toolkit.
FROM rayproject/ray:${RAY_VERSION}-py312-cpu

ARG RAY_VERSION
# torch and torchaudio as vLLM 0.27.0 pinned them when TTS and STT shared its image (validated on the RTX 3050s, CUDA 13)
ARG TORCH_VERSION=2.13.0
ARG TORCHAUDIO_VERSION=2.11.0
ARG QWEN_TTS_VERSION=0.1.1
ARG SILERO_VAD_VERSION=6.3

LABEL org.opencontainers.image.source="https://github.com/hangxie/ray-images" \
      org.opencontainers.image.base.name="rayproject/ray:${RAY_VERSION}-py312-cpu" \
      io.github.hangxie.ray-images.ray-version="${RAY_VERSION}" \
      io.github.hangxie.ray-images.torch-version="${TORCH_VERSION}" \
      io.github.hangxie.ray-images.qwen-tts-version="${QWEN_TTS_VERSION}" \
      io.github.hangxie.ray-images.silero-vad-version="${SILERO_VAD_VERSION}"

# The CPU base doesn't set these; without "compute" the container toolkit may not mount libcuda
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

USER root
# gcc + libc6-dev: Triton (torch's) compiles its launcher stubs with the host C compiler if anything JIT-compiles
RUN apt-get update \
    && apt-get install -y --no-install-recommends git-lfs gcc libc6-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* /tmp/* /var/tmp/*

USER ray
# torch and its CUDA libraries in a layer of their own (most of the image), so a qwen-tts change doesn't re-upload them
RUN pip install --upgrade pip \
    && pip install "torch==${TORCH_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}" \
    && pip check \
    && python3 -c "import torch; assert torch.version.cuda.startswith('13.'), torch.version.cuda; print('torch', torch.__version__, 'CUDA', torch.version.cuda)" \
    && rm -rf /home/ray/.cache /tmp/* /var/tmp/*

# qwen-tts pins transformers==4.57.3, which also has Whisper for STT; Silero VAD (torch only, model in the wheel) tells
# speech from music and noise for STT's live stream. torch/torchaudio are constrained so nothing replaces them.
# The base ships Brotli 1.1, but huggingface_hub's HTTP client passes output_buffer_limit to brotli's Decompressor,
# which needs 1.2: without it `hf download` (the clusters' model-pull init container) fails on brotli-encoded files.
RUN pip install "torch==${TORCH_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}" \
       "qwen-tts==${QWEN_TTS_VERSION}" "silero-vad==${SILERO_VAD_VERSION}" "scipy>=1.14.0" soundfile "brotli>=1.2" \
    && pip check \
    && rm -rf /home/ray/.cache /tmp/* /var/tmp/*

# The Serve apps, one directory each under /home/ray/serve, and common/ (code they share), all on PYTHONPATH so they
# stay importable as serve_tts:app, serve_stt:app, serve_chat:app and serve_meeting:app.
# pyarrow (from the base image, unused here) is blocked for the checks because its native library segfaults under
# amd64 emulation on Apple Silicon; ray.serve and qwen_tts import fine without it.
COPY --chown=ray:users serve/common/ /home/ray/serve/common/
COPY --chown=ray:users serve/tts/ /home/ray/serve/tts/
COPY --chown=ray:users serve/stt/ /home/ray/serve/stt/
COPY --chown=ray:users serve/chat/ /home/ray/serve/chat/
COPY --chown=ray:users serve/meeting/ /home/ray/serve/meeting/
ENV PYTHONPATH=/home/ray/serve/common:/home/ray/serve/tts:/home/ray/serve/stt:/home/ray/serve/chat:/home/ray/serve/meeting${PYTHONPATH:+:${PYTHONPATH}}
RUN python3 -c "import sys; sys.modules['pyarrow'] = None; import qwen_tts, ray, soundfile, torch, transformers, silero_vad, serve_tts, serve_stt, serve_chat, serve_meeting; silero_vad.load_silero_vad(); assert {'MP3', 'OGG', 'FLAC'} <= soundfile.available_formats().keys() and 'OPUS' in soundfile.available_subtypes('OGG'), soundfile.__libsndfile_version__; assert transformers.__version__ == '4.57.3', transformers.__version__; print('Serve apps:', serve_tts.app, serve_stt.app, serve_chat.app, serve_meeting.app); print('Ray', ray.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)"

# Last, so a new version only changes image metadata and keeps the build cache
ARG VERSION=dev
# A file, not ENV: Podman's cache reuses an ENV layer when only VERSION changes, but reruns a RUN that uses it.
# One file beside the app directories, read by every app.
RUN printf '%s\n' "${VERSION}" > /home/ray/serve/VERSION
LABEL org.opencontainers.image.version="${VERSION}"
