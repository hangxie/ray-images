# ray-images

Ray image for the homelab cluster, published as `hangxie/ray-images:<tag>`. It contains Ray with vLLM for LLM serving, plus a Qwen3-TTS voice-cloning Ray Serve app (see [TTS](#tts)).

The head, LLM workers, and TTS workers all run this one image, so the Ray version and the Serve app code always match across the cluster.

```text
Dockerfile                  the image
tts/app/                    Serve app (serve_tts.py, index.html), importable as serve_tts:app
tts/check_requirements.py   build-time dependency check for the TTS virtualenv
```

```bash
make build
docker inspect -f '{{json .Config.Labels}}' hangxie/ray-images:<tag>
```

Image tags are this repo's own `vX.Y.Z`, with `vX.Y.Z-rcN` for release candidates. Upstream versions are `ARG`s in the `Dockerfile`, recorded as `io.github.hangxie.ray-images.*` labels.

Images up to `v0.3.0` were published as `hangxie/ray-vllm` and use the `io.github.hangxie.ray-vllm.` label prefix.

## vLLM

Ray 2.59.0 is paired with vLLM 0.27.0, matching Ray Serve LLM's exact vLLM requirement.

The base is the slim `rayproject/ray:2.59.0-py312-cpu` image, with no CUDA toolkit. torch brings its CUDA runtime as pip wheels, and the NVIDIA container toolkit mounts the driver. `gcc` is installed because Triton compiles small launcher stubs at runtime.

The base's Brotli 1.1 is upgraded to 1.2: `huggingface_hub`'s HTTP client needs it, or `hf download` fails on brotli-encoded files such as a tokenizer's `vocab.json`.

Upgrading changes the compile and Triton cache keys, so expect a cold first start. Revalidate model-specific engine settings and tool-call parsers before rollout.

### FlashInfer

FlashInfer's CUDA 13.0 JIT-cache wheel is included, so the default sampler loads its kernel without `nvcc`. The build checks that the sampler binary is present for `sm_86`.

- The optional `flashinfer-cubin` bundle is omitted to save space. Operations that need it download cubins on demand and need network access.
- Keep `FLASHINFER_VERSION` aligned with `flashinfer-python`, and update the cache's CUDA suffix when torch's CUDA runtime changes.
- Configurations outside the prebuilt set can still need JIT compilation and a CUDA toolkit.

Smoke test on a GPU host, with the nvcc fallback disabled:

```bash
docker run --rm --gpus all -e FLASHINFER_DISABLE_JIT=1 hangxie/ray-images:<tag> \
  python3 -c "import torch; from flashinfer.sampling import top_k_top_p_sampling_from_probs; p = torch.full((4, 128), 1 / 128, device='cuda'); s = top_k_top_p_sampling_from_probs(p, 16, 0.9); torch.cuda.synchronize(); assert s.shape == (4,) and ((s >= 0) & (s < 128)).all().item(); print('FlashInfer sampler OK')"
```

The driver (`libcuda.so.1`) must be mounted by the NVIDIA container runtime. A CPU-only build can check that the cache exists, but cannot run its kernels.

### Loading weights from S3

vLLM is installed with its `runai` extra, so the Run:ai Model Streamer and its S3 backend are in the image. Point vLLM at a bucket with `load_format="runai_streamer"`:

```bash
AWS_ENDPOINT_URL=http://<rgw-host> \
AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... \
RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING=0 \
python3 -c "from vllm import LLM; LLM(model='s3://<bucket>/<model>', load_format='runai_streamer')"
```

- `AWS_ENDPOINT_URL` sets the endpoint for non-AWS stores (Ceph RGW, MinIO, R2). Leave it unset for AWS S3.
- `RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING=0` switches to path-style URLs (`http://host/bucket/key`), which most self-hosted stores need.
- Credentials come from the usual `AWS_*` variables. Some stores also need `AWS_REGION`.

## TTS

`serve_tts:app` runs Qwen3-TTS Base voice cloning as a Ray Serve application on CUDA. It ports site-up's `ai/mlx-voice-svc` (MLX, Apple Silicon only) to the official PyTorch package [`qwen-tts`](https://pypi.org/project/qwen-tts/).

### Python environments

`qwen-tts==0.1.1` pins `transformers==4.57.3`, but vLLM 0.27.0 needs `transformers>=5.5.3`. TTS therefore gets its own virtualenv at `/home/ray/tts-venv`.

- The venv is created with `--system-site-packages`. It reuses the image's Ray, torch, torchaudio, and CUDA libraries, and adds only transformers 4.57.3 and qwen-tts's other dependencies (about 0.4 GB).
- TTS replicas start with the venv's interpreter, through Ray's `runtime_env` `py_executable`.
- Everything else, including the head that builds the app, uses the system Python.

`pip check` passes for the system environment. Inside the venv it always reports vLLM's transformers requirement, so the build runs `tts/check_requirements.py` there instead. It checks every qwen-tts requirement as the venv resolves it.

`gradio` is only used by the qwen-tts demo, but it stays installed because qwen-tts declares it.

### Endpoints

`GET /` (page), `GET /health`, `GET /voices`, and `POST /api/tts` keep the MLX service's fields, ranges, error codes, and headers, so its `index.html` and `txt2mp3.sh` work unchanged. Point `TTS_URL` at the new endpoint.

- `X-Peak-CUDA-GB` replaces `X-Peak-MLX-GB`, and `/health` reports the CUDA device.
- The page uses relative URLs. Behind a path prefix, open it with a trailing slash (`/tts/`).
- The page retries a 503 the way it retried the old 409.

### Settings

| Setting | Default | Effect |
|---|---|---|
| `TTS_MODEL` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Model ID reported by `/health`; loaded from the Hub if `TTS_MODEL_DIR` is unset |
| `TTS_MODEL_DIR` | unset | Local copy of the model (with `HF_HUB_OFFLINE=1`) |
| `TTS_VOICES_DIR` | `/home/ray/tts/voices` | WAV + `.txt` transcript pairs, re-read per request |
| `TTS_NUM_REPLICAS` | `2` | Replicas, one GPU each |
| `TTS_MAX_QUEUED_REQUESTS` | `4` | Requests Serve queues before answering 503 |
| `TTS_PYTHON` | `/home/ray/tts-venv/bin/python` | Interpreter for TTS replicas (`py_executable`) |
| `TTS_CUDA_GRAPHS` | `1` | `0` turns off CUDA-graph decoding (see [Decoding speed](#decoding-speed)) |
| `TTS_KV_CACHE_TOKENS` | `3072` | Talker KV cache size for CUDA-graph decoding, in codec tokens including the prompt |

Set these in the application's `runtime_env` `env_vars`. They are read where Serve builds the app (the head) and passed to the replicas as `TTS.bind()` arguments.

A deployment's `runtime_env` replaces the application's instead of merging with it, so the replica process does not see the application's `env_vars`. The TTS deployment's `runtime_env` therefore carries the head's `HF_*` variables (for example `HF_HUB_OFFLINE`) along with `py_executable`.

### Deployment

Each replica asks for `num_gpus: 1` plus a custom `tts: 1` resource and handles one request at a time. Serve queues further requests and routes each to an idle replica, replacing the MLX service's lock and its 409.

- `/health`, `/voices`, and `/` are requests too, so they wait in the same queue while every replica is synthesizing.
- The model loads in bf16 on `cuda:0`. Startup fails unless it is a Base model whose speech tokenizer has an encoder.
- There is no client-IP allowlist; access control belongs to the Gateway.

### Decoding speed

Each audio frame (80 ms) takes one talker step plus 15 code-predictor steps, all small batch-1 forward passes. In eager HF `generate`, Python and kernel-launch overhead dominate: on an RTX 3050 the replica held one CPU core at 100% with the GPU about 22% busy, for an RTF around 4.

With `TTS_CUDA_GRAPHS=1` (the default), `tts/app/cuda_graphs.py` replays both as CUDA graphs:

- The code predictor's whole per-frame decode (prefill, 14 steps, and sampling) is one graph replay.
- The talker's one-token step is one graph over a static KV cache of `TTS_KV_CACHE_TOKENS` slots. HF `generate` still drives the talker loop, so the end-of-speech and disconnect hooks run once per frame outside the graph.

The replica captures both graphs at startup with a short throwaway synthesis and logs how long it took.

The KV cache costs about 115 KB per token (0.35 GB at 3072). A request's `max_tokens` is capped so the prompt plus generated tokens fit, and `X-Token-Limited` reports reaching that cap. Prompts that don't fit at all fall back to eager decoding.

Code-predictor sampling inside the graph uses Gumbel-max instead of `torch.multinomial`: an exact categorical draw, still seeded per request, but not bit-identical to the eager path's output for the same seed.

### Differences from the MLX service

- qwen-tts does not raise `repetition_penalty` in ICL mode, so the 1.5 floor is gone (range 1.0–3.0).
- Sampling defaults come from the model's `generation_config.json`; repetition penalty is 1.05 for the released checkpoints.
- `temperature: 0` means greedy decoding for the first codebook. The other codebooks keep the model's `subtalker_*` defaults.
- Seeds use `torch.manual_seed` and `torch.cuda.manual_seed_all`, so a seed does not reproduce MLX output.

### Generation hooks

Two hooks wrap the talker's `generate` (the first codebook) at load time. `Qwen3TTSForConditionalGeneration.generate` drops unknown kwargs, so they cannot be passed in; recheck them when upgrading qwen-tts.

- **Greedy end-of-speech**: when end-of-speech is the most likely token, every other token is masked. A missed end-of-speech draw never recovers and runs to the token limit.
- **Client disconnect**: when Serve cancels a request because its client disconnected, generation stops at the next codec frame and decoding is skipped. The handler waits for this, so the replica is idle before Serve hands it the next queued request.
