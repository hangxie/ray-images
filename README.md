# ray-images

Ray images for the homelab cluster, built from this repo and tagged together:

- `hangxie/ray-vllm:<tag>`: Ray with vLLM for LLM serving (see [vLLM](#vllm)).
- `hangxie/ray-voice:<tag>`: Ray with the Serve apps: Qwen3-TTS voice cloning (see [TTS](#tts)), Whisper speech-to-text (see [STT](#stt)), a voice chat app that drives them and an LLM (see [Voice chat](#voice-chat)), and meeting notes that transcribe a meeting live through STT (see [Meeting notes](#meeting-notes)).

Each app runs in its own Ray cluster, whose head and workers all use one image, and the apps talk to each other over HTTP. So the LLM and the voice apps don't need to share an image, and they don't: `qwen-tts==0.1.1` pins `transformers==4.57.3`, while vLLM 0.27.0 needs `transformers>=5.5.3`. Apart, each image has one Python environment where `pip check` passes, the voice image carries no vLLM or FlashInfer, and a vLLM upgrade doesn't rebuild or re-validate TTS and STT.

```text
docker/vllm/Dockerfile      hangxie/ray-vllm
docker/voice/Dockerfile     hangxie/ray-voice
scripts/                    build-time helpers (FlashInfer cache pruning); txt2mp3, txt2m4b (see TTS)
serve/tts/                  TTS Serve app (serve_tts.py, cuda_graphs.py, index.html), importable as serve_tts:app
serve/stt/                  STT Serve app (serve_stt.py, stt.html), importable as serve_stt:app
serve/chat/                 voice chat Serve app (serve_chat.py, chat.html), importable as serve_chat:app
serve/meeting/              meeting notes Serve app (serve_meeting.py, meeting.html), importable as serve_meeting:app
serve/common/               code the Serve apps share (caller_log.py)
tests/                      CPU test suite (pytest)
pyproject.toml, uv.lock     development environment (uv), ruff, mypy and pytest settings
```

```bash
make dev                # development environment in .venv (needs uv)
make check              # ruff format/lint, mypy, Markdown rules, tests
make build              # both images; make build-vllm or make build-voice for one
docker inspect -f '{{json .Config.Labels}}' hangxie/ray-voice:<tag>
```

Image tags are this repo's own `vX.Y.Z`, with `vX.Y.Z-rcN` for release candidates; one git tag releases both images, and the release notes say which one changed. Both Dockerfiles build from the repository root. Upstream versions are `ARG`s in each Dockerfile, recorded as `io.github.hangxie.ray-images.*` labels.

Until `v0.11.0`, one image held both, published as `hangxie/ray-images` (and up to `v0.3.0` as `hangxie/ray-vllm`, with the `io.github.hangxie.ray-vllm.` label prefix).

## Development

`pyproject.toml` and `uv.lock` define a development environment for linting and tests only; the images are still built with pip from their Dockerfiles. The environment mirrors the voice image (Ray, torch, torchaudio, qwen-tts with transformers 4.57.3), so keep its pins in step with `docker/voice/Dockerfile` when upgrading; `tests/test_pins.py` fails when they drift.

Most of the code needs a GPU, so the tests run what they can on CPU: the HTTP APIs and request validation against fake models, audio post-processing, and, with tiny random qwen-tts models, the generation hooks, CUDA-graph decoding bodies (eager on CPU) against HF `generate`, and streamed decoding against the full-clip decode. Model loading, CUDA graph capture, and real synthesis are still validated on the cluster. CI runs `make check` on every push and pull request.

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
- Cached modules that none of `FLASHINFER_KEEP_ARCHS` (default `8.6`) can run are deleted at build time by `scripts/prune_flashinfer_cache.py`, saving about 0.9 GB (0.5 GB compressed). It reads each module's embedded GPU code and drops whole modules with nothing for those architectures: on those GPUs they would only fail. Before deploying to other GPUs, add their architecture (`--build-arg FLASHINFER_KEEP_ARCHS="8.6 8.9"`) or keep everything with `all`.

Smoke test on a GPU host, with the nvcc fallback disabled:

```bash
docker run --rm --gpus all -e FLASHINFER_DISABLE_JIT=1 hangxie/ray-vllm:<tag> \
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

### Python environment

TTS and STT run in the voice image's system Python. Its torch and torchaudio are the versions vLLM 0.27.0 pinned when the apps shared vLLM's image (2.13.0 and 2.11.0, CUDA 13), which is what the cluster validated; they are `ARG`s now and can move on their own. Everything is installed against those pins so no package replaces them, and `pip check` passes.

`gradio` is only used by the qwen-tts demo, but it stays installed because qwen-tts declares it.

### Endpoints

TTS speaks OpenAI's speech API, so OpenAI clients such as Open WebUI's read-aloud use it with their OpenAI TTS base URL set to the app's prefix plus `/v1` (e.g. `/tts/v1`), separately from STT's: `GET /` (page), `POST /v1/audio/speech`, `GET /v1/audio/voices`, `GET /v1/models`, `GET /v1/health`. Errors use OpenAI's `{"error": {"message", "type", "param", "code"}}` shape, 404s included. The MLX service's `/api/tts`, `/health` and `/voices` are gone ([#34](https://github.com/hangxie/ray-images/issues/34)).

- `POST /v1/audio/speech` takes `input`, `voice`, `response_format`, `speed` and `stream_format`, plus this app's own fields: `seed`, `language`, `volume` (LUFS) and the sampling options `/v1/health` lists. `model` and `instructions` are ignored. `voice` must be one of this app's voices (`/v1/audio/voices`, `{"voices": [{"id", "name"}]}` as other OpenAI-style TTS servers list them), not OpenAI's names, and `speed` is 0.5–2.0 rather than OpenAI's 0.25–4.0.
- `response_format` defaults to `mp3`, as OpenAI's does. `mp3`, `opus` (Ogg), `flac`, `wav` and `pcm` (raw 16-bit at `X-Sample-Rate`, 24 kHz) are encoded by libsndfile, and `aac` is refused since the image has no FFmpeg.
- Without `stream_format`, the answer is the whole clip with timing headers: `X-Synthesis-Seconds`, `X-Audio-Seconds`, `X-RTF`, `X-Token-Limited`, `X-Peak-CUDA-GB`, `X-Loudness-LUFS`, `X-Loudness-Limited`, `X-Sample-Rate`. A text too long for `max_tokens` is a 413. `stream_format` `audio` or `sse` streams it instead (see [Streaming](#streaming)).
- `/v1/health` reports the model, the CUDA device, the languages, and the ranges of `speed`, `volume` and the sampling options. It and the page title show the image tag from the `VERSION` file the build writes beside the app directories, as in [Voice chat](#voice-chat).
- The page uses relative URLs. Behind a path prefix, open it with a trailing slash (`/tts/`). It retries a 503.
- [txt2mp3 and txt2m4b](#txt2mp3-and-txt2m4b) narrate text files through it.
- Neither TTS nor STT authenticates: both serve plain HTTP (and STT's WebSocket) and ignore `Authorization`. The deployment provides login and API tokens in front of them (in the homelab, OAuth2 Proxy for the page and a bearer check at the Gateway for the APIs).
- Every app logs the caller of a request that carries `X-Consumer` (the Gateway's bearer check) or an email from OAuth2 Proxy (`X-Auth-Request-Email` as a Gateway auth check, else `X-Forwarded-Email` as a reverse proxy, as in front of the TTS page): one `ray.serve` line with the method, path, status (or `accepted`/`refused` for a WebSocket, `failed` for an exception), seconds, and those values. Requests without them appear only in Serve's access log. `Authorization` is never read or logged. The headers are trusted as they arrive, so the proxy in front must set or strip them. Voice chat and meeting notes don't forward the browser user to STT and TTS: their calls are logged under the identity their own token gets (in the homelab, `consumer=ray-vchat`), and the user under chat's or meeting's own line.

### Settings

| Setting | Default | Effect |
|---|---|---|
| `TTS_MODEL` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Model ID reported by `/v1/health` and `/v1/models`; loaded from the Hub if `TTS_MODEL_DIR` is unset |
| `TTS_MODEL_DIR` | unset | Local copy of the model (with `HF_HUB_OFFLINE=1`) |
| `TTS_VOICES_DIR` | `/home/ray/serve/tts/voices` | WAV + `.txt` transcript pairs; encoded at startup, re-encoded on the next request after either file changes |
| `TTS_NUM_REPLICAS` | `2` | Replicas, one GPU each |
| `TTS_MAX_QUEUED_REQUESTS` | `4` | Requests Serve queues before answering 503 |
| `TTS_CUDA_GRAPHS` | `1` | `0` turns off CUDA-graph decoding (see [Decoding speed](#decoding-speed)) |
| `TTS_KV_CACHE_TOKENS` | `3072` | Talker KV cache size for CUDA-graph decoding, in codec tokens including the prompt |
| `TTS_STREAM_CONTEXT_FRAMES` | `300` | Frames of reference plus speech the decoder transformer sees per streamed piece |

Set these in the application's `runtime_env` `env_vars`. They are read where Serve builds the app (the head) and passed to the replicas as `TTS.bind()` arguments.

### Deployment

Each replica asks for `num_gpus: 1` plus a custom `tts: 1` resource and handles one request at a time. Serve queues further requests and routes each to an idle replica, replacing the MLX service's lock and its 409.

- `/v1/health`, `/v1/audio/voices`, `/v1/models` and `/` are requests too, so they wait in the same queue while every replica is synthesizing.
- The model loads in bf16 on `cuda:0`. Startup fails unless it is a Base model whose speech tokenizer has an encoder.
- At startup, each voice's reference is encoded once into a voice-clone prompt (speech-tokenizer codes plus speaker embedding), logged as `Encoded N voice prompts in X s`. Requests reuse it instead of re-encoding the WAV. A voice that fails to encode is logged and skipped; it returns the same error per request as before.
- There is no client-IP allowlist; access control belongs to the Gateway.

### Decoding speed

Each audio frame (80 ms) takes one talker step plus 15 code-predictor steps, all small batch-1 forward passes. In eager HF `generate`, Python and kernel-launch overhead dominate: on an RTX 3050 the replica held one CPU core at 100% with the GPU about 22% busy, for an RTF around 4.

With `TTS_CUDA_GRAPHS=1` (the default), `serve/tts/cuda_graphs.py` replays both as CUDA graphs:

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

### Streaming

A request with `stream_format` gets its audio about every second of speech as it is generated, instead of after the whole clip. Voice chat asks for `pcm` over SSE and plays it piece by piece.

- `audio` streams the encoded bytes themselves (chunked, with `X-Sample-Rate`). `sse` sends `text/event-stream` events: a `speech.audio.delta` (base64 `audio`) per piece, then `speech.audio.done` with OpenAI's `usage` (text tokens in, codec frames out) plus this app's timings (`synthesis_seconds`, `first_audio_seconds`, `audio_seconds`, `rtf`, `token_limited`, `peak_cuda_gb`, `gain_db`, `clipped_samples`), or an `error` event.
- The reply starts with the first piece, so a failure before it is still an error status. Later, SSE sends an `error` event and raw audio ends unfinished.
- A stream can be `mp3`, `opus`, `wav` or `pcm`, not `flac`, whose header libsndfile fills in only on close, after a stream has sent it. A streamed WAV's header gives the maximum length, as for a WAV still being written. A streamed MP3 is constant bitrate (about 80 kb/s), since it can't get the Xing header that tells readers a variable-bitrate file's length. `speed` is rejected with streaming.

- A third hook wraps the talker's `forward` and hands each completed codec frame (the first code plus the code predictor's 15) to the request's `Streamer`. After the last frame the talker hook raises `StreamFinished`, so `generate_voice_clone`'s full-clip decode is skipped.
- Every 12 frames, `decode_tail` decodes the new frames with the voice's reference codes and the speech so far in front, up to `TTS_STREAM_CONTEXT_FRAMES`, the way the full-clip decode puts the reference first. Only the decoder transformer sees that whole window. The vocoder after it is causal with a receptive field of 5–8 frames, so it re-runs just 8 frames in front of the new ones. A CPU check with a random-weight decoder matched the full-clip decode to float rounding (about 350 dB SNR) whenever reference plus speech fits in 300 frames. Past that, qwen-tts's own decode drops to 25 frames of left context while streaming keeps 300.
- The clip's loudness is unknown until it ends, so the gain comes from the reference recording, whose level the clone follows, and samples past the peak ceiling are clipped (`clipped_samples`).

### txt2mp3 and txt2m4b

`scripts/txt2mp3.py` narrates a text file into one MP3, and `scripts/txt2m4b.py` a book into a chaptered m4b; their `--help` describes them. Each non-empty line is a paragraph, synthesized with `POST /v1/audio/speech` as a WAV, so they need Python 3.11+ and FFmpeg but nothing from this repository's environment. `scripts/tts_client.py`, which both use, spreads paragraphs over every endpoint in `~/.config/mlx-tts/endpoints.toml` (or `--endpoints`), or sends them to one `--tts-url`, with a bearer token for each; an endpoint URL is the app's base, with or without `/v1`. They came from site-up's `ai/mlx-tts-svc`, and the MLX service itself can no longer be an endpoint, since it only speaks `/api/tts`.

## STT

`serve_stt:app` serves Whisper (`openai/whisper-large-v3-turbo` by default) through transformers on CUDA with OpenAI's transcription API. It ports site-up's `ai/mlx-stt-svc`, so clients such as site-up's voice chat only change their STT URL. It shares the voice image with TTS, whose transformers 4.57.3, soundfile, and scipy already cover transcription; the live stream adds only `silero-vad` (a 7 MB wheel needing only torch).

The routes carry `/v1` themselves, as `mlx-stt-svc`'s do, so mount it at `route_prefix: /` on its own host (or at any prefix, such as `/stt`): `GET /` (page), `POST /v1/audio/transcriptions`, `WS /v1/audio/stream`, `GET /v1/models`, `GET /v1/health`. `/v1/health` also lists Whisper's languages (`languages`, English name or alias to code), which the [meeting notes](#meeting-notes) page offers.

- The page is deliberately plain, for trying STT out: Record, then Stop uploads the recording as 16 kHz WAV to `v1/audio/transcriptions` and shows the text, with the language detected. It keeps nothing between visits. A page change redeploys the GPU replica and reloads Whisper, so features belong in the [meeting notes](#meeting-notes) app instead. It uses relative URLs, so behind a prefix open it with the trailing slash (`/stt/`).

`audio/stream` takes 16 kHz mono 16-bit little-endian PCM as binary messages, then `{"type": "stop"}`, with an optional `language` query parameter. It answers with a `segment` event (`start`, `end` in seconds from the first sample, `text`, `language`, `paragraph`, `seconds` spent transcribing) for each piece with text, `error` for a piece that failed (the stream goes on), and `done` after the last piece. While a piece is still going, a `partial` event (`start`, `text`) previews its text about every second of audio, when the GPU is free and no finished piece is waiting; the piece's `segment` has the same `start`, and no partial for a piece comes after it. Previews re-transcribe the piece from its start, so a 15 s piece costs up to 15 extra transcriptions of growing length; on the cluster a 7 s piece took 0.5–0.7 s, so one meeting keeps the GPU about half busy, and finished pieces and HTTP requests go first.

- The stream is cut into pieces on the server, at short pauses, as tried on the page in v0.7.0-rc2: past 5 s the next 0.25 s pause ends a piece, and at 15 s the piece ends after its frame least like speech past 5 s. Speech is found by Silero VAD, a small model run on the CPU (about 0.2 ms per 32 ms frame, one model per stream since it keeps state), because loudness can't tell speech from music: the level-based detection of v0.8.0-rc2 to rc5 took background music with a beat for continuous speech, even 30 dB below the voice, so pauses and paragraphs vanished and music-only stretches were transcribed. A frame is speech at a probability of 0.5 and quiet below 0.35 (Silero's own hysteresis), and runs of speech under 0.25 s are ignored, as in Silero's own segmenting, so a beat or a clatter doesn't count. A piece is trimmed to its speech plus 0.3 s, and one with less than 0.3 s of speech is dropped, since Whisper makes up text from silence and music. Each stream logs its audio length, share of speech, pieces, previews, and peak level when it ends.
- Each piece gets the stream's text so far as Whisper's prompt (its last 160 tokens), so words cut at a piece boundary and names spelled earlier come out right. The language found in the first piece is kept for the rest of the stream rather than detected again on every short piece.

- The multipart form takes `file`, plus optional `language` (code or English name; detected from the first 30 s when omitted), `response_format` (`json`, `verbose_json`, `text`), and `model` (ignored). Replies carry `text`, `language` as a lowercase English name, and `duration`, plus `X-Transcribe-Seconds`, `X-Audio-Seconds`, `X-RTF`. Errors use OpenAI's `{"error": {"message": ...}}`.
- Uploads are limited to 25 MB and 300 s. Audio is decoded by libsndfile (WAV, FLAC, OGG, MP3; no FFmpeg, so no M4A or WebM), mixed to mono, and resampled to 16 kHz. Audio past 30 s is decoded long-form with timestamps.
- The multipart body is parsed with the standard library, so `python-multipart` is not needed.
- One transcription runs at a time per replica, behind a lock in the replica, since a stream holds its Serve request for as long as it lasts. A replica takes `STT_MAX_ONGOING_REQUESTS` (default 16) requests and streams at once, Serve queues up to `STT_MAX_QUEUED_REQUESTS` (default 8) more, and answers 503 past that.
- A replica asks for `num_gpus: 1` only. On 6 GB cards a TTS replica (5.37 GB peak) leaves no room for Whisper, so run one TTS replica and one STT replica on separate cards.

| Setting | Default | Effect |
|---|---|---|
| `STT_MODEL` | `openai/whisper-large-v3-turbo` | Model ID reported by `/v1/health`; loaded from the Hub if `STT_MODEL_DIR` is unset |
| `STT_MODEL_DIR` | unset | Local copy of the model (with `HF_HUB_OFFLINE=1`) |
| `STT_NUM_REPLICAS` | `1` | Replicas, one GPU each |
| `STT_MAX_ONGOING_REQUESTS` | `16` | Requests and streams a replica takes at once; transcription itself runs one at a time |
| `STT_MAX_QUEUED_REQUESTS` | `8` | Requests Serve queues before answering 503 |

## Voice chat

`serve_chat:app` is site-up's voice chat ([`media/voice-chat`](https://github.com/hangxie/site-up/tree/main/media/voice-chat)) as a Serve app. The page records speech; the replica sends it to STT for a transcript, streams a reply from an OpenAI-compatible LLM, and has TTS speak it one sentence at a time, so the first sentence plays while the rest is still being generated. The turn pipeline, the NDJSON events, and the page are the site-up service's; its README describes how a turn works. A WebSocket transport is tracked in [#14](https://github.com/hangxie/ray-images/issues/14).

Mount it at its own route prefix, for example `route_prefix: /chat`, and open the page with a trailing slash (`/chat/`), since it uses relative URLs: `GET /` (page), `GET /health`, `GET /api/voices`, `POST /api/chat`.

- It holds no models, so a replica asks for one CPU (Serve's default) and no GPU, and runs in the system Python: its imports are the standard library, FastAPI, and Ray.
- STT, the LLM, and TTS are URLs, so each can be this cluster's Serve apps, the Mac's llama.cpp, or anything else with the same API.
- A turn is one streamed request that mostly waits on the backends, so a replica takes up to `CHAT_MAX_ONGOING_REQUESTS` turns at once instead of TTS's one.
- A client that disconnects cancels its turn: the replica shuts its in-flight STT, LLM, and TTS connections, so TTS stops synthesizing at the next frame.
- Backend connections are kept alive and shared by a replica's turns, so STT calls and TTS sentences skip the TCP and TLS handshakes; through the homelab's HTTPS Gateway the handshake was ~28 of the ~33 ms each call cost. A connection is reused only after its response was read to the end, a pooled one the server has closed is replaced on the next request, and an interrupted turn's connections are closed, not pooled. The streamed LLM reply's connection is closed after `[DONE]`, since it is one call per turn.
- The page header and `/health` show the image tag, which the build writes to `serve/VERSION`, shared by all the apps; it reads `dev` outside the image. TTS and STT report it the same way.
- There is no client-IP allowlist, unlike on `media`; access control belongs to the deployment in front, as for TTS and STT. The `CHAT_*_TOKEN`s are only what the replica sends to its backends, for example a PAT when TTS sits behind the Gateway's bearer check.

| Setting | Default | Effect |
|---|---|---|
| `CHAT_STT_URL` | `http://localhost:5005/v1` | OpenAI-compatible base URL for `/audio/transcriptions` |
| `CHAT_LLM_URL` | `http://localhost:8080/v1` | OpenAI-compatible base URL for streamed `/chat/completions` |
| `CHAT_LLM_MODEL` | `qwen3.5-9b-q8-uncensored` | Model ID sent with each LLM request |
| `CHAT_LLM_MAX_TOKENS` | `512` | Cap on each reply |
| `CHAT_TTS_URL` | `http://localhost:5004/v1` | This image's TTS, including its `/v1` (`/health`, `/audio/voices`, `/audio/speech` as `pcm` over SSE) |
| `CHAT_VOICE` | unset | Voice for turns that name none; the page defaults to it, else to TTS's first voice |
| `CHAT_SYSTEM_PROMPT` | short, spoken, same-language replies | Replaces the built-in system prompt |
| `CHAT_STT_TOKEN`, `CHAT_LLM_TOKEN`, `CHAT_TTS_TOKEN` | unset | Sent as `Authorization: Bearer <token>` to that backend |
| `CHAT_NUM_REPLICAS` | `1` | Replicas |
| `CHAT_MAX_ONGOING_REQUESTS` | `16` | Turns (and page requests) a replica serves at once |

As for TTS, set these in the application's `runtime_env` `env_vars`; they are read on the head and passed to the replicas through `Chat.bind()`.

## Meeting notes

`serve_meeting:app` transcribes one long recording, such as a meeting, while people keep talking. It is a product on top of STT rather than part of it, so its page and the features planned for it (saved transcripts, summaries, speaker names) change without touching the GPU replica; see [#26](https://github.com/hangxie/ray-images/issues/26).

Mount it at its own route prefix, for example `route_prefix: /meeting`, and open the page with a trailing slash (`/meeting/`), since it uses relative URLs: `GET /` (page), `GET /health`, `WS /api/stream`.

- The page streams the microphone to `api/stream` as 16 kHz PCM. The replica opens STT's `audio/stream` with `MEETING_STT_TOKEN` and relays messages both ways unchanged, so the events are STT's (see [STT](#stt)); when either side closes, the other is closed too, with STT's close code and reason passed to the page. Segmenting, prompts, and previews stay in STT, which needs the same audio and GPU for them.
- Text appears as it is spoken: the piece being spoken shows in grey and is updated about every second, then replaced by its final text when the piece ends. Pieces join into a paragraph until a pause of 2 s or more. The meeting is one WebSocket for its whole length, so nothing queues in the browser; if it drops, recording stops and the transcript so far is kept. Proxies in front of both apps must pass WebSocket upgrades.
- The transcript is kept in the browser's local storage and can be copied, downloaded as text with a clock time per paragraph, or downloaded as WebVTT subtitles with one cue per piece. It stays until New note clears it, so a reload keeps it; a meeting that is running carries on into the new note.
- The language menu offers auto-detect and six common languages; Other… takes any language Whisper knows, by name or code, with suggestions from STT's `/health`. The replica reads that list once STT answers; until then the page still loads, and Other… falls back to auto-detect.
- Only meeting mode moved here. STT's old page also had hands-free and hold-to-talk modes, which uploaded single utterances to `audio/transcriptions`; STT keeps a plain Record/Stop page for that, and voice chat covers short speech too.
- Like voice chat, it holds no models, so a replica asks for one CPU and no GPU and runs in the system Python. The relay uses the `websockets` client that Ray Serve already installs (through `uvicorn[standard]`).
- Access control belongs to the deployment in front (in the homelab, OAuth2 Proxy for the page). `MEETING_STT_TOKEN` is only what the replica sends to STT.

| Setting | Default | Effect |
|---|---|---|
| `MEETING_STT_URL` | `http://localhost:8000/v1` | This image's STT, including its `/v1` (`/health`, `/audio/stream`); `ws://` or `wss://` follows from `http://` or `https://` |
| `MEETING_STT_TOKEN` | unset | Sent to STT as `Authorization: Bearer <token>` |
| `MEETING_NUM_REPLICAS` | `1` | Replicas |
| `MEETING_MAX_ONGOING_REQUESTS` | `16` | Meetings (and page requests) a replica serves at once |

As for chat, set these in the application's `runtime_env` `env_vars`; they are read on the head and passed to the replicas through `Meeting.bind()`.
