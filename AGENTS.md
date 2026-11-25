# AGENTS.md

Container image for the homelab Ray cluster, `hangxie/ray-images`: Ray with vLLM, plus Qwen3-TTS and Whisper STT Ray Serve apps in their own virtualenv and a CPU-only voice chat Serve app that drives them. The head, LLM workers, and TTS workers all run it. See `README.md` for what the image contains and how the TTS app behaves. CI (`.github/workflows/check.yml`) runs `make check` — ruff, mypy, and a CPU test suite against tiny random qwen-tts models. GPU behavior is only testable by building the image and running it on the homelab cluster.

## Layout

- `Dockerfile` — the image: vLLM and FlashInfer steps first, then Brotli, then the TTS virtualenv layer, then the Serve app.
- `tts/app/` — the Ray Serve apps: `serve_tts:app` from `serve_tts.py` (deployment, HTTP API, hooks, streaming), `cuda_graphs.py` (CUDA-graph decoding), `index.html`; `serve_stt:app` from `serve_stt.py` (Whisper, OpenAI transcription API, mounted at `/v1`); `serve_chat:app` from `serve_chat.py` (voice chat: STT → streamed LLM → TTS per sentence, no models, system Python), `chat.html`.
- `tts/check_requirements.py` — build-time dependency check for the TTS virtualenv.
- `scripts/prune_flashinfer_cache.py` — build-time removal of FlashInfer cache modules the target GPUs can't run.
- `scripts/check_markdown.py` — the Markdown rules below, run by `make lint`.
- `tests/` — pytest suite for the CPU-testable parts (see Testing).
- `pyproject.toml`, `uv.lock` — the development environment (uv), ruff, mypy and pytest settings. Not used by the image.
- `Makefile` — `make dev`, `format`, `lint` (ruff, mypy, Markdown), `test`, `check`, `pins`; `make build` → `hangxie/ray-images:$(VERSION)`.

## Rules

- **One image.** Don't split vLLM and TTS into separate images: the head's copy of `serve_tts` decides the deployment options and the workers' copy runs synthesis, so both must come from one build.
- **Two Python environments.** The system Python carries vLLM (transformers 5.x). `/home/ray/tts-venv` (`--system-site-packages`) carries qwen-tts and transformers 4.57.3 and reuses the system Ray, torch, torchaudio, and CUDA libraries; install into it with torch/torchaudio constrained to the system versions so it never gets its own copy. TTS replicas use it through `runtime_env` `py_executable`.
- **Dependency checks.** `pip check` must pass for the system environment. In the venv, `pip check` always reports vLLM's transformers requirement, so `tts/check_requirements.py` must pass there instead. Don't install with `--no-deps` or drop declared dependencies to save space.
- **Layer order.** Keep the TTS layers after the vLLM layers so a TTS change doesn't rebuild vLLM.
- **Serve app imports.** The head imports `serve_tts.py`, `serve_stt.py`, and `serve_chat.py` with the system Python to build the apps. Module-level imports stay limited to what both environments have (stdlib, numpy, scipy, fastapi, ray); import `torch`, `transformers`, `qwen_tts`, `soundfile` inside the replica. Nothing module-level that can't be pickled (e.g. `threading.local`) — Serve pickles the FastAPI app.
- **Replica settings go through `bind()`.** A deployment's `runtime_env` replaces the application's instead of merging, so replicas don't see the application's `env_vars`. Read `TTS_*` settings on the head (`settings()`) and pass them as `TTS.bind()` arguments; module-level globals must not decide replica behavior. Library variables that must reach the replica process (`HF_*`) go in the deployment's own `runtime_env`.
- **CUDA graphs.** `cuda_graphs.py` must stay equivalent to qwen-tts's eager path: greedy codes and talker hidden states from the graph bodies run eagerly must match HF `generate` on small random models (`tests/test_cuda_graphs.py`). Sampling options, the end-of-speech hook and the disconnect check stay outside the captured graphs. `TTS_CUDA_GRAPHS=0` falls back to eager decoding for comparison.
- **Target GPUs.** The image keeps only FlashInfer modules `FLASHINFER_KEEP_ARCHS` (default `8.6`, the RTX 3050s) can run. Add an architecture, or use `all`, before deploying to other GPUs. Peak TTS memory is 5.37 of 6 GB, so check `X-Peak-CUDA-GB` after changes that add GPU memory (`TTS_KV_CACHE_TOKENS` sizes the talker cache).
- **HTTP surface.** Keep `/`, `/health`, `/voices`, `POST /api/tts` (fields, ranges, error codes, `X-*` headers, the `stream` NDJSON events) compatible with site-up's `index.html`, `txt2mp3.sh` and voice chat, `serve_stt` compatible with OpenAI's `/v1/audio/transcriptions` as site-up's `mlx-stt-svc` serves it, and `serve_chat` (`/`, `/health`, `/api/voices`, `POST /api/chat` and its NDJSON events) compatible with site-up's voice chat page.
- **No auth in the image.** Don't add API keys or login checks to the Serve apps; the deployment provides them in front (README, Endpoints).
- **Streaming decode.** `decode_tail` repeats `Qwen3TTSTokenizerV2Decoder.forward` with the vocoder cut to the last frames; `tests/test_tts_generation.py` checks it, and that streamed pieces still match the full-clip decode; rerun it when upgrading qwen-tts.
- **Versions and labels.** Candidates under test are image tags `vX.Y.Z-rcN` only, not git tags. A release is a git tag `vX.Y.Z` on the merged commit plus the image tag, rebuilt from cache with `VERSION=vX.Y.Z` so only the version label changes from the tested rc. Upstream versions are `ARG`s recorded as `io.github.hangxie.ray-images.*` labels. The version label is set last in the Dockerfile and also passed with `--label` by the Makefile.
- **Fix in place.** No parallel versions, shims, or `v2` copies.
- **Docs.** `README.md` records *why* for non-obvious choices (version pairings, what is omitted to save size). Markdown prose must not use hard line breaks: keep each paragraph and list item on a single source line and let renderers wrap; code blocks and tables are exempt. Markdown files must not contain tabs.
- **Short commit messages.** Imperative subject under 72 chars; body at most one line saying what changed. Rationale, issue references (`Closes #N`), and testing go in the PR description.
- **No AI/agent attribution** in commits, PRs, issues, or docs (no `Co-Authored-By`, `Generated with ...`, etc.).

## Commands

```bash
make dev                            # .venv from uv.lock (uv sync --dev --locked)
make format                         # ruff format + safe fixes
make check                          # lint + tests; what CI runs
make lock                           # after changing pyproject.toml
make build                          # tagged with `git describe --tags`
make build VERSION=v0.4.0-rc1       # explicit tag
docker inspect -f '{{json .Config.Labels}}' hangxie/ray-images:<tag>
podman push localhost/hangxie/ray-images:<tag> docker.io/hangxie/ray-images:<tag>
```

## Change workflow

1. **Branch** off `main`.
2. **Check** (`make check`); lint and tests must pass. Add or update tests for what changed where it can run on CPU.
3. **Build locally** (`make build`); the build-time dependency and import checks must pass.
4. **Validate on the homelab cluster** for anything that changes runtime behavior (model loading, generation, CUDA libraries, Python environments). List what was validated, and what still needs it, in the PR description.
5. **Commit and PR** against `main`; push the branch to `origin`. Publish `vX.Y.Z-rcN` images for cluster testing as needed.
6. **Release** after merge: git tag `vX.Y.Z`, rebuild with `VERSION=vX.Y.Z`, push the image, and record cluster measurements (RTF, peak CUDA memory, image size) in the GitHub release notes.

## Testing

`make test` runs on CPU in the uv environment, which mirrors the TTS venv: Ray 2.59.0, torch/torchaudio/fastapi at vLLM 0.27.0's pins, qwen-tts 0.1.1 with transformers 4.57.3 (CPU torch wheels on Linux). Keep those pins in step with the Dockerfile; `tests/test_pins.py` fails when they drift. Its vLLM check reads PyPI, so it runs in `make pins` (and CI), not `make test`.

- `tests/conftest.py` builds tiny random qwen-tts models (talker + code predictor, and a speech-tokenizer decoder at the real upsampling rates), so tests run the code that reaches into qwen-tts internals against the installed version. Their weights are re-initialized larger than the default init, which decodes silence and attends almost uniformly, hiding mistakes.
- Generation runs under `torch.no_grad()`, as `generate_voice_clone` does; inference mode would make the static caches inference tensors.
- HTTP tests drive the real FastAPI routes with a hand-built replica (`object.__new__` on the deployment's class) by overriding Serve's `get_current_servable_instance` dependency; only the model is faked.
- `tests/test_head.py` imports both apps with the replica-only packages blocked, and pickles the app and replica class, as the head does.
- Not covered on CPU: `TTS.__init__`/`STT.__init__` (real checkpoints on `cuda:0`), `STT.transcribe`, `warm_up`, and CUDA graph capture/replay itself. Those still need the cluster.

## Gotchas

- Local builds on Apple Silicon run under amd64 emulation (qemu). The base image's `pyarrow` native library segfaults there, which also breaks `from ray import serve` and `import qwen_tts` (via gradio → pandas); build-time import checks therefore block it with `sys.modules['pyarrow'] = None`. A local Ray cluster doesn't start either (the GCS server never comes up), and GPU code can't run at all, so Serve and CUDA behavior are only testable on the cluster.
- Podman's layer cache reuses the final `LABEL` layer when only `VERSION` changes — hence the Makefile's `--label`.
- Podman stores local images as `localhost/<name>`; `podman push docker.io/<name>` fails with "image not known" but still exits 0. Push with an explicit source and destination.
- Docker Hub creates a repository on its first push, with the account's default visibility.
- The Podman VM disk fills up quickly with ~13 GB images and their build leftovers; a failed layer commit reports "no space".
- `docker build .` sends the whole directory as context; `.dockerignore` keeps only what the Dockerfile copies, so the dev `.venv` stays out. Add a path there when the Dockerfile copies a new one.
- A layer can't shrink what an earlier layer added; deleting base-image files in a later `RUN` doesn't reduce image size. Pruning must happen in the same `RUN` as the install.
- A full rebuild (empty cache) re-resolves unpinned dependencies and gives every layer a new digest, so the next push re-uploads all ~6 GB. Compare package versions with the previous image before shipping it.
- HF `generate` treats a compileable cache (`StaticCache`) as a request to auto-compile the forward and builds a 4D attention mask, which breaks the talker's rotary positions at prefill. `cuda_graphs.GraphCache` reports itself non-compileable for that reason.
- qwen-tts internals: `serve_tts.py` wraps the talker's `generate` for the end-of-speech and disconnect hooks (`Qwen3TTSForConditionalGeneration.generate` drops unknown kwargs), and `cuda_graphs.py` replaces `code_predictor.generate` and `talker.model.forward`. Rerun `make test` when upgrading qwen-tts or transformers, and bump their pins in `pyproject.toml` with the Dockerfile's.
