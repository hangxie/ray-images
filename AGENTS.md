# AGENTS.md

Two container images for the homelab Ray clusters: `hangxie/ray-vllm` (Ray with vLLM) and `hangxie/ray-voice` (Ray Serve apps for TTS with Qwen3-TTS, STT with Whisper, voice chat and meeting notes). `README.md` says what they contain and why. CI runs `make check`; GPU behavior is only testable on the homelab cluster.

## Layout

- `docker/vllm/Dockerfile`, `docker/voice/Dockerfile` — one per image, both built from the repository root.
- `serve/{tts,stt,chat,meeting}/` — one Serve app each (`serve_<name>:app`), plus `serve/common/` for shared code; all on the image's `PYTHONPATH`.
- `scripts/` — FlashInfer pruning for the vLLM image; `scripts/check_markdown.py` enforces the Markdown rules below; `txt2mp3.py`/`txt2m4b.py` (with `tts_client.py`) narrate text through TTS and run outside the image.
- `tests/` — CPU pytest suite. `pyproject.toml`/`uv.lock` are the dev environment only, not an image.

## Rules

- **Two images, one environment each.** Each Ray cluster runs one image on its head and workers; vLLM (transformers 5.x) and qwen-tts (transformers 4.57.3) never share one. Both images use the same `RAY_VERSION` (`tests/test_pins.py`).
- **Dependency checks.** `pip check` must pass in both images. No `--no-deps` or dropped dependencies to save space. In the voice image, keep torch/torchaudio at their `ARG`s on every install so nothing replaces them.
- **Layer order.** Big, rarely changing layers first (vLLM and FlashInfer; torch in the voice image), Serve apps last.
- **Serve app imports.** The head imports every app to build it, and chat and meeting replicas are CPU-only, so module-level imports are limited to stdlib, numpy, scipy, fastapi, ray and websockets; import `torch`, `transformers`, `qwen_tts`, `soundfile` inside the replica. Nothing unpicklable at module level (e.g. `threading.local`): Serve pickles the FastAPI app.
- **Replica settings go through `bind()`.** Read settings on the head (`settings()`) and pass them to `.bind()`; module-level globals must not decide replica behavior. Don't give a deployment its own `runtime_env`: it replaces the application's, so replicas would lose its `env_vars` (e.g. `HF_HUB_OFFLINE`).
- **CUDA graphs.** `cuda_graphs.py` must match qwen-tts's eager path (`tests/test_cuda_graphs.py`). Sampling, the end-of-speech hook and the disconnect check stay outside captured graphs; `TTS_CUDA_GRAPHS=0` decodes eagerly.
- **Target GPUs.** The vLLM image's FlashInfer modules are pruned to `FLASHINFER_KEEP_ARCHS` (default `8.6`, the RTX 3050s); widen it before deploying elsewhere. TTS peaks at 5.37 of 6 GB, so check `X-Peak-CUDA-GB` after changes that add GPU memory.
- **HTTP surface.** TTS and STT speak OpenAI's APIs (`/v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/models`) for OpenAI clients; this repo's own clients (the pages, voice chat, meeting notes, `scripts/tts_client.py`) use them too and change with them in the same PR. Extensions stay under `/v1` and in OpenAI's shapes: extra request fields, `X-*` headers, `/v1/health`, `/v1/audio/voices`, and STT's `/v1/audio/stream` until `/v1/realtime` replaces it (#35). STT's page stays a plain uploader; live features go in the meeting app, which redeploys without reloading Whisper.
- **No auth in the image.** The deployment authenticates in front of the Serve apps; apps only log the identity headers it sets.
- **Versions and labels.** Both images share one version stream. Test candidates are image tags `vX.Y.Z-rcN` only. A release is git tag `vX.Y.Z` on the merged commit plus both images rebuilt from cache with `VERSION=vX.Y.Z`, so only the version label (and the voice image's `serve/VERSION`) differ from the tested rc; the release notes say which image changed.
- **Fix in place.** No parallel versions, shims, or `v2` copies.
- **Comments.** Only where the code can't speak for itself: one terse line on why, as a fragment, not a story. Docstrings are a one-line summary plus anything surprising. Longer rationale goes in `README.md`.
- **Docs.** `README.md` records *why* for non-obvious choices. Markdown prose has no hard line breaks (one line per paragraph or list item) and no tabs.
- **Commits.** Imperative subject under 72 chars, body at most one line; rationale, testing and `Closes #N` go in the PR. No AI/agent attribution anywhere.

## Commands

```bash
make dev                         # .venv from uv.lock
make format                      # ruff format + fixes
make check                       # lint + tests (CI)
make lock                        # after changing pyproject.toml
make build VERSION=v0.4.0-rc1    # both images; build-vllm/build-voice for one. Default tag: git describe --tags
podman push localhost/hangxie/ray-voice:<tag> docker.io/hangxie/ray-voice:<tag>
```

## Change workflow

1. Branch off `main`; `make check` passes, with CPU tests for what changed.
2. `make build`; its dependency and import checks must pass.
3. Validate runtime changes (models, generation, CUDA, Python environments) on the cluster via `vX.Y.Z-rcN` images; list what was and wasn't validated in the PR.
4. After merge, release: tag, rebuild with `VERSION=vX.Y.Z`, push, and put cluster measurements (RTF, peak CUDA memory, image size) in the GitHub release notes.

## Testing

- The uv environment mirrors the voice image's pins; `tests/test_pins.py` fails when they drift from `docker/voice/Dockerfile`.
- `tests/conftest.py` builds tiny random qwen-tts models with enlarged weights (the default init decodes silence and hides mistakes), so tests exercise the code that reaches into qwen-tts internals.
- HTTP tests drive the real FastAPI routes with a hand-built replica (`object.__new__`) through Serve's `get_current_servable_instance` dependency; only the model is faked. Page scripts run in node against fake browser APIs and skip without node; extend them when a page changes.
- Not covered on CPU: model loading on `cuda:0`, `STT.transcribe`, `warm_up`, and CUDA graph capture/replay.

## Gotchas

- Image builds on Apple Silicon run under qemu, where the base image's `pyarrow` segfaults (breaking `ray.serve` and `qwen_tts` imports), so build-time checks set `sys.modules['pyarrow'] = None`. Ray doesn't start inside the emulated image, but Serve runs natively from the dev `.venv` (`serve run serve_meeting:app` with the `serve/*` dirs on `PYTHONPATH`; keep `RAY_TMPDIR` short, sockets are capped at 103 bytes).
- Podman reuses the final `LABEL` layer when only `VERSION` changes, hence the Makefile's `--label`. Local images are `localhost/<name>`; `podman push docker.io/<name>` fails with "image not known" yet exits 0, so push with explicit source and destination.
- The Podman VM disk fills fast with ~13 GB images; a failed layer commit reports "no space".
- `.dockerignore` (shared by both Dockerfiles) excludes everything they don't copy (above all `.venv`); add new paths there.
- Deleting files in a later layer doesn't shrink the image; prune in the same `RUN` as the install.
- A cache-less rebuild re-resolves unpinned packages and re-uploads all ~6 GB; compare package versions with the previous image first.
- HF `generate` auto-compiles with a compileable cache and builds a 4D mask that breaks the talker's rotary positions, so `cuda_graphs.GraphCache` reports itself non-compileable.
- `serve_tts.py` wraps the talker's `generate` for its hooks and `decode_tail` repeats the speech-tokenizer decoder's `forward`, and `cuda_graphs.py` replaces `code_predictor.generate` and `talker.model.forward`; rerun `make test` and bump pins in both `pyproject.toml` and `docker/voice/Dockerfile` when upgrading qwen-tts or transformers.
