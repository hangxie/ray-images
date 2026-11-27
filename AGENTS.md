# AGENTS.md

Container image `hangxie/ray-images` for the homelab Ray cluster: Ray with vLLM, plus Ray Serve apps for TTS (Qwen3-TTS), STT (Whisper), voice chat and meeting notes. `README.md` says what the image contains and why. CI runs `make check`; GPU behavior is only testable on the homelab cluster.

## Layout

- `Dockerfile` — vLLM and FlashInfer layers first, then the voice virtualenv, then the Serve apps.
- `serve/{tts,stt,chat,meeting}/` — one Serve app each (`serve_<name>:app`), plus `serve/common/` for shared code; all on the image's `PYTHONPATH`.
- `serve/check_requirements.py`, `scripts/` — build-time checks and FlashInfer pruning; `scripts/check_markdown.py` enforces the Markdown rules below.
- `tests/` — CPU pytest suite. `pyproject.toml`/`uv.lock` are the dev environment only, not the image.

## Rules

- **One image.** The head's copy of each app decides deployment options and the workers' copy runs it, so vLLM and the Serve apps ship in one build.
- **Two Python environments.** System Python has vLLM (transformers 5.x). `/home/ray/voice-venv` (`--system-site-packages`) adds qwen-tts, transformers 4.57.3 and silero-vad on top of the system Ray, torch and CUDA; install into it with torch/torchaudio constrained to the system versions. TTS and STT replicas use it via `runtime_env` `py_executable`.
- **Dependency checks.** `pip check` must pass in the system environment and `serve/check_requirements.py` in the venv (where `pip check` always flags vLLM's transformers pin). No `--no-deps` or dropped dependencies to save space.
- **Layer order.** Serve app and venv layers stay after the vLLM layers.
- **Serve app imports.** The head imports every app with the system Python, so module-level imports are limited to stdlib, numpy, scipy, fastapi, ray and websockets; import `torch`, `transformers`, `qwen_tts`, `soundfile` inside the replica. Nothing unpicklable at module level (e.g. `threading.local`): Serve pickles the FastAPI app.
- **Replica settings go through `bind()`.** A deployment's `runtime_env` replaces the application's, so read settings on the head (`settings()`) and pass them to `.bind()`; module-level globals must not decide replica behavior. Variables the replica process itself needs (`HF_*`) go in the deployment's `runtime_env`.
- **CUDA graphs.** `cuda_graphs.py` must match qwen-tts's eager path (`tests/test_cuda_graphs.py`). Sampling, the end-of-speech hook and the disconnect check stay outside captured graphs; `TTS_CUDA_GRAPHS=0` decodes eagerly.
- **Target GPUs.** FlashInfer modules are pruned to `FLASHINFER_KEEP_ARCHS` (default `8.6`, the RTX 3050s); widen it before deploying elsewhere. TTS peaks at 5.37 of 6 GB, so check `X-Peak-CUDA-GB` after changes that add GPU memory.
- **HTTP surface.** Existing routes, fields, error codes, `X-*` headers and NDJSON events are a contract with site-up's TTS page, `txt2mp3.sh` and voice chat page, and with OpenAI clients (`/v1/audio/speech`, `/v1/audio/transcriptions`). STT's page stays a plain uploader; live features go in the meeting app, which redeploys without reloading Whisper.
- **No auth in the image.** The deployment authenticates in front; apps only log the identity headers it sets.
- **Versions and labels.** Test candidates are image tags `vX.Y.Z-rcN` only. A release is git tag `vX.Y.Z` on the merged commit plus an image rebuilt from cache with `VERSION=vX.Y.Z`, so only the version label and `serve/VERSION` differ from the tested rc.
- **Fix in place.** No parallel versions, shims, or `v2` copies.
- **Docs.** `README.md` records *why* for non-obvious choices. Markdown prose has no hard line breaks (one line per paragraph or list item) and no tabs.
- **Commits.** Imperative subject under 72 chars, body at most one line; rationale, testing and `Closes #N` go in the PR. No AI/agent attribution anywhere.

## Commands

```bash
make dev                         # .venv from uv.lock
make format                      # ruff format + fixes
make check                       # lint + tests (CI)
make lock                        # after changing pyproject.toml
make build VERSION=v0.4.0-rc1    # default tag: git describe --tags
podman push localhost/hangxie/ray-images:<tag> docker.io/hangxie/ray-images:<tag>
```

## Change workflow

1. Branch off `main`; `make check` passes, with CPU tests for what changed.
2. `make build`; its dependency and import checks must pass.
3. Validate runtime changes (models, generation, CUDA, Python environments) on the cluster via `vX.Y.Z-rcN` images; list what was and wasn't validated in the PR.
4. After merge, release: tag, rebuild with `VERSION=vX.Y.Z`, push, and put cluster measurements (RTF, peak CUDA memory, image size) in the GitHub release notes.

## Testing

- The uv environment mirrors the voice venv's pins; `tests/test_pins.py` fails when they drift from the Dockerfile (its PyPI check runs in `make pins` and CI).
- `tests/conftest.py` builds tiny random qwen-tts models with enlarged weights (the default init decodes silence and hides mistakes), so tests exercise the code that reaches into qwen-tts internals.
- HTTP tests drive the real FastAPI routes with a hand-built replica (`object.__new__`) through Serve's `get_current_servable_instance` dependency; only the model is faked. Page scripts run in node against fake browser APIs and skip without node; extend them when a page changes.
- Not covered on CPU: model loading on `cuda:0`, `STT.transcribe`, `warm_up`, and CUDA graph capture/replay.

## Gotchas

- Image builds on Apple Silicon run under qemu, where the base image's `pyarrow` segfaults (breaking `ray.serve` and `qwen_tts` imports), so build-time checks set `sys.modules['pyarrow'] = None`. Ray doesn't start inside the emulated image, but Serve runs natively from the dev `.venv` (`serve run serve_meeting:app` with the `serve/*` dirs on `PYTHONPATH`; keep `RAY_TMPDIR` short, sockets are capped at 103 bytes).
- Podman reuses the final `LABEL` layer when only `VERSION` changes, hence the Makefile's `--label`. Local images are `localhost/<name>`; `podman push docker.io/<name>` fails with "image not known" yet exits 0, so push with explicit source and destination.
- The Podman VM disk fills fast with ~13 GB images; a failed layer commit reports "no space".
- `.dockerignore` excludes everything the Dockerfile doesn't copy (above all `.venv`); add new paths there.
- Deleting files in a later layer doesn't shrink the image; prune in the same `RUN` as the install.
- A cache-less rebuild re-resolves unpinned packages and re-uploads all ~6 GB; compare package versions with the previous image first.
- HF `generate` auto-compiles with a compileable cache and builds a 4D mask that breaks the talker's rotary positions, so `cuda_graphs.GraphCache` reports itself non-compileable.
- `serve_tts.py` wraps the talker's `generate` for its hooks and `decode_tail` repeats the speech-tokenizer decoder's `forward`, and `cuda_graphs.py` replaces `code_predictor.generate` and `talker.model.forward`; rerun `make test` and bump pins in both `pyproject.toml` and the Dockerfile when upgrading qwen-tts or transformers.
