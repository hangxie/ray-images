# ray-vllm

Ray image (`rayproject/ray`, slim CPU variant) with vLLM installed, published as `hangxie/ray-vllm:<tag>`.

The base has no CUDA toolkit: torch brings its CUDA runtime as pip wheels and the NVIDIA container
toolkit mounts the driver. `gcc` is installed because Triton compiles small launcher stubs at runtime.

FlashInfer's full CUDA 13.0 JIT-cache wheel is included so the default vLLM sampler can load its
kernel without `nvcc`. The optional `flashinfer-cubin` bundle is omitted to reduce image size;
operations that need those cubins download them on demand and require network access.
The build checks that the sampler binary is present for an `sm_86` architecture target. Keep `FLASHINFER_VERSION` in the `Dockerfile` aligned with
`flashinfer-python`, and update the cache's CUDA suffix when torch's CUDA runtime changes.
Prebuilt kernels cover the upstream wheel's supported configurations; other configurations can still
require JIT compilation and a CUDA toolkit. GPU execution must be validated on the target hardware.

```bash
make build
```

Image tags are this repo's own `vX.Y.Z`. The upstream versions are set as `ARG`s at the top of the
`Dockerfile` and recorded as image labels. Ray 2.59.0 is paired with vLLM 0.27.0,
matching Ray Serve LLM's exact vLLM requirement. The base remains the slim
`rayproject/ray:2.59.0-py312-cpu` image; torch supplies the CUDA runtime.

Upgrading changes the compile and Triton cache keys, so expect a cold first start.
Revalidate model-specific engine settings and tool-call parsers before rollout.

Inspect the version labels with:

```bash
docker inspect -f '{{json .Config.Labels}}' hangxie/ray-vllm:<tag>
```

## FlashInfer sampler smoke test

Run this on a GPU host to check the prebuilt sampler without allowing an nvcc fallback:

```bash
docker run --rm --gpus all -e FLASHINFER_DISABLE_JIT=1 hangxie/ray-vllm:<tag> \
  python3 -c "import torch; from flashinfer.sampling import top_k_top_p_sampling_from_probs; p = torch.full((4, 128), 1 / 128, device='cuda'); s = top_k_top_p_sampling_from_probs(p, 16, 0.9); torch.cuda.synchronize(); assert s.shape == (4,) and ((s >= 0) & (s < 128)).all().item(); print('FlashInfer sampler OK')"
```

The driver (`libcuda.so.1`) must be mounted by the NVIDIA container runtime. A CPU-only build
can check that the cache exists, but cannot load or execute its CUDA kernels.

## Loading weights from S3

vLLM is installed with its `runai` extra, so the Run:ai Model Streamer and its S3 backend
(`runai_model_streamer_s3`) are in the image. Point vLLM at a bucket with `load_format="runai_streamer"`:

```bash
AWS_ENDPOINT_URL=http://<rgw-host> \
AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... \
RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING=0 \
python3 -c "from vllm import LLM; LLM(model='s3://<bucket>/<model>', load_format='runai_streamer')"
```

- `AWS_ENDPOINT_URL` sets the endpoint for non-AWS stores (Ceph RGW, MinIO, R2). The streamer reads it
  for both listing and reading objects; leave it unset for AWS S3.
- `RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING=0` switches to path-style URLs (`http://host/bucket/key`),
  which most self-hosted stores need. The default (`1`) uses virtual-hosted style (`http://bucket.host/key`).
- Credentials come from the usual `AWS_*` variables. `AWS_REGION` may be needed for stores that check it.
