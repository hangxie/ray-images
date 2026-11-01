# ray-vllm

Ray image (`rayproject/ray`, slim) with vLLM installed, published as `hangxie/ray-vllm:<tag>`.

```bash
make build
```

Image tags are this repo's own `vX.Y.Z`. The upstream versions are set as `ARG`s at the top of the
`Dockerfile` and recorded as image labels:

```bash
docker inspect -f '{{json .Config.Labels}}' hangxie/ray-vllm:<tag>
```

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
