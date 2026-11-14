# Inkling 2 Small on two DGX Sparks

This experimental native TensorFold profile uses the published mixed K2/K3
EXL3 checkpoint, whole-expert EP2, attention TP2, and exact BF16 KV storage.
It serves one greedy text request at a time. Vision, drafts, structured output,
sampling, and prompt caching are not qualified. The native family rejects
unsupported requests. Python is used only to download files and prepare metadata.

You need two Linux ARM64 DGX Sparks, NVIDIA Container Toolkit, Docker, Python 3,
and a working network between them. The direct ConnectX connection should have
an IPv4 address on each host; choose the API host's address as `MASTER_ADDR`.
Allow the master port and its next port between hosts. Run both ranks together;
each holds approximately half of the expert weights. Leave other GPU workloads
stopped during inference.

On **each Spark**, clone this fork and check out the same commit:

```sh
git clone https://github.com/kortexa-ai/TensorFold.git
cd TensorFold
git checkout kortexa-ai/inkling2-spark
./recipes/inkling2-spark/setup.sh
```

The image pins an ARM64 CUDA 13/NCCL base and verifies Zig 0.17.0's archive.
Its executable is compiled from this checkout; the base image's Python inference
engine is not used. Build on each host, or transfer the resulting Docker image.

On the API Spark:

```sh
./recipes/inkling2-spark/download.sh --rank 0 --directory "$HOME/models/inkling2-native"
RANK=0 MASTER_ADDR=192.168.100.10 MODEL_DIR="$HOME/models/inkling2-native" \
  ./recipes/inkling2-spark/start.sh
```

On the other Spark, use its own directory and the **same** master address:

```sh
./recipes/inkling2-spark/download.sh --rank 1 --directory "$HOME/models/inkling2-native"
RANK=1 MASTER_ADDR=192.168.100.10 MODEL_DIR="$HOME/models/inkling2-native" \
  ./recipes/inkling2-spark/start.sh
```

Substitute your network address for the examples. Downloads are pinned to
`05c6cd0dff9df5bf20b10419c663a1e2adc7eb64` and SHA256-verified. Allow about
103 GiB of disk per host, plus the image/build cache. An existing identical pack
can be reused with `download.sh ... --reuse /path/to/pack`; verified files are
hardlinked when possible. Original weights, config, and tokenizer assets stay
in their published locations. Derived root metadata selects EXL3 and supplies
the non-thinking generation prefix the native chat interface needs.

Once the API rank reports ready:

```sh
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"inkling-2-small","messages":[{"role":"user","content":"What is 6 times 7?"}],"temperature":0,"max_tokens":32}'
```

`CONTEXT` defaults to 524288 tokens. Exact BF16 KV consumes approximately 8.1 GiB
per host at that context, in addition to model weights and scratch. This is a
capacity budget, not a long-context quality or throughput result. Use
`CONTEXT=8192` for a small first trial. The model's architectural ceiling is
1048576; do not infer tested capability from that ceiling.

Other optional environment values are `INKLING_IMAGE`, `GPU_UUID`, `API_PORT`,
`API_HOST`, `MASTER_PORT`, `NCCL_SOCKET_IFNAME`, and `RESERVE_GIB`. Defaults bind
the API to loopback, port 8000, with 6 GiB of host/device headroom. For remote
clients use an authenticated proxy or the native server's authentication flags.
Both ranks must use the same context. The script detects the transport interface;
set `NCCL_SOCKET_IFNAME` explicitly if routing chooses the wrong connection.
`TF_ROCE=0` selects the native NCCL fallback when direct RoCE is unavailable.

Both ranks run in the foreground without a restart policy. To stop, send an
interrupt to each launcher or `docker stop tensorfold-inkling2-rank0` and
`docker stop tensorfold-inkling2-rank1` on their respective hosts. A broken peer
ends the native process. Stop the pair before starting another distributed model.

## Development evidence

The first bounded RTX PRO 6000 test loaded one complete real layer and matched
the pinned PyTorch/ExLlamaV3 reference's top token. Its final FP32 logit relative
L2 error was 2.65e-5. Activation quantization, capped SwiGLU, routing, and static
and dynamic convolutions matched the reference in initial operator probes.
These development checks do not establish whole-model or upstream kernel-contract
qualification. Full-model quality, two-host stability, and speed need measured
receipts before treating this profile as production ready.
