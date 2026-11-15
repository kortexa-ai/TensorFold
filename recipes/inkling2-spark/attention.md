# Warm attention decode on two DGX Sparks

Four independent output partitions improve 32K warm decode from 6.04 to
7.17 tokens/s (+18.7%). The checkpoint, activation rounding, exact BF16 KV,
EP2/TP2 layout and 524288-token allocation remain the same.

| Workload | Original decode tokens/s | Partitioned decode tokens/s | Change |
| --- | ---: | ---: | ---: |
| 33-token prose prompt | 14.85 | 15.00 | +1.0% |
| 48-token code prompt | 14.85 | 15.04 | +1.3% |
| 8086-token prose prompt | 9.91 | 10.95 | +10.5% |
| 32194-token prose prompt | 6.04 | 7.17 | +18.7% |

These are medians of two samples per version in A-B-B-A order, with a fresh
paired process for each arm. Each timed request generates 256 tokens. All four
arms have identical full input and output token arrays. We use one greedy text
request, with no drafts or prompt cache. Weight loading is excluded. Each arm
first runs arithmetic, short/8K warmups and an 8K retrieval check.

Prefill remains essentially level: 426.5 to 425.5 input tokens/s at 8K (-0.23%)
and 349.0 to 348.7 at 32K (-0.09%). The one-row fast path applies on SM121;
multi-row prompt passes and other device classes retain the original kernel.
The configured 512K cache was allocated. Full-model 512K prefill, retrieval
quality and throughput have not been qualified.

## What changed

The original one-query grid has 24 blocks for the GB10's 48 SMs. The new grid
uses four blocks per head (96 total), each responsible for different output
columns. Each output keeps the original Q.K, softmax and three-limb P.V order.
There is no reduction across blocks, precision change or extra scratch buffer.
The original attention entry remains unchanged.

Native source is `b588a33f27f4def4d058d6616ba6f8c129e7252b`; baseline is
`aa7b7c9998131da29569f002727f9b51195f3548`. The baseline's later recipe-only
commit `6d75948b1defb2949f56fe4462536c3e211606d5` has identical native source.
Both images use the pinned CUDA 13.0.88 base and Zig 0.17.0. The weights remain
at `05c6cd0dff9df5bf20b10419c663a1e2adc7eb64`.

Five-second samples during timed requests show 2184–2190 MHz on rank 0 and
2171–2177 MHz on rank 1, with no active clock-event flags. Maximum temperatures
were 67/66 C. Minimum observed host memory availability was 8.05 GiB.

## Correctness and limits

The registered focused CUDA contract passes three operator forms on SM120 and
on both SM121 devices: exceptional values, masked/ring cases, five identical
repeats with guarded outputs and unchanged inputs, 16 shape cases per form,
and cancellation stress through a synthetic 524288-key cache. The missing-output
partition control fails as intended; launch-limit checks report zero problems.
These synthetic checks establish operator behavior, not model quality at 512K.

An additional 37-case RTX PRO 6000 probe is byte-identical to the original.
With cold buffers, its isolated 32K kernel time falls from 8.947 to 6.159 ms.
All 36 original functions retain identical SASS across SM120/121. An initial
variant that changed query-row indexing and the split grid axis was slower;
retaining the original row scheduling and using grid Z produced the measured
winner. This does not isolate a single compiler mechanism as the cause.

Host checks pass all 736 build steps, including tests, kernel-contract host
tests, goldens and Cortheon fuzzing. Plain build passes. Lean has 342 inherited
findings and zero new findings. This is a limited fork experiment; it does not
claim complete backend, cross-platform or broad model-quality qualification.
The experimental bar was +10% 32K decode, exact outputs, and no regression
greater than 3% in short/8K decode or prefill. The measured workloads meet it.

## Reproduce

Use the [portable setup and two-rank launch](README.md). For an A/B comparison,
build the original and candidate commits into separate `INKLING_IMAGE` tags on
both hosts. Keep the same weights and `CONTEXT=524288`. Start both ranks with
the same tag, warm them, send the four workloads, stop both, and repeat in
A-B-B-A order. No service manager is required.

The [receipt](attention-measurements.json) contains exact request bodies, full
expected token arrays, all sample timings, image/executable hashes and sampled
hardware ranges. For example, send its 32K workload to a running API rank:

```sh
python3 - <<'PY' | curl -N http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d @-
import json
from pathlib import Path
receipt = json.loads(Path('recipes/inkling2-spark/attention-measurements.json').read_text())
print(json.dumps(receipt['workloads']['32k-prose']['request']))
PY
```

The final SSE event includes native prefill/decode timing, full output token IDs
and their SHA256. Compare the full IDs with the receipt. The optional registered
operator check needs Zig 0.17.0 and CUDA 13.0.88 on an idle authorized GPU:

```sh
zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -j4
zig-out/bin/tf-cuda-test kernel-contract inkling
```

That command is a focused check. It does not replace complete backend gates.
