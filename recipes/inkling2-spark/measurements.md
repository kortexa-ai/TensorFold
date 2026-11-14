# Measured native text profile

Native source: `aa7b7c9998131da29569f002727f9b51195f3548`.
Weights: `05c6cd0dff9df5bf20b10419c663a1e2adc7eb64`.
Two DGX Spark GB10s, direct RoCE, unchanged mixed K2/K3 EXL3, checkpoint
activation rounding, exact BF16 KV, one greedy request, no prompt cache.

| Workload | Native result | Previous pipeline |
| --- | ---: | ---: |
| 8128-token retrieval, prefill/TTFT | 19.08 s | 409.51 s |
| Same retrieval, total request | 20.08 s | 418.31 s |
| Same retrieval, input rate | 425.9 tokens/s | 19.85 tokens/s |
| Same retrieval, decode rate | 10.01 tokens/s | about 1.25 tokens/s |
| 33-input / 256-output prose | 14.93 decode tokens/s | not measured |

Both retrieval runs answered `orchid, 47, nebula, 42`. These single-request
measurements exclude weight loading. TTFT includes native request setup and
prefill. Decode rate counts intervals after the first output token. The older
pipeline used scaled FP8 KV; execution, transport, and stop handling differ, so
this is an end-to-end comparison rather than an isolated kernel speedup.

The synthetic [retrieval prompt](retrieval-8k.txt) has SHA256
`7fe1ef52180657318dc8d1cceabc33aeedac0455d1de40113866e2ad15a29250`.
From the repository root, send the same request to a running API rank:

```sh
python3 - <<'PY' | curl -sS http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d @-
import json
from pathlib import Path
prompt = Path('recipes/inkling2-spark/retrieval-8k.txt').read_text()
print(json.dumps({'model': 'inkling-2-small', 'messages': [{'role': 'user', 'content': prompt}],
                  'temperature': 0, 'max_tokens': 64}))
PY
```

The native server log reports prompt count, TTFT, elapsed time and decode rate.

The native run allocated the 524288-token BF16 cache, reporting 104.34 GiB of
runtime allocations on the head. Minimum observed host availability was
8.56 GiB. This checks capacity with short requests, not 512K prefill speed or
retrieval quality. The complete upstream kernel-contract runner, a broad quality
suite, vision, sampling, drafts, concurrent requests, and prompt caching remain
unqualified. The profile rejects features it does not implement.

The [machine-readable receipt](measurements.json) records the source-built image,
executable digest, toolchain and successful greedy-only HTTP 400 check.
