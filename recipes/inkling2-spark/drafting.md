# Inkling's bundled DSpark drafter

The `kortexa-ai/inkling2-draft` branch adds the checkpoint's five-layer BF16
DSpark model to the existing native two-Spark profile. Target weights,
activation precision and exact BF16 KV remain unchanged. One block contains
an anchor and up to seven proposals. The shared lane engine chooses the
verification depth; with its default empty cost table it chooses one to three
proposals. A seven-proposal architecture does not mean eight output tokens
are accepted per round.

## Portable launch

On both Sparks, use the same checkout and build its image:

```sh
git checkout kortexa-ai/inkling2-draft
./recipes/inkling2-spark/setup.sh
```

Add `--drafts` to the normal downloader on each host, using its own rank:

```sh
./recipes/inkling2-spark/download.sh --rank 0 --drafts \
  --directory "$HOME/models/inkling2-native"
```

Use `--rank 1` on the peer. Existing verified target files are reused in place.
The extra download is 3,526,756,456 bytes per host: the original BF16
`auxiliary/mtp.safetensors` plus `auxiliary/dspark-fold.safetensors`.
Both are SHA256-verified. The target and MTP remain pinned to revision
`05c6cd0dff9df5bf20b10419c663a1e2adc7eb64`; the folded head is pinned to
`f82bc6f03ec8c61a1d3afd155c820b71bf05bf7c` in the same
[public checkpoint repository](https://huggingface.co/kortexa-ai/Inkling-2-Small-EXL3-2.59bpw).
No PyTorch preparation is required at startup.

Set `DRAFTS=1` on **both** foreground launchers, retaining the normal network,
model path, rank and context settings from the [main recipe](README.md):

```sh
DRAFTS=1 RANK=0 MASTER_ADDR=192.0.2.10 \
  MODEL_DIR="$HOME/models/inkling2-native" CONTEXT=8192 \
  ./recipes/inkling2-spark/start.sh
```

Use `RANK=1` on the peer. `DRAFTS=0` remains the default and passes
`--no-drafts`, so it neither loads nor allocates the draft model. Both ranks
must select the same mode. The profile supports one greedy text request;
sampling, thinking, vision, prompt caching and structured output remain outside
its qualified scope.

## Arithmetic and state

The adapter consumes eight target feature taps, caches each draft layer's
context K/V, and uses the checkpoint's recurrent rank-256 vocabulary head.
The folded head computes the unembedding times the published hidden-output
weight and bias in FP64, then rounds to BF16. Feature-mixture softmax uses
FP32 and rounds to BF16. The pinned artifact includes those mixture weights.
The original draft model stays BF16. The preparation source is
[`inkling2_prepare_draft.py`](../../tools/zig/inkling2_prepare_draft.py).

Target verification retains each ordinary one-row projection's sum order.
Draft mode shards the target input embedding by vocabulary rows, saving
0.863 GiB per rank. A lookup gathers and selects the original owner's BF16
bytes; it does not add the two ranks' outputs. Plain mode retains its previous
embedding layout. The per-stream memory report includes the draft K/V cache.
Convolution history has one snapshot per accepted prefix. Rejected future K/V
is ignored by position; local rings leave enough margin for a complete window.
Committed feature taps are absorbed before the next proposal, including fully
accepted windows where the lane engine does not invoke rollback.

`TF_INKLING_VERIFY_CHECK=1` is a slow diagnostic. It serially replays each
verified input and compares the target tokens, FP32 logits, BF16 final hidden,
all layer K/V rows and convolution prefix histories byte for byte. Leave it
off for performance measurements. A canary with an 8K allocation checks short prose and code against
the frozen plain target's complete 64-token arrays. Closing the client during
decode also leaves the next fresh request's complete output unchanged.

The focused CUDA contract runner covers the new draft and verification
operators through the shared G1–G5 gates and live compile limits. These checks
are specific to GB10/SM121; they do not establish coverage of another GPU or
the complete CUDA backend.

## Measured performance

Two GB10s, a 524288-token allocation, fresh process per A-B-B-A arm, greedy
256-token replies, short/8K warmups, fresh KV and no prompt caching:

| Request | Plain tok/s | DSpark tok/s | Change | Accepted proposals |
|---|---:|---:|---:|---:|
| Short prose (33 inputs) | 16.59 | 18.33 | +10.5% | 61.8% |
| Code (48) | 16.59 | 19.76 | +19.1% | 69.0% |
| 8K (8086) | 11.77 | 12.37 | +5.1% | 61.1% |
| 32K (32194) | 7.47 | 6.74 | −9.8% | 59.5% |

All 4096 generated token IDs match the plain target across the four arms.
Short-prompt prefill rates fall about 5.1%; 8K and 32K prefill stay within 1%.
Minimum observed host availability is 6.17 GiB. Median SM clocks differ by at
most 0.21%, with no active clock-event flags. The draft mode produces
1.67–1.88 output tokens per verification round, using the API's counter definition.

The fixed promotion bar requires at least 10% short/code/8K decode gain and no
regression above 3% in another measured decode or prefill cell. This version
fails that bar. Drafting stays opt-in, and the selected plain profile is
preserved. Use plain decoding when long-context throughput is the priority.
Verification still runs ordinary attention separately for every proposed row;
that work remains a target for long-context optimization.

[Full request bodies, input/output IDs, samples, memory and acceptance](draft-measurements.json)
pin target source `0b78fd235ff268cc136554e6d6ad8f0b7761579d` and draft source
`7511f80febdd58d06387ed05c5ee0755c383fd66`. The draft executable SHA256 is
`6331bf90e797cdeccd8c88b7933b110b0b8a935e79b305847be5271e5e4cf449`.
The 512K allocation demonstrates capacity through the measured workloads;
full 512K prefill speed and retrieval quality remain unqualified.
