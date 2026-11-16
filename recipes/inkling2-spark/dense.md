# Inkling dense projections on two GB10s

The dense path reduces one-row projection latency while preserving the original
BF16 inputs/weights, FP32 sums, and BF16 Q/gate or F32 O/head outputs. It reuses
this tree's `drama3_staged_gemv` and adds a separate F32-output entry. It is
selected only for SM121, cuBLASLt 13.1.1 (`130101`), one row and these dimensions:

| Projection | Output width | Input width | Output type |
|---|---:|---:|---|
| Q and gate | 3072 | 4608 | BF16 |
| O | 4608 | 3072 | F32 |
| Vocabulary head | 100512 | 4608 | F32 |

Other shapes and multi-row GEMMs keep the original cuBLASLt path. Weights,
activation precision, expert kernels, BF16 KV, 512K allocation and EP2/TP2 stay
fixed. The [main recipe](README.md) downloads and starts the same checkpoint;
the dense path needs no extra serving dependency or flag.

## Served comparison

On two DGX Spark GB10s, with a fresh process per A-B-B-A arm, greedy 256-token
replies, short/8K warmups, fresh KV and no drafts or prefix cache:

| Request | Baseline tok/s | Dense tok/s | Gain |
|---|---:|---:|---:|
| Short prose (33 inputs) | 14.97 | 16.52 | +10.4% |
| Code (48) | 14.97 | 16.54 | +10.5% |
| 8K (8086) | 10.92 | 11.75 | +7.7% |
| 32K (32194) | 7.15 | 7.49 | +4.8% |

All full input/output arrays match across all four arms: 4096 generated tokens.
Prefill stays within 1%; 8K input rate is 429.71 to 429.08 tokens/s and 32K is
349.05 to 347.82. Sampled median SM clocks differ by at most 0.14%, with no
active clock-event flags. Minimum observed host availability is 8.04 GiB.
The fixed experimental bar was a 5% short/code/8K decode gain with no regression
over 3% in other measured decode or prefill cells; it passes.

[Exact request bodies, input/output IDs, all samples and sensor summaries](dense-measurements.json)
record the comparison. Native source is `0b78fd235ff268cc136554e6d6ad8f0b7761579d`,
with executable SHA256 `9926a56f37c6064371efe65ddca9221189f50592fab0c2780db761a0103f4ddf`.
The preceding attention baseline is `b588a33f27f4def4d058d6616ba6f8c129e7252b`.
The build uses Zig 0.17.0 and nvcc 13.0.88 from the pinned ARM64 recipe image.

## Isolated and operator evidence

The development probe uses native Linear's exact cuBLASLt descriptors, its
single-result heuristic and 64 MiB workspace. Timing uses CUDA graphs, distinct
rotated weight buffers covering at least 128 MiB and a 64 MiB scrub outside the
CUDA events, over nine alternating-order rounds. Each GB10 reports 24 MiB L2.

| Projection | cuBLASLt, microseconds | Staged, microseconds | Latency reduction |
|---|---:|---:|---:|
| Q | 173.81 | 131.61 | 24.3% |
| Gate | 166.43 | 127.53 | 23.4% |
| O | 161.12 | 126.80 | 21.3% |
| Head | 5396.35 | 3966.56 | 26.5% |

The staged path is slower on smaller projections; V also changes a byte in a
normal-input fixture. Faster alternative cuBLASLt plans often change bytes.
Those alternatives remain excluded. The probe retains them as failed candidates.
`tools/zig/inkling2_dense_probe.py` and `inkling2_dense_sweep.py` reproduce the
development checks in a matching CUDA/Torch environment on an idle GB10. Python
is used only by these tools, not by native serving.

All 48 layers and both shards pass 16 inputs per Q/gate matrix: 3072 inputs and
9437184 output elements. O passes 1536 inputs and 7077888 F32 elements. Each
case runs five times, with exact bytes, unchanged inputs/weights and intact
guards. First-shape normal, cancellation and wide-range checks also compare
independent FP64 errors and worst columns.

The registered focused contract runs on both physical GB10s:

```text
(filtered) SUMMARY 3 kernels: 3 pass gates 1-4, 0 fail, 0 known findings; 1 controls: 1 red as planted, 0 missed; 0 limit problems
```

It checks exceptional classes, repeats, guards, exact replacement and FP32
stress sums. Fixed dimensions pass; eight unsupported tile edges per form are
refused. Live limits report 32 threads, 118 registers and 12288 shared bytes for
each entry. SM120/121 compiles are recorded; functional selection is SM121 only.
The contract runner source is `c2ef619686eb8153f2a192ebc6a5d8779e876c6b`, which
fixes contract-only compilation and cleanup without changing served arithmetic.
The probe and native dense assets have identical SM121 instruction words; the
preceding attention fatbin stays byte-identical. Host checks pass, with zero
new lean findings above 342 inherited ones.

This focused receipt does not establish complete CUDA-backend or broad model
quality qualification. The recipe remains an experimental greedy text profile.

## Upstream check

The 2026-11-15 audit found no Inkling matches in upstream code, issues or PRs,
or in 206 branch names. The delivery recheck used main
`cf4555373554867a72c53263e905f203a03d4d96`; the release was still
[v1.2.9](https://github.com/ashhart/TensorFold/releases/tag/v1.2.9). Shared Linear
and Drama staged GEMV are unchanged there. This recipe is maintained in the fork.
