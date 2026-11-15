# Inkling grouped-query decode experiment

This experiment failed its fixed performance bar and is not selected by native
serving. The [dense recipe](dense.md) remains the measured serving path.

Inkling has six query heads per KV head. The existing one-query decode tile
uses one row of a 16-row MMA. The experiment packs query heads sharing K/V into
independent rows, preserving each output's Q.K, softmax and three-limb FP32 P.V
reduction order. Output partitions retain at least 48 CTAs on a 48-SM GB10.
The kernels are separate assets; the original attention fatbin is byte-identical.

## Measured bound

A fresh trace of the dense build, with the same checkpoint, BF16 KV, EP2/TP2
and 524288 allocation, gives these rank-zero GPU intervals per complete decode
forward. The second rank agrees within 0.3 ms for attention.

| Input context | Step, ms | Local attention, ms | Global attention, ms |
|---|---:|---:|---:|
| 8K | 85.46 | 10.61 | 16.05 |
| 32K | 133.69 | 10.65 | 63.40 |

The fixed bar was at least 10% faster served decode at both lengths, exact
tokens, and no loss over 3% in short decode or prefill. A hypothetical 1.5x–2x
attention improvement cleared that bound. The implementation did not reach it.

## Synthetic kernel comparison

Times below are microseconds for one rank's attention call. G is query heads
per group; S is output partitions. They are not full-model token rates.

| Context | Existing | G2/S4 | G2/S8 | G3/S8 | G6/S16 |
|---|---:|---:|---:|---:|---:|
| Local 1024 | 259.54 | 242.87 | 251.22 | 237.11 | 237.50 |
| Global 8192 | 1980.85 | 1856.69 | 1921.71 | 1798.32 | 1797.80 |
| Global 32768 | 7913.34 | 7411.18 | 7670.11 | 7168.88 | 7168.34 |
| Global 131072 | 32043.89 | 30006.62 | 30746.46 | 28645.73 | 28633.47 |

Even choosing the fastest tile separately for local and global layers projects
only 2.89% faster full decode at 8K and 5.43% at 32K. These are optimistic
compositions of measured GPU times, not candidate served measurements. The
experiment stopped before a model build or promotion.

A separate address calculation tested one remainder per tile followed by an
exact wrap subtraction, retaining the original expression for capacities below
64. Its G1/S4 and G3/S8 entries took 17964/17792 us at 32K, against 7914 us
for the existing tile in that run. It also stopped. These results do not
establish integer division as the binding limit.

## Evidence and limits

Both probes use CUDA graphs, nine alternating-order rounds, 64 MiB of scrub
outside events and distinct K/V buffers exceeding the 24 MiB L2. Rotated K/V
storage is 64–512 MiB, depending on the case. Inputs and outputs are BF16;
accumulation stays FP32. The probe uses a 12% allocator cap on an idle GB10.

Each candidate passes 33 synthetic fixture classes with five repeated outputs,
unchanged inputs and intact guards. Finite bytes and nonfinite classes match
the existing decode tile. Query exceptions retain a per-head fallback so
neighboring finite heads keep their original arithmetic. The initial fixtures
did not record their RNG seed or input bytes. The helper now fixes and records
seed 1701 for future comparisons; it does not backfill the original receipts.
The fixture named `cancel` uses uniform values across keys, so it does not
qualify cancellation stress. The helper now names that fixture `uniform`.

[Raw synthetic checks and timing samples](gqa-measurements.json) retain both
attempts and their source/fatbin hashes. The six entries have compiler and
operator registrations, but their formal device contracts were not run after
the projected speed failed the bar. The changed kernels were exercised on one
physical GB10 only. There is no candidate served token, prefill, concurrency
or whole-model quality qualification, and no functional SM120 claim.

Host checks pass: 713/713 build steps with 4989 tests passed and 45 skipped;
the CUDA host subset passes 395/395 steps with 5022 tests passed and 28 skipped.
The plain build passes 33/33 steps, and the ARM test runner compiles in 162/162.
The lean checker reports 342 inherited findings and zero new findings.

The ordinary dense build's fresh 256-token short/8K/32K profile replies match
the preceding dense A-B-B-A replies exactly. This is baseline evidence, not
candidate model fidelity.

The development probe is `tools/zig/inkling2_gqa_probe.py`. Build the CUDA
assets with the pinned native toolchain, then pass the unchanged `inkling2`
fatbin as `--baseline`, the `inkling2_gqa` fatbin as `--candidate` and a fresh
JSON path as `--output`. Use `--canary` for the small first case. Torch and
Python are development dependencies only; the serving recipe is unchanged.
