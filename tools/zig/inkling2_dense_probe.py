"""Bounded, non-serving comparison of Inkling's native cuBLASLt plans and staged BF16 GEMV."""
import argparse
import hashlib
import json
import mmap
import statistics
import struct
import time
from pathlib import Path

import torch

from inkling2_blas_probe import Linear
from inkling2_probe import Driver


SHAPES = (
    ("q", 3072, 4608, "bf16", "attn.wq_du.weight"),
    ("gate", 3072, 4608, "bf16", "attn.gate_attn.weight"),
    ("k", 512, 4608, "bf16", "attn.wk_dv.weight"),
    ("v", 512, 4608, "bf16", "attn.wv_dv.weight"),
    ("relative", 384, 4608, "bf16", "attn.wr_du.weight"),
    ("sink", 24, 4608, "bf16", "attn.attn_sink_proj.weight"),
    ("output", 4608, 3072, "f32", "attn.wo_ud.weight"),
    ("router", 384, 4608, "f32", "mlp.gate.weight"),
    ("dynamic", 16, 4608, "bf16", "attn.k_sconv.weight_proj"),
    ("head", 100512, 4608, "f32", "model.llm.unembed.weight"),
)


def stored(root, role, suffix, n, k, layer=None):
    if layer is None:
        layer = 5 if role == "dynamic" else 0
    filename = "global.safetensors" if role == "head" else f"layer-{layer:02}.safetensors"
    key = suffix if role == "head" else f"model.llm.layers.{layer}.{suffix}"
    with (root / "dense" / filename).open("rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
        item = header[key]
        if item["dtype"] != "BF16":
            raise ValueError("Stored projection changed precision")
        mapping = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)
        full = torch.frombuffer(mapping, dtype=torch.bfloat16, count=item["shape"][0] * item["shape"][1],
                                offset=8 + size + item["data_offsets"][0]).reshape(item["shape"])
        # Rank zero's row slice, except O which is sharded on input columns.
        result = full[:n, :k].contiguous().clone()
        del full
        mapping.close()
    return result


def bits_equal(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def guarded(shape, dtype):
    elements = 1
    for width in shape:
        elements *= width
    storage = torch.full((elements + 512,), 123, device="cuda", dtype=dtype)
    return storage, storage[256:-256].view(shape)


def guards(storage):
    return bool((storage[:256] == 123).all() and (storage[-256:] == 123).all())


def accuracy(y, oracle):
    error = (y.double() - oracle).abs()
    return {"max_abs": error.max().item(), "mean_abs": error.mean().item(),
            "worst_column": int(error.max(0).values.argmax().item())}


def time_candidates(candidates, weights, x, y, scrub, rounds):
    records = {name: [] for name in candidates}
    names = list(candidates)
    graphs = {}
    for name, run in candidates.items():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for w in weights:
                run(x, w, y)
        graphs[name] = graph
    for round_id in range(rounds):
        order = names if round_id % 2 == 0 else names[::-1]
        for name in order:
            scrub.add_(1)
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            records[name].append(begin.elapsed_time(end) * 1000 / len(weights))
    return {name: {"microseconds": values, "median_us": statistics.median(values)}
            for name, values in records.items()}


def check_candidate(run, baseline, x, w, out, xstore, wstore, ystore, y):
    want = torch.empty_like(y)
    baseline(x, w, want)
    torch.cuda.synchronize()
    errors, stable = [], True
    previous = None
    for _ in range(5):
        y.fill_(float("nan"))
        run(x, w, y)
        torch.cuda.synchronize()
        errors.append(int((y.view(torch.uint8) != want.view(torch.uint8)).sum().item()))
        stable = stable and (previous is None or bits_equal(y, previous))
        previous = y.clone()
    oracle = torch.empty(y.shape, device="cuda", dtype=torch.float64)
    # Bound peak FP64 allocation and compute every output column independently of cuBLASLt.
    for first in range(0, w.shape[0], 1024):
        oracle[:, first:first + 1024] = x.double() @ w[first:first + 1024].double().T
    return {"different_bytes": errors, "repeat_identical": stable,
            "input_unchanged": bits_equal(x, out[0]) and bits_equal(w, out[1]),
            "guards_intact": all(guards(s) for s in (xstore, wstore, ystore)),
            "baseline_fp64": accuracy(want, oracle), "candidate_fp64": accuracy(y, oracle)}


def run_shape(spec, args, driver, workspace, scrub, records):
    role, n, k, kind, suffix = spec
    dtype = torch.bfloat16 if kind == "bf16" else torch.float32
    xstore, x = guarded((1, k), torch.bfloat16)
    wstore, w = guarded((n, k), torch.bfloat16)
    ystore, y = guarded((1, n), dtype)
    w.copy_(stored(args.model, role, suffix, n, k))
    wsave = w.clone()
    plan = Linear(1, n, k, 14 if kind == "bf16" else 0, workspace)
    def blas(h):
        return lambda a, b, c: plan.run(h, a, b, c, torch.cuda.current_stream().cuda_stream)
    candidates = {f"blas-{i}": blas(h) for i, h in enumerate(plan.plans)}
    if kind == "bf16" and n % 8 == 0 and k % 256 == 0:
        candidates["staged"] = lambda a, b, c: driver.launch("drama3_staged_gemv", (n // 8,), 32, a, b, c, k)
    baseline = blas(plan.baseline)
    record = {"role": role, "shape": [1, n, k], "output": kind, "cublaslt_version": plan.version,
              "plans": [{"name": f"blas-{i}", "algorithm_hex": bytes(h.algo).hex(),
                         "workspace_bytes": h.workspace, "waves": h.waves} for i, h in enumerate(plan.plans)],
              "checks": {}, "timings": {}}
    records.append(record)
    try:
        for pattern in ("normal", "cancellation", "wide"):
            torch.manual_seed(1701)
            if pattern == "normal":
                x.copy_(torch.randn(x.shape, device="cuda"))
            elif pattern == "cancellation":
                x.copy_(torch.arange(k, device="cuda").remainder(2).mul_(2).sub_(1).expand_as(x))
            else:
                values = torch.randn(x.shape, device="cuda")
                values *= torch.pow(2., torch.arange(k, device="cuda").remainder(17).sub_(8))
                x.copy_(values)
            xsave = x.clone()
            record["checks"][pattern] = {name: check_candidate(run, baseline, x, w, (xsave, wsave),
                                                           xstore, wstore, ystore, y)
                                         for name, run in candidates.items()}
        # Rotate at least 128 MiB of distinct weights, and scrub another 64 MiB outside the timed interval.
        copies = max(2, (128 * 1024**2 + w.numel() * 2 - 1) // (w.numel() * 2))
        weights = [w] + [w.clone() for _ in range(copies - 1)]
        for run in candidates.values():
            run(x, w, y)
        torch.cuda.synchronize()
        record["timings"] = time_candidates(candidates, weights, x, y, scrub, args.rounds)
        record["rotated_weight_bytes"] = copies * w.numel() * 2
        record["eligible"] = [name for name in candidates if all(
            not any(checks[name]["different_bytes"]) and checks[name]["repeat_identical"]
            and checks[name]["input_unchanged"] and checks[name]["guards_intact"]
            for checks in record["checks"].values())]
        if "blas-0" not in record["eligible"]:
            raise RuntimeError("The reference failed repeat, input or guard checks")
        print(json.dumps({"role": role, "eligible": record["eligible"], "median_us": {
            name: t["median_us"] for name, t in record["timings"].items()}}), flush=True)
    finally:
        torch.cuda.synchronize()
        plan.close()
        args.output.write_text(json.dumps(records, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--fatbin", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=9)
    p.add_argument("--only", choices=[s[0] for s in SHAPES])
    args = p.parse_args()
    if not 3 <= args.rounds <= 21 or args.output.exists():
        raise ValueError("Use 3..21 rounds and a new output file")
    torch.cuda.set_per_process_memory_fraction(.12)
    torch.cuda.init()
    if torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError("This experiment is restricted to an idle GB10")
    # Canary before the bounded real-weight cases.
    assert (torch.ones(256, device="cuda") + 1).sum().item() == 512
    torch.backends.cuda.matmul.allow_tf32 = False
    workspace = torch.empty(64 * 1024**2, device="cuda", dtype=torch.uint8)
    scrub = torch.zeros(64 * 1024**2, device="cuda", dtype=torch.uint8)
    driver, records = Driver(args.fatbin), []
    metadata = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
                "capability": list(torch.cuda.get_device_capability()), "memory_fraction": .12,
                "fatbin_sha256": hashlib.sha256(args.fatbin.read_bytes()).hexdigest(), "started": time.time()}
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with torch.inference_mode():
        for spec in SHAPES:
            if args.only is None or spec[0] == args.only:
                run_shape(spec, args, driver, workspace, scrub, records)
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
