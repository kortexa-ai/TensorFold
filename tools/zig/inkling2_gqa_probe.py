"""Bounded non-serving GQA decode comparison against the qualified native tile."""
import argparse
import gc
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from inkling2_probe import Attention, Driver


VARIANTS = ((2, 4), (2, 8), (3, 8), (6, 16))


def equal(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def guarded(shape):
    size = 1
    for dim in shape:
        size *= dim
    store = torch.full((size + 256,), 123, device="cuda", dtype=torch.bfloat16)
    return store, store[128:-128].view(shape)


def fixture(pos, window, pattern="normal"):
    capacity = pos + 1 if not window else min(pos + 1, 1056)
    pairs = [guarded(shape) for shape in ((1, 24, 128), (capacity, 4, 128), (capacity, 4, 128),
                                         (1, 24, 1024), (1, 24, 128), (1, 24), (1, 24, 128))]
    stores, arrays = map(list, zip(*pairs))
    for tensor in arrays:
        tensor.copy_(torch.randn(tensor.shape, device="cuda") * .25)
    q, k, v, relative, gate, sink, out = arrays
    if pattern == "cancel":
        q.fill_(1)
        k.fill_(0)
        relative.zero_()
        v.copy_((torch.arange(128, device="cuda") % 4 == 0).expand_as(v))
    elif pattern != "normal":
        target = {"query": q, "key": k, "value": v, "relative": relative, "gate": gate, "sink": sink}
        field, kind = pattern.split("/")
        target[field].view(-1)[0] = {"nan": float("nan"), "inf": float("inf"),
                                    "ninf": -float("inf"), "subnormal": 2.**-133}[kind]
    return stores, arrays, capacity


def launch(driver, name, group, splits, arrays, pos, window, capacity):
    args = Attention(*(t.data_ptr() for t in arrays), 1, pos, 24, 4, capacity, window)
    driver.launch(name, (1, 24 // group, splits), 128, args)


def check(baseline, candidate, pos, window, pattern):
    stores, arrays, capacity = fixture(pos, window, pattern)
    saved = [a.clone() for a in arrays[:-1]]
    launch(baseline, "tf_ik_attention_decode", 1, 4, arrays, pos, window, capacity)
    torch.cuda.synchronize()
    want = arrays[-1].clone()
    result = {"position": pos, "window": window, "pattern": pattern, "variants": {}}
    for group, splits in VARIANTS:
        different, stable, previous = [], True, None
        for _ in range(5):
            arrays[-1].fill_(123)
            launch(candidate, f"tf_ik_gqa_{group}_{splits}", group, splits, arrays, pos, window, capacity)
            torch.cuda.synchronize()
            got = arrays[-1]
            # NaN payload is not portable; preserve classes and all finite output bytes.
            finite = torch.isfinite(want)
            classes = torch.equal(torch.isnan(got), torch.isnan(want)) and torch.equal(
                torch.isinf(got), torch.isinf(want)) and torch.equal(got[torch.isinf(want)], want[torch.isinf(want)])
            different.append(int((got[finite].view(torch.int16) != want[finite].view(torch.int16)).sum().item()))
            stable &= previous is None or equal(got, previous)
            previous = got.clone()
            if not classes:
                different[-1] += 1
        intact = all(bool((s[:128] == 123).all() and (s[-128:] == 123).all()) for s in stores)
        inputs = all(equal(a, b) for a, b in zip(arrays[:-1], saved))
        record = {"different": different, "repeat_identical": stable, "guards_intact": intact,
                  "input_unchanged": inputs}
        result["variants"][f"g{group}s{splits}"] = record
        if any(different) or not stable or not intact or not inputs:
            raise AssertionError(result)
    return result


def timing(baseline, candidate, pos, window, rounds):
    stores, arrays, capacity = fixture(pos, window)
    kv_bytes = arrays[1].numel() * 4
    copies = max(2, (64 * 1024**2 + kv_bytes - 1) // kv_bytes)
    buffers = [(arrays[1], arrays[2])] + [(arrays[1].clone(), arrays[2].clone()) for _ in range(copies - 1)]
    configs = [("baseline", baseline, "tf_ik_attention_decode", 1, 4)] + [
        (f"g{g}s{s}", candidate, f"tf_ik_gqa_{g}_{s}", g, s) for g, s in VARIANTS]
    graphs, samples = {}, {name: [] for name, *_ in configs}
    scrub = torch.zeros(64 * 1024**2, dtype=torch.uint8, device="cuda")
    for name, driver, symbol, group, splits in configs:
        launch(driver, symbol, group, splits, arrays, pos, window, capacity)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for k, v in buffers:
                values = [arrays[0], k, v, *arrays[3:]]
                launch(driver, symbol, group, splits, values, pos, window, capacity)
        graphs[name] = graph
    for round_id in range(rounds):
        for name in list(graphs)[::1 if round_id % 2 == 0 else -1]:
            scrub.add_(1)
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(begin.elapsed_time(end) * 1000 / copies)
    return {"position": pos, "window": window, "rotated_kv_bytes": copies * kv_bytes,
            "samples_us": samples, "median_us": {name: statistics.median(v) for name, v in samples.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--canary", action="store_true")
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("Use a fresh output path")
    torch.cuda.set_per_process_memory_fraction(.12)
    if torch.cuda.get_device_capability() != (12, 1):
        raise ValueError("Only an idle GB10 is admitted")
    assert (torch.ones(256, device="cuda") + 1).sum().item() == 512
    torch.backends.cuda.matmul.allow_tf32 = False
    baseline, candidate = Driver(a.baseline), Driver(a.candidate)
    record = {"baseline_sha256": hashlib.sha256(a.baseline.read_bytes()).hexdigest(),
              "candidate_sha256": hashlib.sha256(a.candidate.read_bytes()).hexdigest(),
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "memory_fraction": .12,
              "started": time.time(), "checks": [], "timings": []}
    try:
        shapes = [(31, 0)] if a.canary else [(0, 0), (63, 0), (64, 0), (1023, 0), (1024, 1023),
                                             (2049, 1023), (8191, 0), (32767, 0)]
        for pos, window in shapes:
            record["checks"].append(check(baseline, candidate, pos, window, "normal"))
            print(json.dumps({"checked": [pos, window, "normal"]}), flush=True)
        if not a.canary:
            for pattern in ["cancel"] + [f"{f}/{v}" for f in ("query", "key", "value", "relative", "gate", "sink")
                                          for v in ("nan", "inf", "ninf", "subnormal")]:
                record["checks"].append(check(baseline, candidate, 127, 0, pattern))
            for pos, window in ((2049, 1023), (8191, 0), (32767, 0), (131071, 0)):
                gc.collect()
                torch.cuda.empty_cache()
                item = timing(baseline, candidate, pos, window, 9)
                record["timings"].append(item)
                print(json.dumps(item), flush=True)
        record["status"] = "passed"
    finally:
        a.output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
