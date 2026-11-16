"""Check batched verification projections and convolution rollback against plain native steps."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from inkling2_dense_probe import accuracy, bits_equal, guarded, guards, stored, time_candidates
from inkling2_probe import Driver

SHAPES = (("q", 3072, 4608, "attn.wq_du.weight", torch.bfloat16),
          ("gate", 3072, 4608, "attn.gate_attn.weight", torch.bfloat16),
          ("output", 4608, 3072, "attn.wo_ud.weight", torch.float32),
          ("head", 100512, 4608, "model.llm.unembed.weight", torch.float32))


def projections(a, candidate, baseline_bf16, baseline_f32, records):
    scrub = torch.zeros(64 * 1024**2, device="cuda", dtype=torch.uint8)
    for role, n, k, suffix, dtype in SHAPES[:1] if a.canary else SHAPES:
        ws, w = guarded((n, k), torch.bfloat16)
        w.copy_(stored(a.model, role, suffix, n, k))
        saved_w = w.clone()
        old = baseline_bf16 if dtype == torch.bfloat16 else baseline_f32
        old_name = "drama3_staged_gemv" if dtype == torch.bfloat16 else "tf_ik_dense_gemv_f32"
        name = "tf_ik_verify_bf16" if dtype == torch.bfloat16 else "tf_ik_verify_f32"
        for rows in (1, 2) if a.canary else (1, 2, 4, 8):
            xs, x = guarded((rows, k), torch.bfloat16)
            ys, y = guarded((rows, n), dtype)
            want = torch.empty_like(y)

            def plain(xx, ww, yy):
                for row in range(rows):
                    old.launch(old_name, (n // 8,), 32, xx[row:row + 1], ww, yy[row:row + 1], k)

            def batch(xx, ww, yy):
                candidate.launch(name, (n // 8 * rows,), 32, xx, ww, yy, k, n, rows)

            item = {"role": role, "shape": [rows, n, k], "checks": []}
            records.append(item)
            for pattern in ("normal",) if a.canary else ("normal", "cancellation", "wide"):
                x.copy_(torch.randn(x.shape, device="cuda"))
                if pattern == "cancellation":
                    x.copy_((torch.arange(k, device="cuda") % 2 * 2 - 1).expand_as(x))
                elif pattern == "wide":
                    x.mul_(torch.pow(2., torch.arange(k, device="cuda") % 17 - 8))
                saved_x = x.clone()
                plain(x, w, want)
                torch.cuda.synchronize()
                changes, stable, previous = [], True, None
                for _ in range(5):
                    y.fill_(float("nan"))
                    batch(x, w, y)
                    torch.cuda.synchronize()
                    changes.append(int((y.view(torch.uint8) != want.view(torch.uint8)).sum().item()))
                    stable &= previous is None or bits_equal(y, previous)
                    previous = y.clone()
                result = {"pattern": pattern, "different_bytes": changes, "repeat_identical": stable,
                          "input_unchanged": bits_equal(x, saved_x) and bits_equal(w, saved_w),
                          "guards_intact": all(guards(s) for s in (xs, ws, ys))}
                if not a.canary:
                    oracle = torch.empty(y.shape, device="cuda", dtype=torch.float64)
                    for start in range(0, n, 1024):
                        oracle[:, start:start + 1024] = x.double() @ w[start:start + 1024].double().T
                    result["baseline_fp64"] = accuracy(want, oracle)
                    result["candidate_fp64"] = accuracy(y, oracle)
                item["checks"].append(result)
                if any(changes) or not stable or not result["input_unchanged"] or not result["guards_intact"]:
                    raise AssertionError(item)
            if rows == 8:
                copies = max(2, (128 * 1024**2 + w.numel() * 2 - 1) // (w.numel() * 2))
                weights = [w] + [w.clone() for _ in range(copies - 1)]
                item["timings"] = time_candidates({"plain-eight": plain, "row-equivalent-batch": batch},
                                                 weights, x, y, scrub, 9)
                item["rotated_weight_bytes"] = copies * w.numel() * 2
                del weights
            print(json.dumps({"role": role, "rows": rows, "checks": "passed", "timings": item.get("timings")}), flush=True)
        del w, ws, saved_w
        torch.cuda.empty_cache()


def convolutions(candidate, records):
    for rows in (1, 2, 3, 4, 8):
        for width in (512, 4608):
            x = torch.randn((rows, width), device="cuda").bfloat16()
            old = torch.randn((3, width), device="cuda").bfloat16()
            stride = 3 * width + 32
            storage, out = guarded(((rows + 1) * stride,), torch.bfloat16)
            out.fill_(123)
            previous = old.clone()
            candidate.launch("tf_ik_verify_conv", ((width + 255) // 256, rows + 1), 256,
                             x, old, out, rows, width, stride)
            torch.cuda.synchronize()
            for keep in range(rows + 1):
                want = torch.cat((old, x[:keep]))[-3:]
                got = out[keep * stride:keep * stride + 3 * width].view(3, width)
                if not bits_equal(got, want) or not bool((out[keep * stride + 3 * width:(keep + 1) * stride] == 123).all()):
                    raise AssertionError((rows, width, keep))
            assert bits_equal(old, previous) and guards(storage)
            records.append({"convolution_rows": rows, "width": width, "prefixes_exact": rows + 1})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "candidate", "baseline-bf16", "baseline-f32", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--canary", action="store_true")
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("Use a new output path")
    torch.cuda.set_per_process_memory_fraction(.12)
    if torch.cuda.get_device_capability() != (12, 1):
        raise ValueError("This probe requires an idle GB10")
    assert (torch.ones(256, device="cuda") + 1).sum().item() == 512
    torch.manual_seed(1701)
    torch.backends.cuda.matmul.allow_tf32 = False
    record = {"seed": 1701, "torch": torch.__version__, "memory_fraction": .12,
              "fatbins": {key: hashlib.sha256(getattr(a, key).read_bytes()).hexdigest()
                          for key in ("candidate", "baseline_bf16", "baseline_f32")}, "projections": [], "convolution": []}
    try:
        candidate = Driver(a.candidate)
        with torch.inference_mode():
            projections(a, candidate, Driver(a.baseline_bf16), Driver(a.baseline_f32), record["projections"])
            convolutions(candidate, record["convolution"])
        record["status"] = "passed"
    finally:
        a.output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
