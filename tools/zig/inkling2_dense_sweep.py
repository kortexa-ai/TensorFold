"""Check the staged GEMV against native cuBLASLt on both TP shards of all Q and gate weights."""
import argparse
import json
from pathlib import Path

import torch

from inkling2_blas_probe import Linear
from inkling2_dense_probe import accuracy, bits_equal, guarded, guards, stored
from inkling2_probe import Driver


def check(plan, driver, x, w, y, reference):
    stream = torch.cuda.current_stream().cuda_stream
    plan.run(plan.baseline, x, w, reference, stream)
    saved, stable, differences = None, True, []
    for _ in range(5):
        y.fill_(float("nan"))
        driver.launch("drama3_staged_gemv", (384,), 32, x, w, y, 4608)
        torch.cuda.synchronize()
        differences.append(int((y.view(torch.uint8) != reference.view(torch.uint8)).sum().item()))
        stable = stable and (saved is None or bits_equal(saved, y))
        saved = y.clone()
    result = {"different_bytes": differences, "repeat_identical": stable}
    if any(differences):
        columns = (y.view(torch.int16) != reference.view(torch.int16)).flatten().nonzero().flatten()
        oracle = x.double() @ w[columns].double().T
        result.update(columns=columns.cpu().tolist(), baseline=reference[:, columns].float().cpu().tolist(),
                      candidate=y[:, columns].float().cpu().tolist(), oracle=oracle.cpu().tolist(),
                      baseline_fp64=accuracy(reference[:, columns], oracle),
                      candidate_fp64=accuracy(y[:, columns], oracle))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--fatbin", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Preserve the previous receipt")
    torch.cuda.set_per_process_memory_fraction(.12)
    torch.cuda.init()
    if torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError("This experiment is restricted to an idle GB10")
    torch.backends.cuda.matmul.allow_tf32 = False
    workspace = torch.empty(64 * 1024**2, device="cuda", dtype=torch.uint8)
    driver = Driver(args.fatbin)
    plan = Linear(1, 3072, 4608, 14, workspace)
    records = []
    try:
        with torch.inference_mode():
            for layer in range(48):
                for role, suffix in (("q", "attn.wq_du.weight"), ("gate", "attn.gate_attn.weight")):
                    for shard in range(2):
                        # Slice both physical tensor partitions with the same public model geometry.
                        full = stored(args.model, role, suffix, 6144, 4608, layer)
                        wstore, w = guarded((3072, 4608), torch.bfloat16)
                        w.copy_(full[shard * 3072:(shard + 1) * 3072])
                        del full
                        wsave = w.clone()
                        xstore, x = guarded((1, 4608), torch.bfloat16)
                        ystore, y = guarded((1, 3072), torch.bfloat16)
                        reference = torch.empty_like(y)
                        record = {"layer": layer, "role": role, "shard": shard, "checks": []}
                        for seed in range(1701, 1717):
                            torch.manual_seed(seed)
                            x.copy_(torch.randn(x.shape, device="cuda"))
                            xsave = x.clone()
                            result = check(plan, driver, x, w, y, reference)
                            result.update(seed=seed, input_unchanged=bits_equal(x, xsave))
                            record["checks"].append(result)
                        record.update(weight_unchanged=bits_equal(w, wsave),
                                      guards_intact=all(guards(s) for s in (xstore, wstore, ystore)))
                        records.append(record)
                        args.output.write_text(json.dumps(records, indent=2) + "\n")
                        print(json.dumps({"layer": layer, "role": role, "shard": shard,
                                          "failed_inputs": sum(any(r["different_bytes"]) for r in record["checks"])}),
                              flush=True)
                        del wstore, w, wsave, xstore, x, ystore, y, reference
                        torch.cuda.empty_cache()
    finally:
        torch.cuda.synchronize()
        plan.close()


if __name__ == "__main__":
    main()
