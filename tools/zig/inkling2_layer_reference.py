"""Compare the native layer dump with the pinned standalone oracle, using exact BF16 KV on both sides."""
import argparse
import importlib
import json
import sys
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--ids", type=Path, required=True)
    p.add_argument("--native-dir", type=Path, required=True)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if torch.cuda.device_count() != 1 or "RTX PRO 6000" not in torch.cuda.get_device_name(0):
        raise RuntimeError("This bounded layer oracle requires the authorized RTX PRO 6000")
    torch.cuda.set_per_process_memory_fraction(.20)
    if torch.cuda.mem_get_info()[0] < 16 * 1024**3:
        raise RuntimeError("Less than 16 GiB free before layer reference")
    if args.layers != 1:
        raise ValueError("The bounded reference currently admits one complete real layer")
    sys.path.insert(0, str(args.reference_dir.resolve()))
    ref = importlib.import_module("model")
    attn = importlib.import_module("attention")

    def append(cache, k, v, *, local):
        pos = cache.get("position", 0)
        rows = len(k)
        if "k" in cache:
            k, v = torch.cat((cache["k"], k)), torch.cat((cache["v"], v))
        key_offset = pos + rows - len(k)
        cache["k"], cache["v"] = (k[-1023:].clone(), v[-1023:].clone()) if local else (k, v)
        cache["position"] = pos + rows
        return k, v, key_offset

    def exact_attention(q, k, v, relative, query_offset, key_offset=0, window=None):
        qp = torch.arange(query_offset, query_offset + len(q), device=q.device)
        kp = torch.arange(key_offset, key_offset + len(k), device=q.device)
        return ref.ref.attention(q, k, v, relative, qp, kp, window=window)

    # Change only the development oracle's KV storage and attention backend. The published source stays untouched.
    ref.pack_kv = lambda x: x
    attn.append_cache = append
    attn.attention = exact_attention
    reader = ref.Checkpoint(args.checkpoint)
    ids = json.loads(args.ids.read_text())
    ops = ref.Ops()
    state = ref.Residual(ref.embed(reader, ids))
    layer = ref.Layer(reader, 0, ops)
    layer.load_packed()
    cache = {"capacity": len(ids)}
    attention, x, picks, weights = layer.prologue(state, cache, fp8_cache=True)
    router = ref.ref.linear(x, layer.weights["mlp.gate.weight"], output_dtype=torch.float32)
    mlp = layer.convolve(layer.moe(x, picks, weights), "mlp_sconv", cache)
    state.finish(0, attention, mlp)
    logits = ref.logits(reader, state)[-1]
    torch.cuda.synchronize()
    report = {"status": "diagnostic", "cache": "bf16", "layers": 1, "rows": len(ids), "comparisons": {}}
    for name, want, dtype in (("partial", state.partial, torch.bfloat16),
                              ("attention", attention, torch.bfloat16), ("mlp", mlp, torch.bfloat16),
                              ("router", router, torch.float32), ("logits", logits, torch.float32)):
        suffix = "bf16" if dtype == torch.bfloat16 else "f32"
        raw = bytearray((args.native_dir / f"{name}.{suffix}").read_bytes())
        got = torch.frombuffer(raw, dtype=dtype).float()
        if name == "logits":
            got = got[:len(want)]
        got = got.reshape(want.shape)
        target = want.detach().float().cpu()
        diff = got - target
        comparison = {"elements": got.numel(), "different": (got != target).sum().item(),
                      "max_abs": diff.abs().max().item(),
                      "relative_l2": (diff.norm() / target.norm().clamp_min(1e-30)).item()}
        if name == "logits":
            comparison.update(native_top1=got.argmax().item(), reference_top1=target.argmax().item())
        report["comparisons"][name] = comparison
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
