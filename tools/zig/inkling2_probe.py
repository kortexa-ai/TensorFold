"""Development oracle checks for native Inkling CUDA operators; never used for serving."""
import argparse
import ctypes as C
import importlib
import json
import sys
from pathlib import Path

import torch


class Driver:
    def __init__(self, fatbin):
        self.lib = C.CDLL("libcuda.so.1")
        self.lib.cuModuleLoad.argtypes = [C.POINTER(C.c_void_p), C.c_char_p]
        self.lib.cuModuleGetFunction.argtypes = [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p]
        self.lib.cuLaunchKernel.argtypes = [C.c_void_p, *([C.c_uint] * 7), C.c_void_p,
                                          C.POINTER(C.c_void_p), C.c_void_p]
        self.module = C.c_void_p()
        self.check(self.lib.cuModuleLoad(C.byref(self.module), str(Path(fatbin).resolve()).encode()))
        self.functions = {}

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f"CUDA driver error {code}")

    def launch(self, name, grid, threads, *args):
        if name not in self.functions:
            f = C.c_void_p()
            self.check(self.lib.cuModuleGetFunction(C.byref(f), self.module, name.encode()))
            self.functions[name] = f
        values = [C.c_void_p(a.data_ptr()) if isinstance(a, torch.Tensor)
                  else C.c_int(a) if isinstance(a, int) else a for a in args]
        pointers = (C.c_void_p * len(values))(*(C.cast(C.byref(v), C.c_void_p) for v in values))
        grid = (*grid, 1, 1)[:3]
        self.check(self.lib.cuLaunchKernel(self.functions[name], *grid, threads, 1, 1, 0,
                                          torch.cuda.current_stream().cuda_stream, pointers, None))


def exact(got, want, label, records):
    torch.cuda.synchronize()
    count = (got != want).sum().item()
    error = (got.float() - want.float()).abs().max().item()
    records.append({"case": label, "elements": got.numel(), "different": count, "max_abs": error})
    if count:
        raise AssertionError(f"{label}: {count}/{got.numel()} differ, max abs {error}")


def check_quant(d, ref, records):
    ops = ref.Ops()
    for width in (2304, 4608):
        for rows in (1, 2, 8, 17, 32):
            x = (torch.randn((rows, width), device="cuda") * 64).bfloat16()
            x[0, :32] = 0
            for nv in (0, 1):
                if nv:
                    packed, scales, amax = ops.quant(x)
                    inner = torch.empty_like(x)
                    ref.kernels.decode(packed, scales, torch.ones(1, device="cuda"), inner)
                    want = (inner.float() * (amax[:, None] * (1 / 1536))).half()
                else:
                    want = ref.mxfp8_dequant(*ops.mxfp8(x)).half()
                got = torch.empty_like(x, dtype=torch.float16)
                d.launch("tf_ik_quant", (rows,), 256, x, got, width, nv)
                exact(got, want, f"quant/{'nvfp4' if nv else 'mxfp8'}/{rows}x{width}", records)


def check_swiglu(d, ref, records):
    ops = ref.Ops()
    for rows in (1, 8, 17, 32):
        x = (torch.randn((rows, 4608), device="cuda") * 30).bfloat16()
        gamma = torch.rand(rows, device="cuda") * 2
        want = ops.swiglu(x, gamma, 4, 25)
        got = torch.empty_like(want)
        d.launch("tf_ik_swiglu", ((got.numel() + 255) // 256,), 256,
                 x, gamma, got, got.numel(), 2304)
        exact(got, want, f"swiglu/{rows}", records)


def check_route(d, ref, records):
    ops = ref.Ops()
    for rows in (1, 2, 8, 17, 32):
        logits = torch.randn((rows, 384), device="cuda") * 15
        bias = torch.randn(384, device="cuda") * .1
        logits[0] = 0  # exercise ties: lower expert ID wins
        bias[:16] = bias[0]
        _, ids, gamma = ops.route(logits, bias)
        picks = torch.empty_like(ids, dtype=torch.int32)
        weights = torch.empty_like(gamma)
        d.launch("tf_ik_route", (rows,), 512, logits, bias, picks, weights, 0, 384)
        exact(picks, ids, f"route/ids/{rows}", records)
        # libdevice and CUDA exp/log use the same precision; report their weight error explicitly.
        error = (weights - gamma).abs().max().item()
        records.append({"case": f"route/weights/{rows}", "max_abs": error})
        torch.testing.assert_close(weights, gamma, atol=0, rtol=0)


class Attention(C.Structure):
    _fields_ = [(name, C.c_void_p) for name in ("q", "k", "v", "relative", "gate", "sink", "out")] + [
        (name, C.c_int) for name in ("rows", "pos", "hq", "hk", "capacity", "window")]


def check_conv(d, ref, records):
    ops = ref.Ops()
    for width in (512, 4608):
        for rows in (1, 2, 8, 17, 32):
            for dynamic in (False, True):
                x = torch.randn((rows, width), device="cuda").bfloat16()
                history = torch.randn((3, width), device="cuda").bfloat16()
                if dynamic:
                    w = torch.randn((4, width), device="cuda").bfloat16()
                    dyn = torch.randn((rows, width // 128, 4), device="cuda").bfloat16()
                    want, state = ops.conv_dynamic(x, w, dyn, history)
                else:
                    w = torch.randn((width, 4), device="cuda").bfloat16()
                    dyn = C.c_void_p(0)
                    want, state = ops.conv(x, w, history)
                got = torch.empty_like(x)
                d.launch("tf_ik_conv", ((x.numel() + 255) // 256,), 256,
                         x, w, dyn, history, got, rows, width)
                exact(got, want, f"conv/{dynamic}/{rows}x{width}", records)
                d.launch("tf_ik_conv_commit", ((width + 255) // 256,), 256, x, history, rows, width)
                exact(history, state, f"conv/state/{dynamic}/{rows}x{width}", records)


def check_attention(d, ref, records):
    # Independent FP64 Q.K, softmax and P.V, with the public reference's BF16 boundaries.
    for rows, pos, window in ((1, 0, 0), (2, 31, 0), (17, 77, 0), (32, 1010, 1023), (8, 2049, 1023)):
        hq, hk, width = 6, 1, 128
        capacity = pos + rows if not window else 1024 + 32
        q = (torch.randn((rows, hq, width), device="cuda") * .5).bfloat16()
        all_k = (torch.randn((pos + rows, hk, width), device="cuda") * .5).bfloat16()
        all_v = torch.randn_like(all_k)
        k, v = (torch.empty((capacity, hk, width), device="cuda", dtype=torch.bfloat16) for _ in range(2))
        for key in range(max(0, pos - window) if window else 0, pos + rows):
            k[key % capacity] = all_k[key]
            v[key % capacity] = all_v[key]
        relative = (torch.randn((rows, hq, 1024), device="cuda") * .2).bfloat16()
        gate = torch.randn_like(q)
        sink = torch.randn((rows, hq), device="cuda").bfloat16()
        got = torch.empty_like(q)
        fields = [x.data_ptr() for x in (q, k, v, relative, gate, sink, got)]
        a = Attention(*fields, rows, pos, hq, hk, capacity, window)
        d.launch("tf_ik_attention", ((rows + 15) // 16, hq), 128, a)
        qp = torch.arange(pos, pos + rows, device="cuda")
        kp = torch.arange(pos + rows, device="cuda")
        dist = qp[:, None] - kp[None, :]
        scores = torch.einsum("rhd,khd->hrk", q.double(), all_k.double().repeat_interleave(6, 1)) * width**-.5
        bias = relative.double().permute(1, 0, 2)[:, torch.arange(rows, device="cuda")[:, None], dist.clamp(0, 1023)]
        scores += torch.where(((dist >= 0) & (dist < 1024))[None], bias, 0)
        mask = dist < 0
        if window:
            mask |= dist > window
        scores.masked_fill_(mask[None], -torch.inf)
        raw = torch.einsum("hrk,khd->rhd", scores.softmax(-1), all_v.double().repeat_interleave(6, 1))
        want = (raw * gate.double().sigmoid() * (scores.logsumexp(-1).T - sink.double()).sigmoid()[..., None])
        error = (got.double() - want).abs()
        # FP32 sums and exp/log/gates may cross a final BF16 midpoint; compare to its FP64 rounding envelope.
        rounding = want.abs() * (2**-8) + 2e-6
        records.append({"case": f"attention/{rows}/{pos}/{window}",
                        "max_abs": error.max().item(), "outside_envelope": (error > rounding).sum().item()})
        if (error > rounding).any():
            raise AssertionError(records[-1])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fatbin", required=True)
    p.add_argument("--reference-dir", type=Path, required=True,
                   help="Pinned standalone Inkling reference checkout, with its audited source files fetched")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--only", choices=("quant", "swiglu", "route", "conv", "attention"))
    args = p.parse_args()
    if torch.cuda.device_count() != 1 or "RTX PRO 6000" not in torch.cuda.get_device_name(0):
        raise RuntimeError("The bounded operator probe requires exactly the authorized RTX PRO 6000")
    torch.cuda.set_per_process_memory_fraction(.20)
    if torch.cuda.mem_get_info()[0] < 8 * 1024**3:
        raise RuntimeError("Less than 8 GiB free before bounded operator probe")
    torch.manual_seed(20261114)
    sys.path.insert(0, str(args.reference_dir.resolve()))
    ref = importlib.import_module("model")
    torch.empty(1, device="cuda")
    d, records = Driver(args.fatbin), []
    report = {"status": "running", "gpu": torch.cuda.get_device_name(0), "cases": records}
    try:
        for name, fn in (("quant", check_quant), ("swiglu", check_swiglu), ("route", check_route), ("conv", check_conv), ("attention", check_attention)):
            if args.only is None or args.only == name:
                fn(d, ref, records)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))


if __name__ == "__main__":
    main()
