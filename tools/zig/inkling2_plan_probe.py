"""Regression for whole-expert EP skip buckets through both shared planner paths; development only."""
import argparse
import json
from pathlib import Path
import torch
from inkling2_probe import Driver

GUARD = 0x5A5A5A5A
SYMBOLS = {"small": "_ZN10tf_experts11plan_kernelEPKiiiiPiS2_S2_",
           "rank": "_ZN10tf_experts9plan_rankEPKiiiPiS2_",
           "offsets": "_ZN10tf_experts12plan_offsetsEiiiPiS0_S0_",
           "scatter": "_ZN10tf_experts12plan_scatterEPKiiiS1_S1_Pi"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fatbins", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if torch.cuda.device_count() != 1 or "RTX PRO 6000" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Pin this bounded development test to the authorized 6000")
    torch.cuda.set_per_process_memory_fraction(.1)
    if torch.cuda.mem_get_info()[0] < 8 * 1024**3:
        raise RuntimeError("Less than 8 GiB free")
    torch.manual_seed(20261114)
    torch.empty(1, device="cuda")
    route = Driver(a.fatbins / "inkling2.fatbin")
    plan = Driver(a.fatbins / "experts.fatbin")
    records = []
    for rows in (1, 17, 128, 129, 256):
        pairs, experts = rows * 8, 193
        ids = torch.stack([torch.randperm(384)[:8] for _ in range(rows)])
        logits = torch.full((rows, 384), -10., device="cuda")
        logits.scatter_(1, ids.cuda(), torch.arange(8., 0., -1, device="cuda").expand(rows, -1))
        bias = torch.zeros(384, device="cuda")
        for rank in (0, 1):
            backing = {}

            def guarded(name, n):
                data = torch.full((n + 128,), GUARD, device="cuda", dtype=torch.int32)
                backing[name] = data
                return data[64:-64]

            picks, gamma = guarded("picks", pairs), torch.empty((rows, 8), device="cuda")
            route.launch("tf_ik_route", (rows,), 512, logits, bias, picks, gamma, rank * 192, 192)
            expected = torch.where((ids >= rank * 192) & (ids < (rank + 1) * 192), ids - rank * 192, 192).flatten()
            # This assertion catches the former negative-ID bug before giving the planner unsafe input.
            assert torch.equal(picks.cpu(), expected.int())
            original = picks.clone()
            items = guarded("items", (min(pairs, experts) + pairs // 16) * 3)
            members, counts = guarded("members", pairs), guarded("counts", experts + 2)
            ranks = guarded("ranks", pairs)
            nb = (pairs + 1023) // 1024
            hist = guarded("hist", nb * experts)
            want_members, want_items, offset, used = [], [], 0, 0
            for expert in range(experts):
                positions = (expected == expert).nonzero().flatten().tolist()
                want_members.extend(positions)
                used += bool(positions)
                for first in range(0, len(positions), 16):
                    want_items.extend((expert, offset + first, min(16, len(positions) - first)))
                offset += len(positions)
            for repeat in range(5):
                for name, data in backing.items():
                    if name != "picks":
                        data[64:-64].fill_(GUARD)
                if pairs <= 1024:
                    plan.launch(SYMBOLS["small"], (1,), 1024, picks, pairs, experts, 16, members, items, counts)
                else:
                    plan.launch(SYMBOLS["rank"], (nb,), 1024, picks, pairs, experts, ranks, hist)
                    plan.launch(SYMBOLS["offsets"], (1,), 1024, nb, experts, 16, hist, items, counts)
                    plan.launch(SYMBOLS["scatter"], ((pairs + 255) // 256,), 256, picks, pairs, experts, ranks, hist, members)
                torch.cuda.synchronize()
                assert members.cpu().tolist() == want_members
                assert items[:len(want_items)].cpu().tolist() == want_items
                assert counts[:2].cpu().tolist() == [len(want_items) // 3, used]
                assert torch.equal(picks, original)
                for data in backing.values():
                    assert (data[:64] == GUARD).all() and (data[-64:] == GUARD).all()
            records.append({"rows": rows, "rank": rank, "path": "small" if pairs <= 1024 else "wide",
                            "repeats": 5, "guards": "unchanged", "members": pairs, "items": len(want_items) // 3})
    report = {"status": "passed", "cases": records, "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
