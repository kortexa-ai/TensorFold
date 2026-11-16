"""Prepare the bundled BF16 DSpark head's FP64-folded vocabulary correction on CPU."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--mtp', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    if a.output.exists():
        raise FileExistsError(a.output)
    with safe_open(a.mtp, framework='pt', device='cpu') as draft, \
            safe_open(a.model / 'dense/global.safetensors', framework='pt', device='cpu') as target:
        projection = draft.get_tensor('markov_head.hidden_out.weight').double()
        bias = draft.get_tensor('markov_head.hidden_out.bias').double()
        head = target.get_slice('model.llm.unembed.weight')
        n, k = head.get_shape()
        assert (n, k) == (201024, 4608) and projection.shape == (4608, 256)
        weight = torch.empty((n, 256), dtype=torch.bfloat16)
        folded_bias = torch.empty(n, dtype=torch.bfloat16)
        for start in range(0, n, 4096):
            end = min(n, start + 4096)
            rows = head[start:end].double()
            weight[start:end] = rows @ projection
            folded_bias[start:end] = rows @ bias
        alpha = draft.get_tensor('fuse_weight').float().softmax(dim=-1).bfloat16()
        save_file({'weight': weight, 'bias': folded_bias, 'alpha': alpha}, a.output)
    digest = hashlib.file_digest(a.output.open('rb'), 'sha256').hexdigest()
    print(json.dumps({'sha256': digest, 'bytes': a.output.stat().st_size, 'arithmetic': 'FP64 fold to BF16'}))


if __name__ == '__main__':
    main()
