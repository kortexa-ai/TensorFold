// Q/R clipping and long-context scaling; K clipping and V RMSNorm followed by an exact BF16 cache write.
#pragma once
#include "inkling2_common.cuh"
using namespace inkling2;

extern "C" __global__ void tf_ik_prep(const bf16* x, const bf16* weight, bf16* out, bf16* cache,
                                     int heads, int width, int pos, int capacity, int kind, int global) {
    const int row = blockIdx.y, head = blockIdx.x, c = threadIdx.x;
    const size_t at = ((size_t)row * heads + head) * width + c;
    float value = c < width ? f(x[at]) : 0.f;
    if (weight && c < width) value *= f(weight[head * width + c]);
    const float squared = sum(value * value);
    float factor;
    if (kind == 2) factor = rsqrtf(squared / width + 1e-6f);
    else {
        factor = (kind == 3 ? 8.f : 22.627416997969522f) / (sqrtf(squared) + 1e-6f);
        if (factor > 1.f) factor = 1.f;
    }
    value = f(b(value * factor));
    if (global && (kind == 0 || kind == 3)) {
        const float tau = 1.f + .1f * logf(fmaxf((pos + row + 1.f) / 128000.f, 1.f));
        value *= tau;
    }
    if (c < width) {
        out[at] = b(value);
        if (cache) cache[((size_t)((pos + row) % capacity) * heads + head) * width + c] = b(value);
    }
}

extern "C" __global__ void tf_ik_relative(const bf16* r, const bf16* proj, bf16* out, int heads) {
    const int key = blockIdx.x * blockDim.x + threadIdx.x, pair = blockIdx.y;
    if (key >= 1024) return;
    float value = 0.f;
    for (int d = 0; d < 16; ++d) value += f(r[pair * 16 + d]) * f(proj[d * 1024 + key]);
    out[pair * 1024 + key] = b(value);
}
