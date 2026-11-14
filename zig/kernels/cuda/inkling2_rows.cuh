// Embedding, the initial row norm and the FP32 vocabulary pick at the checkpoint's original precision.
#pragma once
#include "inkling2_common.cuh"
using namespace inkling2;

extern "C" __global__ void tf_ik_embed(const int* ids, const bf16* table, bf16* out, int rows, int width) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * width) out[i] = table[(size_t)ids[i / width] * width + i % width];
}

extern "C" __global__ void tf_ik_norm(const bf16* x, const bf16* weight, bf16* out, int width) {
    const int row = blockIdx.x, t = threadIdx.x;
    float square = 0.f;
    for (int c = t; c < width; c += blockDim.x) { const float v = f(x[(size_t)row * width + c]); square += v * v; }
    const float scale = rsqrtf(sum(square) / width + 1e-6f);
    for (int c = t; c < width; c += blockDim.x) out[(size_t)row * width + c] =
        b((f(x[(size_t)row * width + c]) * scale) * f(weight[c]));
}

struct InklingPick { float value; unsigned token, nonfinite, pad; };
extern "C" __global__ void tf_ik_top1(const float* x, InklingPick* out, int count, int first) {
    __shared__ float values[256];
    __shared__ int ids[256], bad[256];
    const int t = threadIdx.x;
    float value = -INFINITY;
    int id = 0x7fffffff, nonfinite = 0;
    for (int i = t; i < count; i += 256) {
        const float v = x[i];
        nonfinite |= !isfinite(v);
        if (v > value || (v == value && i < id)) { value = v; id = i; }
    }
    values[t] = value; ids[t] = id; bad[t] = nonfinite;
    __syncthreads();
    for (int s = 128; s; s >>= 1) {
        if (t < s) {
            const float a = values[t], b = values[t + s];
            if (b > a || (b == a && ids[t + s] < ids[t])) { values[t] = b; ids[t] = ids[t + s]; }
            bad[t] |= bad[t + s];
        }
        __syncthreads();
    }
    if (!t) *out = {values[0], (unsigned)(ids[0] + first), (unsigned)bad[0], 0};
}
