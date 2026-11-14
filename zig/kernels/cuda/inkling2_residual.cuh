// Inkling's depth mixtures normalize both keys and values. Partial+attention remains FP32.
#pragma once
#include "inkling2_common.cuh"
using namespace inkling2;

struct InklingMix {
    const bf16 *partial, *blocks, *attention;
    const float* query;
    const bf16* norm;
    bf16* out;
    int rows, width, block_stride, blocks_count, queries, mode;
};

__device__ inline float ik_source(const InklingMix a, int row, int s, int c) {
    if (s < a.blocks_count) return f(a.blocks[(size_t)s * a.block_stride + (size_t)row * a.width + c]);
    if (a.mode == 1 && s == a.blocks_count + 1) return f(a.attention[(size_t)row * a.width + c]);
    float v = f(a.partial[(size_t)row * a.width + c]);
    if (a.mode == 2) v += f(a.attention[(size_t)row * a.width + c]);
    return v;
}

extern "C" __global__ void tf_ik_mix(InklingMix a) {
    const int row = blockIdx.x, q = blockIdx.y, t = threadIdx.x;
    const int sources = a.blocks_count + 1 + (a.mode == 1);
    __shared__ float inv[10], prob[10];
    for (int s = 0; s < sources; ++s) {
        float square = 0.f, dot = 0.f;
        for (int c = t; c < a.width; c += blockDim.x) {
            const float v = ik_source(a, row, s, c);
            square += v * v;
            dot += v * (a.query[q * a.width + c] * rsqrtf((float)a.width));
        }
        square = sum(square);
        dot = sum(dot);
        if (!t) {
            inv[s] = rsqrtf(square / a.width + 1e-6f);
            prob[s] = dot * inv[s];
        }
        __syncthreads();
    }
    if (!t) {
        float m = -INFINITY, z = 0.f;
        for (int s = 0; s < sources; ++s) m = isnan(prob[s]) ? prob[s] : fmaxf(m, prob[s]);
        for (int s = 0; s < sources; ++s) z += (prob[s] = expf(prob[s] - m));
        for (int s = 0; s < sources; ++s) prob[s] /= z;
    }
    __syncthreads();
    float square = 0.f, values[18]; // exactly 4608 / 256 elements per thread
    for (int j = 0; j < 18; ++j) {
        const int c = t + j * 256;
        float v = 0.f;
        if (c < a.width) for (int s = 0; s < sources; ++s)
            v += prob[s] * (ik_source(a, row, s, c) * inv[s]);
        values[j] = v;
        square += v * v;
    }
    const float scale = rsqrtf(sum(square) / a.width + 1e-6f);
    for (int j = 0; j < 18; ++j) {
        const int c = t + j * 256;
        if (c < a.width) a.out[((size_t)q * a.rows + row) * a.width + c] =
            b(values[j] * scale * f(a.norm[q * a.width + c]));
    }
}

extern "C" __global__ void tf_ik_finish(bf16* partial, bf16* blocks, const bf16* attn,
                                       const bf16* mlp, int count, int boundary) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    if (boundary) {
        blocks[i] = partial[i];
        partial[i] = b(f(attn[i]) + f(mlp[i]));
    } else partial[i] = b((f(partial[i]) + f(attn[i])) + f(mlp[i]));
}

extern "C" __global__ void tf_ik_rank_sum(const float* parts, bf16* out, int count, int world) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float v = 0.f;
    for (int r = 0; r < world; ++r) v += parts[(size_t)r * count + i];
    out[i] = b(v);
}

extern "C" __global__ void tf_ik_logits(float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 30.f * tanhf(out[i] / 30.f);
}
