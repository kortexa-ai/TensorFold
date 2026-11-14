// Sigmoid+bias selects experts; unbiassed log-sigmoid scores determine top-8 route weights.
#pragma once
#include "inkling2_common.cuh"
using namespace inkling2;

extern "C" __global__ void tf_ik_route(const float* logits, const float* bias,
                                      int* picks, float* gamma, int first, int owned) {
    const int row = blockIdx.x, t = threadIdx.x;
    __shared__ float scores[512], tmp[512];
    __shared__ int ids[512], chosen[8];
    float selection = t < 384 ? sigmoid(logits[row * 384 + t]) + bias[t] : -INFINITY;
    scores[t] = selection;
    __syncthreads();
    for (int slot = 0; slot < 8; ++slot) {
        tmp[t] = scores[t];
        ids[t] = t;
        __syncthreads();
        for (int stride = 256; stride; stride >>= 1) {
            if (t < stride) {
                const float a = tmp[t], b = tmp[t + stride];
                const bool take = isnan(b) ? !isnan(a) || ids[t + stride] < ids[t]
                    : !isnan(a) && (b > a || (b == a && ids[t + stride] < ids[t]));
                if (take) { tmp[t] = b; ids[t] = ids[t + stride]; }
            }
            __syncthreads();
        }
        if (!t) chosen[slot] = ids[0];
        __syncthreads();
        if (t == chosen[slot]) scores[t] = -INFINITY;
        __syncthreads();
    }
    if (!t) {
        float logs[8], values[8], maximum = -INFINITY;
        for (int s = 0; s < 8; ++s) {
            const float x = logits[row * 384 + chosen[s]];
            logs[s] = fminf(x, 0.f) - log1pf(expf(-fabsf(x)));
            maximum = fmaxf(maximum, logs[s]);
        }
        for (int s = 0; s < 8; ++s) values[s] = expf(logs[s] - maximum);
        const float total = ((values[0] + values[2]) + (values[1] + values[3]))
                          + ((values[4] + values[6]) + (values[5] + values[7]));
        for (int s = 0; s < 8; ++s) {
            const int id = chosen[s];
            picks[row * 8 + s] = id >= first && id < first + owned ? id - first : -1;
            gamma[row * 8 + s] = expf(logs[s] - (maximum + logf(total))) * 8.f;
        }
    }
}

// Each rank contributes complete expert outputs. Round each expert before summing in original top-k order.
extern "C" __global__ void tf_ik_moe_sum(const bf16* parts, bf16* out, int rows, int width, int world) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * width) return;
    const int row = i / width, col = i % width;
    float total = 0.f;
    for (int s = 0; s < 8; ++s) {
        float value = 0.f;
        for (int rank = 0; rank < world; ++rank)
            value += f(parts[((size_t)rank * rows * 8 + row * 8 + s) * width + col]);
        total += value;
    }
    out[i] = b(total);
}
