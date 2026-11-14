// Four raw-input taps. Dynamic global k/v use lag-major static weights plus a per-head input projection.
#pragma once
#include "inkling2_common.cuh"
using namespace inkling2;

extern "C" __global__ void tf_ik_conv(const bf16* x, const bf16* weight, const bf16* dynamic,
                                     const bf16* history, bf16* y, int rows, int width) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * width) return;
    const int row = i / width, col = i % width;
    float acc = 0.f;
    for (int tap = 0; tap < 4; ++tap) {
        const int lag = dynamic ? tap : 3 - tap, source = row - lag;
        const float v = source >= 0 ? f(x[(size_t)source * width + col])
                                   : f(history[(source + 3) * width + col]);
        float w = dynamic ? f(weight[tap * width + col]) : f(weight[col * 4 + tap]);
        if (dynamic) w += f(dynamic[(row * (width / 128) + col / 128) * 4 + lag]);
        acc = dynamic && !tap ? v * w : fmaf(v, w, acc);
    }
    y[i] = b((acc + f(x[i])) * (dynamic ? (1.f / 3.f) : 0.4472135954999579f));
}

// A separate launch keeps every old cache value alive until the convolution has consumed it.
extern "C" __global__ void tf_ik_conv_commit(const bf16* x, bf16* history, int rows, int width) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= width) return;
    bf16 next[3];
    for (int j = 0; j < 3; ++j) {
        const int src = rows - 3 + j;
        next[j] = src >= 0 ? x[src * width + c] : history[(src + 3) * width + c];
    }
    for (int j = 0; j < 3; ++j) history[j * width + c] = next[j];
}
