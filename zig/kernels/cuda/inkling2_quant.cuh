// Checkpoint-defined activation round trips. The grouped EXL3 dot consumes FP16, not a new quantization.
#pragma once
#include "inkling2_common.cuh"
using namespace inkling2;

__device__ inline float ik_e4(float x) { return (float)__nv_fp8_e4m3(x); }
__device__ inline float ik_e2(float x) { return (float)__nv_fp4_e2m1(x); }

// A whole row's amax precedes the independent 16-value NVFP4 groups. One CTA owns the row.
extern "C" __global__ void tf_ik_quant(const bf16* x, half* y, int width, int nvfp4) {
    const int row = blockIdx.x, t = threadIdx.x;
    x += (size_t)row * width;
    y += (size_t)row * width;
    if (!nvfp4) {
        for (int g = t; g < width / 32; g += blockDim.x) {
            float amax = 0.f;
            for (int j = 0; j < 32; ++j) {
                const float v = fabsf(f(x[g * 32 + j]));
                amax = isnan(v) ? v : fmaxf(amax, v);
            }
            const float raw = amax * (1.f / 448.f), tiny = 0x1p-127f;
            int exponent = (__float_as_uint(raw) + 0x007fffffu) >> 23;
            exponent = raw <= tiny ? 0 : min(exponent, 254);
            if (isnan(raw)) exponent = 255;
            const float scale = exponent == 255 ? nanf("") : fmaxf(__uint_as_float(exponent << 23), tiny);
            const float inv = exponent == 255 ? 0.f : exponent == 0 ? 0x1p127f : 1.f / scale;
            for (int j = 0; j < 32; ++j) {
                float v = f(x[g * 32 + j]) * inv;
                v = isnan(v) ? v : fminf(448.f, fmaxf(-448.f, v));
                y[g * 32 + j] = __float2half_rn(f(b(ik_e4(v) * scale)));
            }
        }
        return;
    }
    float amax = 0.f;
    for (int c = t; c < width; c += blockDim.x) {
        const float v = fabsf(f(x[c]));
        amax = isnan(v) ? v : fmaxf(amax, v);
    }
    amax = maximum(amax);
    const float encoder = fminf(1536.f / amax, 0x1.fffffep127f);
    const float decoder = 1.f / encoder, outer = amax * (1.f / 1536.f);
    for (int g = t; g < width / 16; g += blockDim.x) {
        float values[16], block_max = 0.f;
        for (int j = 0; j < 16; ++j) {
            values[j] = f(x[g * 16 + j]);
            block_max = isnan(values[j]) ? nanf("") : fmaxf(block_max, fabsf(values[j]));
        }
        const float s6 = ik_e4(fminf(block_max * (encoder * (1.f / 6.f)), 448.f));
        const float s4 = ik_e4(fminf(block_max * (encoder * 0.25f), 448.f));
        const float inv6 = fminf(1.f / (s6 * decoder), 0x1.fffffep127f);
        const float inv4 = fminf(1.f / (s4 * decoder), 0x1.fffffep127f);
        float e6[16], e4[16];
        for (int j = 0; j < 16; ++j) {
            e6[j] = fabsf((ik_e2(values[j] * inv6) * s6) * outer - values[j]);
            e4[j] = fabsf((ik_e2(values[j] * inv4) * s4) * outer - values[j]);
        }
        // The checkpoint sums adjacent pairs, then pairs of pairs. A different order can pick another scale.
        for (int n = 8; n; n >>= 1) for (int j = 0; j < n; ++j) {
            e6[j] = e6[2 * j] + e6[2 * j + 1];
            e4[j] = e4[2 * j] + e4[2 * j + 1];
        }
        const bool four = e4[0] < e6[0];
        const float scale = four ? s4 : s6, inv = four ? inv4 : inv6;
        for (int j = 0; j < 16; ++j)
            y[g * 16 + j] = __float2half_rn((ik_e2(values[j] * inv) * scale) * outer);
    }
}

// The projection's interleaved gate/up is rounded to BF16 before Inkling's smooth caps and route scale.
extern "C" __global__ void tf_ik_swiglu(const bf16* gu, const float* gamma, bf16* y, int count, int width) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float g = f(gu[2 * i]), u = f(gu[2 * i + 1]), sg, capped_g, capped_u;
    asm("{ .reg .f32 t; mul.rn.f32 t, %1, 0fBFB8AA3B; ex2.approx.ftz.f32 t, t; "
        "add.rn.f32 t, t, 0f3F800000; rcp.approx.ftz.f32 %0, t; }" : "=f"(sg) : "f"(g));
    const float g4 = g / 4.f, u25 = u / 25.f;
    asm("tanh.approx.f32 %0, %1;" : "=f"(capped_g) : "f"(g4));
    asm("tanh.approx.f32 %0, %1;" : "=f"(capped_u) : "f"(u25));
    y[i] = b((((4.f * capped_g) * sg) * (25.f * capped_u)) * gamma[i / width]);
}
