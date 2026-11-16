// The bundled BF16 DSpark operators; the target graph and arithmetic remain separately owned.
#include "inkling2_common.cuh"
using namespace inkling2;

__device__ __constant__ float draft_frequency[64] = {
    0x1.0000000000000p+0f, 0x1.9c97580000000p-1f, 0x1.4c7bc00000000p-1f, 0x1.0beddc0000000p-1f,
    0x1.afd1360000000p-2f, 0x1.5bf9fe0000000p-2f, 0x1.186a080000000p-2f, 0x1.c3f06c0000000p-3f,
    0x1.6c310e0000000p-3f, 0x1.257b220000000p-3f, 0x1.d8ffaa0000000p-4f, 0x1.7d298e0000000p-4f,
    0x1.33281c0000000p-4f, 0x1.ef0a240000000p-5f, 0x1.8eec7e0000000p-5f, 0x1.41783a0000000p-5f,
    0x1.030dc40000000p-5f, 0x1.a183500000000p-6f, 0x1.5073160000000p-6f, 0x1.0f20100000000p-6f,
    0x1.b4f7e20000000p-7f, 0x1.6020a40000000p-7f, 0x1.1bc25a0000000p-7f, 0x1.c9548c0000000p-8f,
    0x1.7089380000000p-8f, 0x1.28fb5c0000000p-8f, 0x1.dea41a0000000p-9f, 0x1.81b58a0000000p-9f,
    0x1.36d21a0000000p-9f, 0x1.f4f1e20000000p-10f, 0x1.93aeb80000000p-10f, 0x1.454dec0000000p-10f,
    0x1.0624de0000000p-10f, 0x1.a67e4e0000000p-11f, 0x1.5476860000000p-11f, 0x1.125c040000000p-11f,
    0x1.ba2e4c0000000p-12f, 0x1.6453f60000000p-12f, 0x1.1f24e40000000p-12f, 0x1.cec9220000000p-13f,
    0x1.74eea60000000p-13f, 0x1.2c86460000000p-13f, 0x1.e459c60000000p-14f, 0x1.864f6a0000000p-14f,
    0x1.3a87480000000p-14f, 0x1.faebaa0000000p-15f, 0x1.987f7a0000000p-15f, 0x1.492f560000000p-15f,
    0x1.0945660000000p-15f, 0x1.ab88840000000p-16f, 0x1.58863a0000000p-16f, 0x1.15a1da0000000p-16f,
    0x1.bf749e0000000p-17f, 0x1.68941c0000000p-17f, 0x1.2291c40000000p-17f, 0x1.d44e620000000p-18f,
    0x1.7961820000000p-18f, 0x1.301c040000000p-18f, 0x1.ea20e00000000p-19f, 0x1.8af7560000000p-19f,
    0x1.3e47c80000000p-19f, 0x1.007bd80000000p-19f, 0x1.9d5ef20000000p-20f, 0x1.4d1c980000000p-20f
};

extern "C" __global__ void tf_ik_draft_norm(const bf16* x, const bf16* norm, bf16* out, int width) {
    const int row = blockIdx.x, t = threadIdx.x;
    float square = 0.f;
    for (int c = t; c < width; c += blockDim.x) { const float v = f(x[(size_t)row * width + c]); square += v * v; }
    const float scale = rsqrtf(sum(square) / width + 1e-6f);
    for (int c = t; c < width; c += blockDim.x)
        out[(size_t)row * width + c] = b(f(b(f(x[(size_t)row * width + c]) * scale)) * f(norm[c]));
}

extern "C" __global__ void tf_ik_draft_mix(const bf16* taps, const bf16* alpha, const bf16* norm,
                                         bf16* out, int stride) {
    const int row = blockIdx.x, t = threadIdx.x;
    float square = 0.f;
    for (int c = t; c < 4608; c += 256) {
        float v = 0.f;
        for (int i = 0; i < 8; ++i) v += f(taps[(size_t)i * stride + row * 4608 + c]) * f(alpha[i]);
        square += v * v;
    }
    const float scale = rsqrtf(sum(square) / 4608.f + 1e-6f);
    for (int c = t; c < 4608; c += 256) {
        float v = 0.f;
        for (int i = 0; i < 8; ++i) v += f(taps[(size_t)i * stride + row * 4608 + c]) * f(alpha[i]);
        out[row * 4608 + c] = b((v * scale) * f(norm[c]));
    }
}

extern "C" __global__ void tf_ik_draft_prep(const bf16* x, const bf16* norm, bf16* out, bf16* cache,
                                          int heads, int pos, int capacity) {
    const int row = blockIdx.y, head = blockIdx.x, c = threadIdx.x;
    const size_t at = ((size_t)row * heads + head) * 128;
    const float value = f(x[at + c]);
    const float inv = rsqrtf(sum(value * value) / 128.f + 1e-6f);
    __shared__ float normalized[128];
    normalized[c] = f(b(f(b(value * inv)) * f(norm[c])));
    __syncthreads();
    const int other = (c + 64) % 128;
    const float v = normalized[c];
    const float w = normalized[other];
    const float angle = (pos + row) * draft_frequency[c % 64];
    const bf16 result = b(v * cosf(angle) + (c < 64 ? -w : w) * sinf(angle));
    out[at + c] = result;
    if (cache) cache[((size_t)((pos + row) % capacity) * heads + head) * 128 + c] = result;
}

extern "C" __global__ void tf_ik_draft_store(const bf16* x, bf16* cache, int width, int pos, int capacity, int rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * width) cache[(size_t)((pos + i / width) % capacity) * width + i % width] = x[i];
}

// One warp owns a query and a head; fixed key partitions keep global history work distributed.
extern "C" __global__ void tf_ik_draft_attention(const bf16* q, const bf16* k, const bf16* v,
                                                float* parts, int pos, int hq, int hk, int capacity, int local) {
    const int row = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int head = blockIdx.x, split = blockIdx.y, keyhead = head / (hq / hk);
    const int first = local ? max(0, pos - 1024) : 0, end = pos + 8;
    const int chunk = (end - first + 31) / 32;
    const int low = min(end, first + split * chunk), high = min(end, low + chunk);
    float query[4], value[4] = {0.f, 0.f, 0.f, 0.f};
    for (int d = 0; d < 4; ++d) query[d] = f(q[((size_t)row * hq + head) * 128 + lane + d * 32]);
    float maximum = -INFINITY, denominator = 0.f;
    for (int key = low; key < high; ++key) {
        const size_t base = ((size_t)(key % capacity) * hk + keyhead) * 128 + lane;
        float dot = 0.f;
        for (int d = 0; d < 4; ++d) dot += query[d] * f(k[base + d * 32]);
        for (int delta = 16; delta; delta >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, delta);
        dot *= 0.08838834764831845f;
        const float next = fmaxf(maximum, dot), scale = expf(maximum - next), p = expf(dot - next);
        denominator = denominator * scale + p;
        for (int d = 0; d < 4; ++d) value[d] = value[d] * scale + p * f(v[base + d * 32]);
        maximum = next;
    }
    const size_t out = (((size_t)row * hq + head) * 32 + split) * 130;
    if (!lane) { parts[out] = maximum; parts[out + 1] = denominator; }
    for (int d = 0; d < 4; ++d) parts[out + 2 + lane + d * 32] = value[d];
}

extern "C" __global__ void tf_ik_draft_merge(const float* parts, bf16* out, int hq) {
    const int head = blockIdx.x, row = blockIdx.y, c = threadIdx.x;
    const size_t base = ((size_t)row * hq + head) * 32 * 130;
    float maximum = -INFINITY;
    for (int i = 0; i < 32; ++i) maximum = fmaxf(maximum, parts[base + i * 130]);
    float denominator = 0.f, value = 0.f;
    for (int i = 0; i < 32; ++i) {
        const float scale = expf(parts[base + i * 130] - maximum);
        denominator += parts[base + i * 130 + 1] * scale;
        value += parts[base + i * 130 + 2 + c] * scale;
    }
    out[((size_t)row * hq + head) * 128 + c] = b(value / denominator);
}

extern "C" __global__ void tf_ik_draft_add(bf16* residual, const bf16* value, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) residual[i] = b(f(residual[i]) + f(value[i]));
}

extern "C" __global__ void tf_ik_draft_swiglu(const bf16* gate, const bf16* up, bf16* out, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { const float x = f(gate[i]); out[i] = b(f(b(x / (1.f + expf(-x)))) * f(up[i])); }
}

extern "C" __global__ void tf_ik_draft_gates(const float* projection, const bf16* bias, bf16* state, bf16* latent) {
    const int c = threadIdx.x;
    const float g = f(b(projection[c] + f(bias[c])));
    const float candidate = f(b(projection[256 + c] + f(bias[256 + c])));
    const float output = f(b(projection[512 + c] + f(bias[512 + c])));
    const float gate = f(b(1.f / (1.f + expf(-g))));
    state[c] = b(f(b(gate * f(state[c]))) + f(b(f(b(1.f - gate)) * f(b(tanhf(candidate))))));
    latent[c] = b(tanhf(output));
}

extern "C" __global__ void tf_ik_draft_logits(const bf16* base, const float* correction,
                                            const bf16* bias, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float delta = f(b(correction[i] + f(bias[i])));
        const float logit = f(b(f(base[i]) + delta));
        out[i] = 30.f * tanhf(logit / 30.f);
    }
}
