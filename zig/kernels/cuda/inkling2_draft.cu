// The bundled BF16 DSpark operators; the target graph and arithmetic remain separately owned.
#include "inkling2_common.cuh"
using namespace inkling2;

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
    const float angle = (pos + row) * powf(1000000.f, -2.f * (c % 64) / 128.f);
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

extern "C" __global__ void tf_ik_draft_gates(const bf16* projection, const bf16* bias, bf16* state, bf16* latent) {
    const int c = threadIdx.x;
    const float g = f(b(f(projection[c]) + f(bias[c])));
    const float candidate = f(b(f(projection[256 + c]) + f(bias[256 + c])));
    const float output = f(b(f(projection[512 + c]) + f(bias[512 + c])));
    const float gate = f(b(1.f / (1.f + expf(-g))));
    state[c] = b(f(b(gate * f(state[c]))) + f(b(f(b(1.f - gate)) * f(b(tanhf(candidate))))));
    latent[c] = b(tanhf(output));
}

extern "C" __global__ void tf_ik_draft_logits(const bf16* base, const bf16* correction,
                                            const bf16* bias, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float delta = f(b(f(correction[i]) + f(bias[i])));
        const float logit = f(b(f(base[i]) + delta));
        out[i] = 30.f * tanhf(logit / 30.f);
    }
}
