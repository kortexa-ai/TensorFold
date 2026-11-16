// The published w13 retains interleaved gate/up in each Hadamard block. Only its epilogue splits them.
#pragma once
#include "inkling2_common.cuh"
#include "exl3_decode.cuh"
using namespace inkling2;

extern "C" __global__ void tf_ik_rotate(const half* x, const int* picks, const half* suh,
                                       half* out, int width, int input_slots, int experts) {
    const int pair = blockIdx.x, block = blockIdx.y, lane = threadIdx.x;
    const int expert = picks[pair];
    if (expert < 0 || expert >= experts) return;
    const int row = input_slots ? pair / input_slots : pair;
    const int c = block * 128 + lane * 4;
    float v[4];
    for (int j = 0; j < 4; ++j)
        v[j] = __half2float(__hmul(x[(size_t)row * width + c + j], suh[(size_t)expert * width + c + j]));
    tf_exl3::fwht128(v, lane);
    for (int j = 0; j < 4; ++j) out[(size_t)pair * width + c + j] =
        __float2half_rn(v[j] * tf_exl3::HAD_SCALE);
}

extern "C" __global__ void tf_ik_unrotate(const float* z, const int* picks, const half* svh,
                                         bf16* out, int pairs, int width, int splits, int experts) {
    const int pair = blockIdx.x, block = blockIdx.y, lane = threadIdx.x;
    const int expert = picks[pair], c = block * 128 + lane * 4;
    if (expert < 0 || expert >= experts) return;
    float v[4] = {};
    for (int s = 0; s < splits; ++s) for (int j = 0; j < 4; ++j)
        v[j] += z[((size_t)s * pairs + pair) * width + c + j];
    tf_exl3::fwht128(v, lane);
    for (int j = 0; j < 4; ++j) out[(size_t)pair * width + c + j] =
        b((v[j] * tf_exl3::HAD_SCALE) * __half2float(svh[(size_t)expert * width + c + j]));
}
