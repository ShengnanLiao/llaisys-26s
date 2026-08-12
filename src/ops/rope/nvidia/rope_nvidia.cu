#include "rope_nvidia.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

constexpr int BLOCK_SIZE = 256;

inline void checkCuda(cudaError_t err, const char *expr) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error in ") +
            expr +
            ": " +
            cudaGetErrorString(err)
        );
    }
}

// 与 CPU 实现保持相同的精度行为：
// 指数部分用 double 计算 pow，结果舍入回 float，
// 再对 float 角度求 cos/sin。
__device__ inline void rope_angle(
    float p,
    float theta,
    size_t j,
    size_t dim,
    float *cos_out,
    float *sin_out
) {
    const float freq = static_cast<float>(
        static_cast<double>(p) /
        pow(
            static_cast<double>(theta),
            (2.0 * static_cast<double>(j)) / static_cast<double>(dim)
        )
    );

    *cos_out = cosf(freq);
    *sin_out = sinf(freq);
}

__global__ void rope_f32_kernel(
    float *out,
    const float *in,
    const int64_t *pos_ids,
    float theta,
    size_t nhead,
    size_t half,
    size_t dim,
    size_t numel
) {
    size_t i =
        static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    if (i < numel) {
        const size_t th = i / half; // s * nhead + h
        const size_t j = i % half;
        const size_t s = th / nhead;

        const size_t row_offset = th * dim;

        float cos_, sin_;
        rope_angle(
            static_cast<float>(pos_ids[s]),
            theta,
            j,
            dim,
            &cos_,
            &sin_);

        const float a = in[row_offset + j];
        const float b = in[row_offset + j + half];

        out[row_offset + j] = a * cos_ - b * sin_;
        out[row_offset + j + half] = a * sin_ + b * cos_;
    }
}

__global__ void rope_f16_kernel(
    __half *out,
    const __half *in,
    const int64_t *pos_ids,
    float theta,
    size_t nhead,
    size_t half,
    size_t dim,
    size_t numel
) {
    size_t i =
        static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    if (i < numel) {
        const size_t th = i / half;
        const size_t j = i % half;
        const size_t s = th / nhead;

        const size_t row_offset = th * dim;

        float cos_, sin_;
        rope_angle(
            static_cast<float>(pos_ids[s]),
            theta,
            j,
            dim,
            &cos_,
            &sin_);

        const float a = __half2float(in[row_offset + j]);
        const float b = __half2float(in[row_offset + j + half]);

        out[row_offset + j] = __float2half(a * cos_ - b * sin_);
        out[row_offset + j + half] = __float2half(a * sin_ + b * cos_);
    }
}

__global__ void rope_bf16_kernel(
    __nv_bfloat16 *out,
    const __nv_bfloat16 *in,
    const int64_t *pos_ids,
    float theta,
    size_t nhead,
    size_t half,
    size_t dim,
    size_t numel
) {
    size_t i =
        static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    if (i < numel) {
        const size_t th = i / half;
        const size_t j = i % half;
        const size_t s = th / nhead;

        const size_t row_offset = th * dim;

        float cos_, sin_;
        rope_angle(
            static_cast<float>(pos_ids[s]),
            theta,
            j,
            dim,
            &cos_,
            &sin_);

        const float a = __bfloat162float(in[row_offset + j]);
        const float b = __bfloat162float(in[row_offset + j + half]);

        out[row_offset + j] = __float2bfloat16(a * cos_ - b * sin_);
        out[row_offset + j + half] = __float2bfloat16(a * sin_ + b * cos_);
    }
}

} // anonymous namespace


namespace llaisys::ops::nvidia {

void rope(
    std::byte *out,
    const std::byte *in,
    const std::byte *pos_ids,
    llaisysDataType_t type,
    float theta,
    size_t seq_len,
    size_t nhead,
    size_t dim
) {
    if (seq_len == 0 || nhead == 0 || dim == 0) {
        return;
    }

    const size_t half = dim / 2;
    const size_t numel = seq_len * nhead * half;

    const int grid_size =
        static_cast<int>(
            (numel + BLOCK_SIZE - 1) /
            BLOCK_SIZE
        );

    switch (type) {

    case LLAISYS_DTYPE_F32:
        rope_f32_kernel<<<grid_size, BLOCK_SIZE>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const int64_t *>(pos_ids),
            theta,
            nhead,
            half,
            dim,
            numel
        );
        break;

    case LLAISYS_DTYPE_F16:
        rope_f16_kernel<<<grid_size, BLOCK_SIZE>>>(
            reinterpret_cast<__half *>(out),
            reinterpret_cast<const __half *>(in),
            reinterpret_cast<const int64_t *>(pos_ids),
            theta,
            nhead,
            half,
            dim,
            numel
        );
        break;

    case LLAISYS_DTYPE_BF16:
        rope_bf16_kernel<<<grid_size, BLOCK_SIZE>>>(
            reinterpret_cast<__nv_bfloat16 *>(out),
            reinterpret_cast<const __nv_bfloat16 *>(in),
            reinterpret_cast<const int64_t *>(pos_ids),
            theta,
            nhead,
            half,
            dim,
            numel
        );
        break;

    default:
        throw std::runtime_error(
            "NVIDIA rope: unsupported datatype"
        );
    }

    checkCuda(
        cudaGetLastError(),
        "rope kernel launch"
    );
}

} // namespace llaisys::ops::nvidia
