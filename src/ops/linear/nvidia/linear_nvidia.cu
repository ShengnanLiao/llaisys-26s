#include "linear_nvidia.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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

__global__ void linear_f32_kernel(
    float *out,
    const float *in,
    const float *weight,
    const float *bias,
    size_t K,
    size_t O,
    size_t numel
) {
    size_t i =
        static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    if (i < numel) {
        const size_t m = i / O;
        const size_t o = i % O;

        const float *in_row = in + m * K;
        const float *weight_row = weight + o * K;

        float acc = 0.0f;
        if (bias != nullptr) {
            acc = bias[o];
        }

        // 保持与 CPU 实现相同的 k 累加顺序
        for (size_t k = 0; k < K; ++k) {
            acc += in_row[k] * weight_row[k];
        }

        out[i] = acc;
    }
}

__global__ void linear_f16_kernel(
    __half *out,
    const __half *in,
    const __half *weight,
    const __half *bias,
    size_t K,
    size_t O,
    size_t numel
) {
    size_t i =
        static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    if (i < numel) {
        const size_t m = i / O;
        const size_t o = i % O;

        const __half *in_row = in + m * K;
        const __half *weight_row = weight + o * K;

        float acc = 0.0f;
        if (bias != nullptr) {
            acc = __half2float(bias[o]);
        }

        for (size_t k = 0; k < K; ++k) {
            acc += __half2float(in_row[k]) *
                   __half2float(weight_row[k]);
        }

        out[i] = __float2half(acc);
    }
}

__global__ void linear_bf16_kernel(
    __nv_bfloat16 *out,
    const __nv_bfloat16 *in,
    const __nv_bfloat16 *weight,
    const __nv_bfloat16 *bias,
    size_t K,
    size_t O,
    size_t numel
) {
    size_t i =
        static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    if (i < numel) {
        const size_t m = i / O;
        const size_t o = i % O;

        const __nv_bfloat16 *in_row = in + m * K;
        const __nv_bfloat16 *weight_row = weight + o * K;

        float acc = 0.0f;
        if (bias != nullptr) {
            acc = __bfloat162float(bias[o]);
        }

        for (size_t k = 0; k < K; ++k) {
            acc += __bfloat162float(in_row[k]) *
                   __bfloat162float(weight_row[k]);
        }

        out[i] = __float2bfloat16(acc);
    }
}

} // anonymous namespace


namespace llaisys::ops::nvidia {

void linear(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    const std::byte *bias,
    llaisysDataType_t type,
    size_t M,
    size_t K,
    size_t O
) {
    if (M == 0 || K == 0 || O == 0) {
        return;
    }

    const size_t numel = M * O;

    const int grid_size =
        static_cast<int>(
            (numel + BLOCK_SIZE - 1) /
            BLOCK_SIZE
        );

    switch (type) {

    case LLAISYS_DTYPE_F32:
        linear_f32_kernel<<<grid_size, BLOCK_SIZE>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            reinterpret_cast<const float *>(bias),
            K,
            O,
            numel
        );
        break;

    case LLAISYS_DTYPE_F16:
        linear_f16_kernel<<<grid_size, BLOCK_SIZE>>>(
            reinterpret_cast<__half *>(out),
            reinterpret_cast<const __half *>(in),
            reinterpret_cast<const __half *>(weight),
            reinterpret_cast<const __half *>(bias),
            K,
            O,
            numel
        );
        break;

    case LLAISYS_DTYPE_BF16:
        linear_bf16_kernel<<<grid_size, BLOCK_SIZE>>>(
            reinterpret_cast<__nv_bfloat16 *>(out),
            reinterpret_cast<const __nv_bfloat16 *>(in),
            reinterpret_cast<const __nv_bfloat16 *>(weight),
            reinterpret_cast<const __nv_bfloat16 *>(bias),
            K,
            O,
            numel
        );
        break;

    default:
        throw std::runtime_error(
            "NVIDIA linear: unsupported datatype"
        );
    }

    checkCuda(
        cudaGetLastError(),
        "linear kernel launch"
    );
}

} // namespace llaisys::ops::nvidia
