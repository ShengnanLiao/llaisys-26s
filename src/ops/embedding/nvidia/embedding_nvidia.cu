#include "embedding_nvidia.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

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

__global__ void embedding_f32_kernel(
    float *out,
    const int64_t *index,
    const float *weight,
    size_t numel,
    size_t embedding_dim
) {
    size_t i =
        static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    if (i < numel) {
        size_t row = i / embedding_dim;
        size_t col = i % embedding_dim;

        out[i] = weight[index[row] * embedding_dim + col];
    }
}

__global__ void embedding_f16_kernel(
    __half *out,
    const int64_t *index,
    const __half *weight,
    size_t numel,
    size_t embedding_dim
) {
    size_t i =
        static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    if (i < numel) {
        size_t row = i / embedding_dim;
        size_t col = i % embedding_dim;

        out[i] = weight[index[row] * embedding_dim + col];
    }
}

__global__ void embedding_bf16_kernel(
    __nv_bfloat16 *out,
    const int64_t *index,
    const __nv_bfloat16 *weight,
    size_t numel,
    size_t embedding_dim
) {
    size_t i =
        static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    if (i < numel) {
        size_t row = i / embedding_dim;
        size_t col = i % embedding_dim;

        out[i] = weight[index[row] * embedding_dim + col];
    }
}

} // anonymous namespace


namespace llaisys::ops::nvidia {

void embedding(
    std::byte *out,
    const std::byte *index,
    const std::byte *weight,
    llaisysDataType_t type,
    size_t num_indices,
    size_t embedding_dim
) {
    if (num_indices == 0 || embedding_dim == 0) {
        return;
    }

    constexpr int BLOCK_SIZE = 256;

    const size_t numel = num_indices * embedding_dim;

    const int grid_size =
        static_cast<int>(
            (numel + BLOCK_SIZE - 1) /
            BLOCK_SIZE
        );

    switch (type) {

    case LLAISYS_DTYPE_F32:
        embedding_f32_kernel<<<grid_size, BLOCK_SIZE>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const int64_t *>(index),
            reinterpret_cast<const float *>(weight),
            numel,
            embedding_dim
        );
        break;

    case LLAISYS_DTYPE_F16:
        embedding_f16_kernel<<<grid_size, BLOCK_SIZE>>>(
            reinterpret_cast<__half *>(out),
            reinterpret_cast<const int64_t *>(index),
            reinterpret_cast<const __half *>(weight),
            numel,
            embedding_dim
        );
        break;

    case LLAISYS_DTYPE_BF16:
        embedding_bf16_kernel<<<grid_size, BLOCK_SIZE>>>(
            reinterpret_cast<__nv_bfloat16 *>(out),
            reinterpret_cast<const int64_t *>(index),
            reinterpret_cast<const __nv_bfloat16 *>(weight),
            numel,
            embedding_dim
        );
        break;

    default:
        throw std::runtime_error(
            "NVIDIA embedding: unsupported datatype"
        );
    }

    checkCuda(
        cudaGetLastError(),
        "embedding kernel launch"
    );
}

} // namespace llaisys::ops::nvidia
