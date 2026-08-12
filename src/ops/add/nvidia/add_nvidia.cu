#include "add_nvidia.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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

__global__ void add_f32_kernel(
    float *c,
    const float *a,
    const float *b,
    size_t numel
) {
    size_t i =
        static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    if (i < numel) {
        c[i] = a[i] + b[i];
    }
}

__global__ void add_f16_kernel(
    __half *c,
    const __half *a,
    const __half *b,
    size_t numel
) {
    size_t i =
        static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    if (i < numel) {
        float av = __half2float(a[i]);
        float bv = __half2float(b[i]);

        c[i] = __float2half(av + bv);
    }
}

__global__ void add_bf16_kernel(
    __nv_bfloat16 *c,
    const __nv_bfloat16 *a,
    const __nv_bfloat16 *b,
    size_t numel
) {
    size_t i =
        static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    if (i < numel) {
        float av = __bfloat162float(a[i]);
        float bv = __bfloat162float(b[i]);

        c[i] = __float2bfloat16(av + bv);
    }
}

} // anonymous namespace


namespace llaisys::ops::nvidia {

void add(
    std::byte *c,
    const std::byte *a,
    const std::byte *b,
    llaisysDataType_t type,
    size_t numel
) {
    if (numel == 0) {
        return;
    }

    constexpr int BLOCK_SIZE = 256;

    const int grid_size =
        static_cast<int>(
            (numel + BLOCK_SIZE - 1) /
            BLOCK_SIZE
        );

    switch (type) {

    case LLAISYS_DTYPE_F32:
        add_f32_kernel<<<grid_size, BLOCK_SIZE>>>(
            reinterpret_cast<float *>(c),
            reinterpret_cast<const float *>(a),
            reinterpret_cast<const float *>(b),
            numel
        );
        break;

    case LLAISYS_DTYPE_F16:
        add_f16_kernel<<<grid_size, BLOCK_SIZE>>>(
            reinterpret_cast<__half *>(c),
            reinterpret_cast<const __half *>(a),
            reinterpret_cast<const __half *>(b),
            numel
        );
        break;

    case LLAISYS_DTYPE_BF16:
        add_bf16_kernel<<<grid_size, BLOCK_SIZE>>>(
            reinterpret_cast<__nv_bfloat16 *>(c),
            reinterpret_cast<const __nv_bfloat16 *>(a),
            reinterpret_cast<const __nv_bfloat16 *>(b),
            numel
        );
        break;

    default:
        throw std::runtime_error(
            "NVIDIA add: unsupported datatype"
        );
    }

    checkCuda(
        cudaGetLastError(),
        "add kernel launch"
    );
}

} // namespace llaisys::ops::nvidia