#include "swiglu_nvidia.cuh"

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

__global__ void swiglu_f32_kernel(
    float *out,
    const float *gate,
    const float *up,
    size_t numel
) {
    size_t i =
        static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    if (i < numel) {
        const float g = gate[i];
        const float u = up[i];

        const float silu = g / (1.0f + expf(-g));

        out[i] = u * silu;
    }
}

__global__ void swiglu_f16_kernel(
    __half *out,
    const __half *gate,
    const __half *up,
    size_t numel
) {
    size_t i =
        static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    if (i < numel) {
        const float g = __half2float(gate[i]);
        const float u = __half2float(up[i]);

        const float silu = g / (1.0f + expf(-g));

        out[i] = __float2half(u * silu);
    }
}

__global__ void swiglu_bf16_kernel(
    __nv_bfloat16 *out,
    const __nv_bfloat16 *gate,
    const __nv_bfloat16 *up,
    size_t numel
) {
    size_t i =
        static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    if (i < numel) {
        const float g = __bfloat162float(gate[i]);
        const float u = __bfloat162float(up[i]);

        const float silu = g / (1.0f + expf(-g));

        out[i] = __float2bfloat16(u * silu);
    }
}

} // anonymous namespace


namespace llaisys::ops::nvidia {

void swiglu(
    std::byte *out,
    const std::byte *gate,
    const std::byte *up,
    llaisysDataType_t type,
    size_t numel
) {
    if (numel == 0) {
        return;
    }

    const int grid_size =
        static_cast<int>(
            (numel + BLOCK_SIZE - 1) /
            BLOCK_SIZE
        );

    switch (type) {

    case LLAISYS_DTYPE_F32:
        swiglu_f32_kernel<<<grid_size, BLOCK_SIZE>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(gate),
            reinterpret_cast<const float *>(up),
            numel
        );
        break;

    case LLAISYS_DTYPE_F16:
        swiglu_f16_kernel<<<grid_size, BLOCK_SIZE>>>(
            reinterpret_cast<__half *>(out),
            reinterpret_cast<const __half *>(gate),
            reinterpret_cast<const __half *>(up),
            numel
        );
        break;

    case LLAISYS_DTYPE_BF16:
        swiglu_bf16_kernel<<<grid_size, BLOCK_SIZE>>>(
            reinterpret_cast<__nv_bfloat16 *>(out),
            reinterpret_cast<const __nv_bfloat16 *>(gate),
            reinterpret_cast<const __nv_bfloat16 *>(up),
            numel
        );
        break;

    default:
        throw std::runtime_error(
            "NVIDIA swiglu: unsupported datatype"
        );
    }

    checkCuda(
        cudaGetLastError(),
        "swiglu kernel launch"
    );
}

} // namespace llaisys::ops::nvidia
