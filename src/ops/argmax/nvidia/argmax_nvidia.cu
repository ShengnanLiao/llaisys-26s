#include "argmax_nvidia.cuh"

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

__global__ void argmax_f32_kernel(
    int64_t *max_idx,
    float *max_val,
    const float *vals,
    size_t numel
) {
    __shared__ float s_val[BLOCK_SIZE];
    __shared__ int64_t s_idx[BLOCK_SIZE];

    // 每个线程在自己跨步扫描的范围内求局部最大值；
    // 严格 > 比较，等值时保留靠前的索引，与 CPU 实现一致。
    size_t i = threadIdx.x;
    float best_val = -INFINITY;
    size_t best_idx = 0;

    if (i < numel) {
        best_val = vals[i];
        best_idx = i;
    }

    for (i += BLOCK_SIZE; i < numel; i += BLOCK_SIZE) {
        float x = vals[i];
        if (x > best_val) {
            best_val = x;
            best_idx = i;
        }
    }

    s_val[threadIdx.x] = best_val;
    s_idx[threadIdx.x] = best_idx;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            int64_t other_idx = s_idx[threadIdx.x + stride];

            if (s_val[threadIdx.x + stride] > s_val[threadIdx.x]) {
                s_val[threadIdx.x] = s_val[threadIdx.x + stride];
                s_idx[threadIdx.x] = other_idx;
            } else if (s_val[threadIdx.x + stride] == s_val[threadIdx.x] &&
                       other_idx < s_idx[threadIdx.x]) {
                s_idx[threadIdx.x] = other_idx;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        max_idx[0] = s_idx[0];
        max_val[0] = s_val[0];
    }
}

__global__ void argmax_f16_kernel(
    int64_t *max_idx,
    __half *max_val,
    const __half *vals,
    size_t numel
) {
    __shared__ float s_val[BLOCK_SIZE];
    __shared__ int64_t s_idx[BLOCK_SIZE];

    size_t i = threadIdx.x;
    float best_val = -INFINITY;
    size_t best_idx = 0;

    if (i < numel) {
        best_val = __half2float(vals[i]);
        best_idx = i;
    }

    for (i += BLOCK_SIZE; i < numel; i += BLOCK_SIZE) {
        float x = __half2float(vals[i]);
        if (x > best_val) {
            best_val = x;
            best_idx = i;
        }
    }

    s_val[threadIdx.x] = best_val;
    s_idx[threadIdx.x] = best_idx;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            int64_t other_idx = s_idx[threadIdx.x + stride];

            if (s_val[threadIdx.x + stride] > s_val[threadIdx.x]) {
                s_val[threadIdx.x] = s_val[threadIdx.x + stride];
                s_idx[threadIdx.x] = other_idx;
            } else if (s_val[threadIdx.x + stride] == s_val[threadIdx.x] &&
                       other_idx < s_idx[threadIdx.x]) {
                s_idx[threadIdx.x] = other_idx;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        max_idx[0] = s_idx[0];
        max_val[0] = __float2half(s_val[0]);
    }
}

__global__ void argmax_bf16_kernel(
    int64_t *max_idx,
    __nv_bfloat16 *max_val,
    const __nv_bfloat16 *vals,
    size_t numel
) {
    __shared__ float s_val[BLOCK_SIZE];
    __shared__ int64_t s_idx[BLOCK_SIZE];

    size_t i = threadIdx.x;
    float best_val = -INFINITY;
    size_t best_idx = 0;

    if (i < numel) {
        best_val = __bfloat162float(vals[i]);
        best_idx = i;
    }

    for (i += BLOCK_SIZE; i < numel; i += BLOCK_SIZE) {
        float x = __bfloat162float(vals[i]);
        if (x > best_val) {
            best_val = x;
            best_idx = i;
        }
    }

    s_val[threadIdx.x] = best_val;
    s_idx[threadIdx.x] = best_idx;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            int64_t other_idx = s_idx[threadIdx.x + stride];

            if (s_val[threadIdx.x + stride] > s_val[threadIdx.x]) {
                s_val[threadIdx.x] = s_val[threadIdx.x + stride];
                s_idx[threadIdx.x] = other_idx;
            } else if (s_val[threadIdx.x + stride] == s_val[threadIdx.x] &&
                       other_idx < s_idx[threadIdx.x]) {
                s_idx[threadIdx.x] = other_idx;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        max_idx[0] = s_idx[0];
        max_val[0] = __float2bfloat16(s_val[0]);
    }
}

} // anonymous namespace


namespace llaisys::ops::nvidia {

void argmax(
    std::byte *max_idx,
    std::byte *max_val,
    const std::byte *vals,
    llaisysDataType_t type,
    size_t numel
) {
    if (numel == 0) {
        return;
    }

    switch (type) {

    case LLAISYS_DTYPE_F32:
        argmax_f32_kernel<<<1, BLOCK_SIZE>>>(
            reinterpret_cast<int64_t *>(max_idx),
            reinterpret_cast<float *>(max_val),
            reinterpret_cast<const float *>(vals),
            numel
        );
        break;

    case LLAISYS_DTYPE_F16:
        argmax_f16_kernel<<<1, BLOCK_SIZE>>>(
            reinterpret_cast<int64_t *>(max_idx),
            reinterpret_cast<__half *>(max_val),
            reinterpret_cast<const __half *>(vals),
            numel
        );
        break;

    case LLAISYS_DTYPE_BF16:
        argmax_bf16_kernel<<<1, BLOCK_SIZE>>>(
            reinterpret_cast<int64_t *>(max_idx),
            reinterpret_cast<__nv_bfloat16 *>(max_val),
            reinterpret_cast<const __nv_bfloat16 *>(vals),
            numel
        );
        break;

    default:
        throw std::runtime_error(
            "NVIDIA argmax: unsupported datatype"
        );
    }

    checkCuda(
        cudaGetLastError(),
        "argmax kernel launch"
    );
}

} // namespace llaisys::ops::nvidia
