#include "rms_norm_nvidia.cuh"

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

__global__ void rms_norm_f32_kernel(
    float *out,
    const float *in,
    const float *weight,
    float eps,
    size_t dim
) {
    __shared__ float s_sum[BLOCK_SIZE];
    __shared__ float s_rms;

    const size_t r = blockIdx.x;
    const float *in_row = in + r * dim;
    float *out_row = out + r * dim;

    // 阶段 1：每个线程累加行内跨步元素的平方和
    float sum_sq = 0.0f;
    for (size_t k = threadIdx.x; k < dim; k += BLOCK_SIZE) {
        float v = in_row[k];
        sum_sq += v * v;
    }

    s_sum[threadIdx.x] = sum_sq;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        s_rms = sqrtf(s_sum[0] / static_cast<float>(dim) + eps);
    }
    __syncthreads();

    // 阶段 2：按行归一化并与 weight 逐元素相乘
    const float rms = s_rms;
    for (size_t k = threadIdx.x; k < dim; k += BLOCK_SIZE) {
        out_row[k] = (in_row[k] / rms) * weight[k];
    }
}

__global__ void rms_norm_f16_kernel(
    __half *out,
    const __half *in,
    const __half *weight,
    float eps,
    size_t dim
) {
    __shared__ float s_sum[BLOCK_SIZE];
    __shared__ float s_rms;

    const size_t r = blockIdx.x;
    const __half *in_row = in + r * dim;
    __half *out_row = out + r * dim;

    float sum_sq = 0.0f;
    for (size_t k = threadIdx.x; k < dim; k += BLOCK_SIZE) {
        float v = __half2float(in_row[k]);
        sum_sq += v * v;
    }

    s_sum[threadIdx.x] = sum_sq;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        s_rms = sqrtf(s_sum[0] / static_cast<float>(dim) + eps);
    }
    __syncthreads();

    const float rms = s_rms;
    for (size_t k = threadIdx.x; k < dim; k += BLOCK_SIZE) {
        float v = __half2float(in_row[k]);
        float w = __half2float(weight[k]);
        out_row[k] = __float2half((v / rms) * w);
    }
}

__global__ void rms_norm_bf16_kernel(
    __nv_bfloat16 *out,
    const __nv_bfloat16 *in,
    const __nv_bfloat16 *weight,
    float eps,
    size_t dim
) {
    __shared__ float s_sum[BLOCK_SIZE];
    __shared__ float s_rms;

    const size_t r = blockIdx.x;
    const __nv_bfloat16 *in_row = in + r * dim;
    __nv_bfloat16 *out_row = out + r * dim;

    float sum_sq = 0.0f;
    for (size_t k = threadIdx.x; k < dim; k += BLOCK_SIZE) {
        float v = __bfloat162float(in_row[k]);
        sum_sq += v * v;
    }

    s_sum[threadIdx.x] = sum_sq;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        s_rms = sqrtf(s_sum[0] / static_cast<float>(dim) + eps);
    }
    __syncthreads();

    const float rms = s_rms;
    for (size_t k = threadIdx.x; k < dim; k += BLOCK_SIZE) {
        float v = __bfloat162float(in_row[k]);
        float w = __bfloat162float(weight[k]);
        out_row[k] = __float2bfloat16((v / rms) * w);
    }
}

} // anonymous namespace


namespace llaisys::ops::nvidia {

void rms_norm(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    llaisysDataType_t type,
    float eps,
    size_t rows,
    size_t dim
) {
    if (rows == 0 || dim == 0) {
        return;
    }

    const int grid_size = static_cast<int>(rows);

    switch (type) {

    case LLAISYS_DTYPE_F32:
        rms_norm_f32_kernel<<<grid_size, BLOCK_SIZE>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            eps,
            dim
        );
        break;

    case LLAISYS_DTYPE_F16:
        rms_norm_f16_kernel<<<grid_size, BLOCK_SIZE>>>(
            reinterpret_cast<__half *>(out),
            reinterpret_cast<const __half *>(in),
            reinterpret_cast<const __half *>(weight),
            eps,
            dim
        );
        break;

    case LLAISYS_DTYPE_BF16:
        rms_norm_bf16_kernel<<<grid_size, BLOCK_SIZE>>>(
            reinterpret_cast<__nv_bfloat16 *>(out),
            reinterpret_cast<const __nv_bfloat16 *>(in),
            reinterpret_cast<const __nv_bfloat16 *>(weight),
            eps,
            dim
        );
        break;

    default:
        throw std::runtime_error(
            "NVIDIA rms_norm: unsupported datatype"
        );
    }

    checkCuda(
        cudaGetLastError(),
        "rms_norm kernel launch"
    );
}

} // namespace llaisys::ops::nvidia
