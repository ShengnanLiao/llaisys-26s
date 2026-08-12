#include "self_attention_nvidia.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <cmath>

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

// 每个 (query token i, query head h) 一个 block。
//
// 阶段 1: 计算 scores = (QK^T) * scale，归约出 max_score；
// 阶段 2: 计算 softmax 分子 exp(score - max) 存入共享内存，
//         归约出 sum_exp；
// 阶段 3: 每线程负责若干 c，按 j 升序累加 (w_j / sum_exp) * V[j][c]。
//
// 各累加顺序与 CPU 实现一致：dot 按 t 升序、输出按 j 升序。
__global__ void self_attention_f32_kernel(
    float *attn_val,
    const float *q,
    const float *k,
    const float *v,
    float scale,
    size_t seq_len,
    size_t total_len,
    size_t nhead,
    size_t nkvhead,
    size_t d,
    size_t dv
) {
    extern __shared__ float s_weights[]; // [total_len]
    __shared__ float s_partial[BLOCK_SIZE];
    __shared__ float s_max_score;
    __shared__ float s_sum_exp;

    const size_t token = blockIdx.x / nhead;
    const size_t h = blockIdx.x % nhead;
    const size_t kv_head = h * nkvhead / nhead;

    const size_t past_len = total_len - seq_len;
    const size_t eff_kv_len = past_len + token + 1;

    const float *q_vec = q + (token * nhead + h) * d;
    const float *k_head = k + kv_head * d;
    const float *v_head = v + kv_head * dv;

    // 阶段 1
    float max_score = -INFINITY;
    for (size_t j = threadIdx.x; j < eff_kv_len; j += BLOCK_SIZE) {
        const float *k_vec = k_head + j * nkvhead * d;

        float score = 0.0f;
        for (size_t t = 0; t < d; ++t) {
            score += q_vec[t] * k_vec[t];
        }
        score *= scale;

        if (score > max_score) {
            max_score = score;
        }
    }

    s_partial[threadIdx.x] = max_score;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_partial[threadIdx.x] =
                fmaxf(s_partial[threadIdx.x],
                      s_partial[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        s_max_score = s_partial[0];
    }
    __syncthreads();

    // 阶段 2
    float sum_exp = 0.0f;
    for (size_t j = threadIdx.x; j < eff_kv_len; j += BLOCK_SIZE) {
        const float *k_vec = k_head + j * nkvhead * d;

        float score = 0.0f;
        for (size_t t = 0; t < d; ++t) {
            score += q_vec[t] * k_vec[t];
        }
        score *= scale;

        const float w = expf(score - s_max_score);
        s_weights[j] = w;
        sum_exp += w;
    }

    s_partial[threadIdx.x] = sum_exp;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_partial[threadIdx.x] +=
                s_partial[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        s_sum_exp = s_partial[0];
    }
    __syncthreads();

    // 阶段 3
    for (size_t c = threadIdx.x; c < dv; c += BLOCK_SIZE) {
        float acc = 0.0f;
        for (size_t j = 0; j < eff_kv_len; ++j) {
            const float *v_vec = v_head + j * nkvhead * dv;
            acc += (s_weights[j] / s_sum_exp) * v_vec[c];
        }
        attn_val[(token * nhead + h) * dv + c] = acc;
    }
}

__global__ void self_attention_f16_kernel(
    __half *attn_val,
    const __half *q,
    const __half *k,
    const __half *v,
    float scale,
    size_t seq_len,
    size_t total_len,
    size_t nhead,
    size_t nkvhead,
    size_t d,
    size_t dv
) {
    extern __shared__ float s_weights[];
    __shared__ float s_partial[BLOCK_SIZE];
    __shared__ float s_max_score;
    __shared__ float s_sum_exp;

    const size_t token = blockIdx.x / nhead;
    const size_t h = blockIdx.x % nhead;
    const size_t kv_head = h * nkvhead / nhead;

    const size_t past_len = total_len - seq_len;
    const size_t eff_kv_len = past_len + token + 1;

    const __half *q_vec = q + (token * nhead + h) * d;
    const __half *k_head = k + kv_head * d;
    const __half *v_head = v + kv_head * dv;

    float max_score = -INFINITY;
    for (size_t j = threadIdx.x; j < eff_kv_len; j += BLOCK_SIZE) {
        const __half *k_vec = k_head + j * nkvhead * d;

        float score = 0.0f;
        for (size_t t = 0; t < d; ++t) {
            score += __half2float(q_vec[t]) *
                     __half2float(k_vec[t]);
        }
        score *= scale;

        if (score > max_score) {
            max_score = score;
        }
    }

    s_partial[threadIdx.x] = max_score;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_partial[threadIdx.x] =
                fmaxf(s_partial[threadIdx.x],
                      s_partial[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        s_max_score = s_partial[0];
    }
    __syncthreads();

    float sum_exp = 0.0f;
    for (size_t j = threadIdx.x; j < eff_kv_len; j += BLOCK_SIZE) {
        const __half *k_vec = k_head + j * nkvhead * d;

        float score = 0.0f;
        for (size_t t = 0; t < d; ++t) {
            score += __half2float(q_vec[t]) *
                     __half2float(k_vec[t]);
        }
        score *= scale;

        const float w = expf(score - s_max_score);
        s_weights[j] = w;
        sum_exp += w;
    }

    s_partial[threadIdx.x] = sum_exp;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_partial[threadIdx.x] +=
                s_partial[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        s_sum_exp = s_partial[0];
    }
    __syncthreads();

    for (size_t c = threadIdx.x; c < dv; c += BLOCK_SIZE) {
        float acc = 0.0f;
        for (size_t j = 0; j < eff_kv_len; ++j) {
            const __half *v_vec = v_head + j * nkvhead * dv;
            acc += (s_weights[j] / s_sum_exp) *
                   __half2float(v_vec[c]);
        }
        attn_val[(token * nhead + h) * dv + c] =
            __float2half(acc);
    }
}

__global__ void self_attention_bf16_kernel(
    __nv_bfloat16 *attn_val,
    const __nv_bfloat16 *q,
    const __nv_bfloat16 *k,
    const __nv_bfloat16 *v,
    float scale,
    size_t seq_len,
    size_t total_len,
    size_t nhead,
    size_t nkvhead,
    size_t d,
    size_t dv
) {
    extern __shared__ float s_weights[];
    __shared__ float s_partial[BLOCK_SIZE];
    __shared__ float s_max_score;
    __shared__ float s_sum_exp;

    const size_t token = blockIdx.x / nhead;
    const size_t h = blockIdx.x % nhead;
    const size_t kv_head = h * nkvhead / nhead;

    const size_t past_len = total_len - seq_len;
    const size_t eff_kv_len = past_len + token + 1;

    const __nv_bfloat16 *q_vec = q + (token * nhead + h) * d;
    const __nv_bfloat16 *k_head = k + kv_head * d;
    const __nv_bfloat16 *v_head = v + kv_head * dv;

    float max_score = -INFINITY;
    for (size_t j = threadIdx.x; j < eff_kv_len; j += BLOCK_SIZE) {
        const __nv_bfloat16 *k_vec = k_head + j * nkvhead * d;

        float score = 0.0f;
        for (size_t t = 0; t < d; ++t) {
            score += __bfloat162float(q_vec[t]) *
                     __bfloat162float(k_vec[t]);
        }
        score *= scale;

        if (score > max_score) {
            max_score = score;
        }
    }

    s_partial[threadIdx.x] = max_score;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_partial[threadIdx.x] =
                fmaxf(s_partial[threadIdx.x],
                      s_partial[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        s_max_score = s_partial[0];
    }
    __syncthreads();

    float sum_exp = 0.0f;
    for (size_t j = threadIdx.x; j < eff_kv_len; j += BLOCK_SIZE) {
        const __nv_bfloat16 *k_vec = k_head + j * nkvhead * d;

        float score = 0.0f;
        for (size_t t = 0; t < d; ++t) {
            score += __bfloat162float(q_vec[t]) *
                     __bfloat162float(k_vec[t]);
        }
        score *= scale;

        const float w = expf(score - s_max_score);
        s_weights[j] = w;
        sum_exp += w;
    }

    s_partial[threadIdx.x] = sum_exp;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_partial[threadIdx.x] +=
                s_partial[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        s_sum_exp = s_partial[0];
    }
    __syncthreads();

    for (size_t c = threadIdx.x; c < dv; c += BLOCK_SIZE) {
        float acc = 0.0f;
        for (size_t j = 0; j < eff_kv_len; ++j) {
            const __nv_bfloat16 *v_vec = v_head + j * nkvhead * dv;
            acc += (s_weights[j] / s_sum_exp) *
                   __bfloat162float(v_vec[c]);
        }
        attn_val[(token * nhead + h) * dv + c] =
            __float2bfloat16(acc);
    }
}

} // anonymous namespace


namespace llaisys::ops::nvidia {

void self_attention(
    std::byte *attn_val,
    const std::byte *q,
    const std::byte *k,
    const std::byte *v,
    llaisysDataType_t type,
    float scale,
    size_t seq_len,
    size_t total_len,
    size_t nhead,
    size_t nkvhead,
    size_t d,
    size_t dv
) {
    if (seq_len == 0 || total_len == 0 ||
        nhead == 0 || nkvhead == 0 || d == 0 || dv == 0) {
        return;
    }

    // 动态共享内存：s_weights[total_len] + 归约缓冲等
    const size_t shared_bytes =
        total_len * sizeof(float) +
        BLOCK_SIZE * sizeof(float) +
        2 * sizeof(float);

    if (shared_bytes > 48 * 1024) {
        throw std::runtime_error(
            "NVIDIA self_attention: total_len too large for shared memory"
        );
    }

    const int grid_size =
        static_cast<int>(seq_len * nhead);

    switch (type) {

    case LLAISYS_DTYPE_F32:
        self_attention_f32_kernel<<<grid_size, BLOCK_SIZE, shared_bytes>>>(
            reinterpret_cast<float *>(attn_val),
            reinterpret_cast<const float *>(q),
            reinterpret_cast<const float *>(k),
            reinterpret_cast<const float *>(v),
            scale,
            seq_len,
            total_len,
            nhead,
            nkvhead,
            d,
            dv
        );
        break;

    case LLAISYS_DTYPE_F16:
        self_attention_f16_kernel<<<grid_size, BLOCK_SIZE, shared_bytes>>>(
            reinterpret_cast<__half *>(attn_val),
            reinterpret_cast<const __half *>(q),
            reinterpret_cast<const __half *>(k),
            reinterpret_cast<const __half *>(v),
            scale,
            seq_len,
            total_len,
            nhead,
            nkvhead,
            d,
            dv
        );
        break;

    case LLAISYS_DTYPE_BF16:
        self_attention_bf16_kernel<<<grid_size, BLOCK_SIZE, shared_bytes>>>(
            reinterpret_cast<__nv_bfloat16 *>(attn_val),
            reinterpret_cast<const __nv_bfloat16 *>(q),
            reinterpret_cast<const __nv_bfloat16 *>(k),
            reinterpret_cast<const __nv_bfloat16 *>(v),
            scale,
            seq_len,
            total_len,
            nhead,
            nkvhead,
            d,
            dv
        );
        break;

    default:
        throw std::runtime_error(
            "NVIDIA self_attention: unsupported datatype"
        );
    }

    checkCuda(
        cudaGetLastError(),
        "self_attention kernel launch"
    );
}

} // namespace llaisys::ops::nvidia
