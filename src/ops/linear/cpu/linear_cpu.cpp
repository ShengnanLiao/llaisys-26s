#include "linear_cpu.hpp"
#include "../../../utils.hpp"
#include <cstddef>

template <typename T>
void linear_(
    T *out,
    const T *in,
    const T *weight,
    const T *bias,
    size_t M,
    size_t K,
    size_t O)
{
    for (size_t m = 0; m < M; ++m)
    {
        const T *in_row = in + m * K;
        T *out_row = out + m * O;

        for (size_t o = 0; o < O; ++o)
        {
            const T *weight_row = weight + o * K;
            float acc = 0.0f;

            if (bias != nullptr)
            {
                acc = llaisys::utils::cast<float>(bias[o]);
            }

            for (size_t k = 0; k < K; ++k)
            {
                float x = llaisys::utils::cast<float>(in_row[k]);
                float w = llaisys::utils::cast<float>(weight_row[k]);
                acc += x * w;
            }

            out_row[o] = llaisys::utils::cast<T>(acc);
        }
    }
}

namespace llaisys::ops::cpu
{

void linear(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    const std::byte *bias,
    llaisysDataType_t type,
    size_t M,
    size_t K,
    size_t O)
{
    switch (type)
    {
    case LLAISYS_DTYPE_F32:
    {
        float *out_ptr = reinterpret_cast<float *>(out);
        const float *in_ptr = reinterpret_cast<const float *>(in);
        const float *weight_ptr = reinterpret_cast<const float *>(weight);
        const float *bias_ptr = nullptr;
        if (bias != nullptr)
            bias_ptr = reinterpret_cast<const float *>(bias);

        linear_(out_ptr, in_ptr, weight_ptr, bias_ptr, M, K, O);
        break;
    }

    case LLAISYS_DTYPE_F16:
    {
        llaisys::fp16_t *out_ptr = reinterpret_cast<llaisys::fp16_t *>(out);
        const llaisys::fp16_t *in_ptr = reinterpret_cast<const llaisys::fp16_t *>(in);
        const llaisys::fp16_t *weight_ptr = reinterpret_cast<const llaisys::fp16_t *>(weight);
        const llaisys::fp16_t *bias_ptr = nullptr;
        if (bias != nullptr)
            bias_ptr = reinterpret_cast<const llaisys::fp16_t *>(bias);

        linear_(out_ptr, in_ptr, weight_ptr, bias_ptr, M, K, O);
        break;
    }

    case LLAISYS_DTYPE_BF16:
    {
        llaisys::bf16_t *out_ptr = reinterpret_cast<llaisys::bf16_t *>(out);
        const llaisys::bf16_t *in_ptr = reinterpret_cast<const llaisys::bf16_t *>(in);
        const llaisys::bf16_t *weight_ptr = reinterpret_cast<const llaisys::bf16_t *>(weight);
        const llaisys::bf16_t *bias_ptr = nullptr;
        if (bias != nullptr)
            bias_ptr = reinterpret_cast<const llaisys::bf16_t *>(bias);

        linear_(out_ptr, in_ptr, weight_ptr, bias_ptr, M, K, O);
        break;
    }

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
        break;
    }
}

} // namespace llaisys::ops::cpu