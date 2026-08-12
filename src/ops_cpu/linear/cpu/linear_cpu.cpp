#include "linear_cpu.hpp"
#include "../../../utils.hpp"

#include <cstddef>

#ifdef _OPENMP
#include <omp.h>
#endif


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
    /*
     * 每个 (m, o) 输出完全独立。
     *
     * 并行这个维度不会改变单个输出内部 k 的累加顺序，
     * 因此相比对 k 做并行 reduction，数值稳定性更好。
     */
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t m = 0; m < M; ++m)
    {
        for (size_t o = 0; o < O; ++o)
        {
            const T *in_row =
                in + m * K;

            const T *weight_row =
                weight + o * K;

            float acc = 0.0f;

            if (bias != nullptr)
            {
                acc =
                    llaisys::utils::cast<float>(
                        bias[o]);
            }

            /*
             * 保持原来的 k 顺序。
             *
             * 不使用 OpenMP reduction，
             * 避免改变浮点累加顺序。
             */
            for (size_t k = 0; k < K; ++k)
            {
                const float x =
                    llaisys::utils::cast<float>(
                        in_row[k]);

                const float w =
                    llaisys::utils::cast<float>(
                        weight_row[k]);

                acc += x * w;
            }

            out[m * O + o] =
                llaisys::utils::cast<T>(acc);
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
        auto *out_ptr =
            reinterpret_cast<float *>(out);

        const auto *in_ptr =
            reinterpret_cast<const float *>(in);

        const auto *weight_ptr =
            reinterpret_cast<const float *>(weight);

        const float *bias_ptr = nullptr;

        if (bias != nullptr)
        {
            bias_ptr =
                reinterpret_cast<const float *>(
                    bias);
        }

        linear_(
            out_ptr,
            in_ptr,
            weight_ptr,
            bias_ptr,
            M,
            K,
            O);

        break;
    }

    case LLAISYS_DTYPE_F16:
    {
        auto *out_ptr =
            reinterpret_cast<llaisys::fp16_t *>(
                out);

        const auto *in_ptr =
            reinterpret_cast<
                const llaisys::fp16_t *>(
                in);

        const auto *weight_ptr =
            reinterpret_cast<
                const llaisys::fp16_t *>(
                weight);

        const llaisys::fp16_t *bias_ptr =
            nullptr;

        if (bias != nullptr)
        {
            bias_ptr =
                reinterpret_cast<
                    const llaisys::fp16_t *>(
                    bias);
        }

        linear_(
            out_ptr,
            in_ptr,
            weight_ptr,
            bias_ptr,
            M,
            K,
            O);

        break;
    }

    case LLAISYS_DTYPE_BF16:
    {
        auto *out_ptr =
            reinterpret_cast<llaisys::bf16_t *>(
                out);

        const auto *in_ptr =
            reinterpret_cast<
                const llaisys::bf16_t *>(
                in);

        const auto *weight_ptr =
            reinterpret_cast<
                const llaisys::bf16_t *>(
                weight);

        const llaisys::bf16_t *bias_ptr =
            nullptr;

        if (bias != nullptr)
        {
            bias_ptr =
                reinterpret_cast<
                    const llaisys::bf16_t *>(
                    bias);
        }

        linear_(
            out_ptr,
            in_ptr,
            weight_ptr,
            bias_ptr,
            M,
            K,
            O);

        break;
    }

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
        break;
    }
}

} // namespace llaisys::ops::cpu