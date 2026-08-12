#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(
    T *out,
    const T *in,
    const T *weight,
    float eps,
    size_t rows,
    size_t dim) {

    for (size_t r = 0; r < rows; r++) {
        const T *in_row = in + r * dim;
        T *out_row = out + r * dim;

        // mean of squares over the last dimension
        float sum_sq = 0.0f;
        for (size_t k = 0; k < dim; k++) {
            float v = llaisys::utils::cast<float>(in_row[k]);
            sum_sq += v * v;
        }
        float rms = std::sqrt(sum_sq / static_cast<float>(dim) + eps);

        for (size_t k = 0; k < dim; k++) {
            float v = llaisys::utils::cast<float>(in_row[k]);
            float w = llaisys::utils::cast<float>(weight[k]);
            out_row[k] = llaisys::utils::cast<T>((v / rms) * w);
        }
    }
}

namespace llaisys::ops::cpu {

void rms_norm(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    llaisysDataType_t type,
    float eps,
    size_t rows,
    size_t dim) {

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            eps, rows, dim);

    case LLAISYS_DTYPE_F16:
        return rms_norm_(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            eps, rows, dim);

    case LLAISYS_DTYPE_BF16:
        return rms_norm_(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            eps, rows, dim);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu
