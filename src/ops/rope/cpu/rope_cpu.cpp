#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rope_(
    T *out,
    const T *in,
    const int64_t *pos_ids,
    float theta,
    size_t seq_len,
    size_t nhead,
    size_t dim) {

    const size_t half = dim / 2;

    for (size_t s = 0; s < seq_len; s++) {
        const float p = static_cast<float>(pos_ids[s]);

        for (size_t h = 0; h < nhead; h++) {
            const size_t row_offset = (s * nhead + h) * dim;
            const T *in_row = in + row_offset;
            T *out_row = out + row_offset;

            for (size_t j = 0; j < half; j++) {
                // phi = p / theta^(2j/d)
                const float freq = p / std::pow(theta, (2.0 * static_cast<float>(j)) / dim);
                const float cos_ = std::cos(freq);
                const float sin_ = std::sin(freq);

                const float a = llaisys::utils::cast<float>(in_row[j]);
                const float b = llaisys::utils::cast<float>(in_row[j + half]);

                out_row[j] = llaisys::utils::cast<T>(a * cos_ - b * sin_);
                out_row[j + half] = llaisys::utils::cast<T>(a * sin_ + b * cos_);
            }
        }
    }
}

namespace llaisys::ops::cpu {

void rope(
    std::byte *out,
    const std::byte *in,
    const std::byte *pos_ids,
    llaisysDataType_t type,
    float theta,
    size_t seq_len,
    size_t nhead,
    size_t dim) {

    const auto *positions = reinterpret_cast<const int64_t *>(pos_ids);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            positions,
            theta, seq_len, nhead, dim);

    case LLAISYS_DTYPE_F16:
        return rope_(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            positions,
            theta, seq_len, nhead, dim);

    case LLAISYS_DTYPE_BF16:
        return rope_(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            positions,
            theta, seq_len, nhead, dim);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu
