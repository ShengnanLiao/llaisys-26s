#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstring>

template <typename T>
void embedding_(
    T *out,
    const int64_t *index,
    const T *weight,
    size_t num_indices,
    size_t embedding_dim) {

    for (size_t i = 0; i < num_indices; i++) {
        int64_t row = index[i];

        const T *src_row =
            weight + row * embedding_dim;

        T *dst_row =
            out + i * embedding_dim;

        std::memcpy(
            dst_row,
            src_row,
            embedding_dim * sizeof(T));
    }
}

namespace llaisys::ops::cpu {

void embedding(
    std::byte *out,
    const std::byte *index,
    const std::byte *weight,
    llaisysDataType_t type,
    size_t num_indices,
    size_t embedding_dim) {

    const auto *indices =
        reinterpret_cast<const int64_t *>(index);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(
            reinterpret_cast<float *>(out),
            indices,
            reinterpret_cast<const float *>(weight),
            num_indices,
            embedding_dim);

    case LLAISYS_DTYPE_F16:
        return embedding_(
            reinterpret_cast<llaisys::fp16_t *>(out),
            indices,
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            num_indices,
            embedding_dim);

    case LLAISYS_DTYPE_BF16:
        return embedding_(
            reinterpret_cast<llaisys::bf16_t *>(out),
            indices,
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            num_indices,
            embedding_dim);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu