#pragma once
#include "llaisys.h"

#include <cstddef>
namespace llaisys::ops::cpu {

// Y_i[k] = (X_i[k] / sqrt(mean(X_i^2) + eps)) * W_i[k]
//
// X      : (rows, dim)
// W      : (dim, )
// out(Y) : (rows, dim)
// eps    : small epsilon to avoid division by zero
void rms_norm(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    llaisysDataType_t type,
    float eps,
    size_t rows,
    size_t dim);

} // namespace llaisys::ops::cpu
