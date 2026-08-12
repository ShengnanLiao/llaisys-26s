#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {

void rms_norm(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    llaisysDataType_t dtype,
    float eps,
    size_t rows,
    size_t dim
);

} // namespace llaisys::ops::nvidia
