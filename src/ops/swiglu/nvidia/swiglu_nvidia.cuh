#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {

// Elementwise SwiGLU: out_i = up_i * SiLU(gate_i)
//   SiLU(x) = x / (1 + exp(-x))
//
// gate, up, out all have shape (numel,), treated as contiguous.
void swiglu(
    std::byte *out,
    const std::byte *gate,
    const std::byte *up,
    llaisysDataType_t dtype,
    size_t numel
);

} // namespace llaisys::ops::nvidia
