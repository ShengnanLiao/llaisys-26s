#pragma once
#include "llaisys.h"

#include <cstddef>
namespace llaisys::ops::cpu {

// Y = X * W^T + b
//
// X      : (M, K)
// W      : (O, K)   -- not transposed, transposition handled inside the kernel
// b      : (O, )    -- optional, may be nullptr
// out(Y) : (M, O)
void linear(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    const std::byte *bias,
    llaisysDataType_t type,
    size_t M,
    size_t K,
    size_t O);

} // namespace llaisys::ops::cpu
