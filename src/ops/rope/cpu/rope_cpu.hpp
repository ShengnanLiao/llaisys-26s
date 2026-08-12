#pragma once
#include "llaisys.h"

#include <cstddef>
namespace llaisys::ops::cpu {

// RoPE: for each token s and each head h, take the d-length vector
// [a | b] (each half of length d/2) and 2D-rotate feature pair j by angle
//   phi = p_s / theta^(2j/d)
//   new_a = a*cos(phi) - b*sin(phi)
//   new_b = a*sin(phi) + b*cos(phi)
//
// in     : (seq_len, nhead, dim) contiguous
// pos_ids: (seq_len,) int64 positions
// out    : same shape/layout as in
// theta  : frequency base, e.g. 10000
void rope(
    std::byte *out,
    const std::byte *in,
    const std::byte *pos_ids,
    llaisysDataType_t type,
    float theta,
    size_t seq_len,
    size_t nhead,
    size_t dim);

} // namespace llaisys::ops::cpu
