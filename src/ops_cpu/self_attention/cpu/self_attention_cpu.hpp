#pragma once
#include "llaisys.h"

#include <cstddef>
namespace llaisys::ops::cpu {

// Causal self-attention:
//   scores = Q K^T * scale, causal masked, softmax, then Y = softmax(scores) V
//
// q       : (seq_len, nhead, d)
// k       : (total_len, nkvhead, d)   total_len >= seq_len (may include history)
// v       : (total_len, nkvhead, dv)
// attn_val: (seq_len, nhead, dv)
// scale   : attention scale, usually 1/sqrt(d)
//
// Supports MQA/GQA: query head h maps to kv head (h * nkvhead / nhead).
void self_attention(
    std::byte *attn_val,
    const std::byte *q,
    const std::byte *k,
    const std::byte *v,
    llaisysDataType_t type,
    float scale,
    size_t seq_len,
    size_t total_len,
    size_t nhead,
    size_t nkvhead,
    size_t d,
    size_t dv);

} // namespace llaisys::ops::cpu
