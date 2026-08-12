#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);

    // 当前任务要求：
    // q       : 3-D [seq_len, nhead, d]
    // k       : 3-D [total_len, nkvhead, d]
    // v       : 3-D [total_len, nkvhead, dv]
    // attn_val: 3-D [seq_len, nhead, dv]
    ASSERT(q->ndim() == 3,
           "SelfAttention: q must be a 3D tensor.");

    ASSERT(k->ndim() == 3,
           "SelfAttention: k must be a 3D tensor.");

    ASSERT(v->ndim() == 3,
           "SelfAttention: v must be a 3D tensor.");

    ASSERT(attn_val->ndim() == 3,
           "SelfAttention: attn_val must be a 3D tensor.");

    // 所有张量 dtype 必须一致
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());

    const size_t seq_len = q->shape()[0];
    const size_t nhead = q->shape()[1];
    const size_t d = q->shape()[2];

    const size_t total_len = k->shape()[0];
    const size_t nkvhead = k->shape()[1];
    const size_t dv = v->shape()[2];

    // k/v 的序列长度必须一致，total_len 可 >= seq_len (含 KV Cache 历史)
    ASSERT(k->shape()[0] == total_len &&
               v->shape()[0] == total_len,
           "SelfAttention: k/v seq length must match.");

    // q 与 k 的特征维 d 一致
    ASSERT(k->shape()[2] == d,
           "SelfAttention: k.shape[2] must equal q.shape[2].");

    // q 与 k 的 head 数一致
    ASSERT(k->shape()[1] == nkvhead &&
               v->shape()[1] == nkvhead,
           "SelfAttention: k/v head count must match.");

    // GQA/MQA: nhead 必须是 nkvhead 的整数倍
    ASSERT(nhead % nkvhead == 0,
           "SelfAttention: nhead must be a multiple of nkvhead.");

    // attn_val 与 q 的 seq/head 一致，与 v 的 dv 一致
    ASSERT(attn_val->shape()[0] == seq_len,
           "SelfAttention: attn_val.shape[0] must equal q.shape[0].");

    ASSERT(attn_val->shape()[1] == nhead,
           "SelfAttention: attn_val.shape[1] must equal q.shape[1].");

    ASSERT(attn_val->shape()[2] == dv,
           "SelfAttention: attn_val.shape[2] must equal v.shape[2].");

    // 当前 CPU 实现按连续内存处理
    ASSERT(attn_val->isContiguous() &&
               q->isContiguous() &&
               k->isContiguous() &&
               v->isContiguous(),
           "SelfAttention: all tensors must be contiguous.");

    // CPU
    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(
            attn_val->data(),
            q->data(),
            k->data(),
            v->data(),
            q->dtype(),
            scale,
            seq_len, total_len, nhead, nkvhead, d, dv);
    }

    // 切换当前设备
    llaisys::core::context().setDevice(
        attn_val->deviceType(),
        attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(
            attn_val->data(),
            q->data(),
            k->data(),
            v->data(),
            q->dtype(),
            scale,
            seq_len, total_len, nhead, nkvhead, d, dv);

#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif

    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops
