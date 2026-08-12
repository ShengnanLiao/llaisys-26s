#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);

    // 当前任务要求：
    // in     : 3-D [seq_len, nhead, d]
    // out    : 3-D, 与 in 形状一致
    // pos_ids: 1-D [seq_len] Int64
    ASSERT(in->ndim() == 3,
           "RoPE: in must be a 3D tensor.");

    ASSERT(out->ndim() == 3,
           "RoPE: out must be a 3D tensor.");

    ASSERT(pos_ids->ndim() == 1,
           "RoPE: pos_ids must be a 1D tensor.");

    // pos_ids 必须是 Int64
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64,
           "RoPE: pos_ids dtype must be int64.");

    // in 和 out dtype 必须一致
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());

    const size_t seq_len = in->shape()[0];
    const size_t nhead = in->shape()[1];
    const size_t dim = in->shape()[2];

    // 最后一维 d 必须能被 2 整除
    ASSERT(dim % 2 == 0,
           "RoPE: last dimension must be even.");

    // 输出 shape 与输入一致
    ASSERT(out->shape()[0] == seq_len &&
               out->shape()[1] == nhead &&
               out->shape()[2] == dim,
           "RoPE: out.shape must equal in.shape.");

    // pos_ids 长度与 seq_len 一致
    ASSERT(pos_ids->shape()[0] == seq_len,
           "RoPE: pos_ids length must equal seq_len.");

    // 当前 CPU 实现按连续内存处理
    ASSERT(out->isContiguous() &&
               in->isContiguous() &&
               pos_ids->isContiguous(),
           "RoPE: all tensors must be contiguous.");

    // CPU
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(
            out->data(),
            in->data(),
            pos_ids->data(),
            in->dtype(),
            theta,
            seq_len, nhead, dim);
    }

    // 切换当前设备
    llaisys::core::context().setDevice(
        out->deviceType(),
        out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(
            out->data(),
            in->data(),
            pos_ids->data(),
            in->dtype(),
            theta,
            seq_len, nhead, dim);

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
