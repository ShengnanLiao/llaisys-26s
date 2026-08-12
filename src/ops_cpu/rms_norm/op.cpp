#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);

    // 当前任务要求：
    // in    : 2-D (rows, dim)
    // weight: 1-D (dim,)
    // out   : 2-D (rows, dim)
    ASSERT(in->ndim() == 2,
           "RMSNorm: in must be a 2D tensor.");

    ASSERT(out->ndim() == 2,
           "RMSNorm: out must be a 2D tensor.");

    ASSERT(weight->ndim() == 1,
           "RMSNorm: weight must be a 1D tensor.");

    // 所有张量 dtype 必须一致
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());

    const size_t rows = in->shape()[0];
    const size_t dim = in->shape()[1];

    // 输出 shape:
    //
    // out.shape == in.shape
    // weight.shape[0] == dim
    ASSERT(out->shape()[0] == rows,
           "RMSNorm: out.shape[0] must equal in.shape[0].");

    ASSERT(out->shape()[1] == dim,
           "RMSNorm: out.shape[1] must equal in.shape[1].");

    ASSERT(weight->shape()[0] == dim,
           "RMSNorm: weight.shape[0] must equal in.shape[1].");

    // 当前 CPU 实现按连续内存处理
    ASSERT(out->isContiguous() &&
               in->isContiguous() &&
               weight->isContiguous(),
           "RMSNorm: all tensors must be contiguous.");

    // CPU
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(
            out->data(),
            in->data(),
            weight->data(),
            in->dtype(),
            eps,
            rows, dim);
    }

    // 切换当前设备
    llaisys::core::context().setDevice(
        out->deviceType(),
        out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(
            out->data(),
            in->data(),
            weight->data(),
            in->dtype(),
            eps,
            rows, dim);

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
