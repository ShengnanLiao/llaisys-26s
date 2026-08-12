#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/swiglu_cpu.hpp"

namespace llaisys::ops {

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);

    // 当前任务要求：
    // gate, up, out 都是形状相同的二维连续张量
    ASSERT(gate->ndim() == 2,
           "SwiGLU: gate must be a 2D tensor.");

    ASSERT(up->ndim() == 2,
           "SwiGLU: up must be a 2D tensor.");

    ASSERT(out->ndim() == 2,
           "SwiGLU: out must be a 2D tensor.");

    // 所有张量 dtype 必须一致
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());

    // 输出 shape 与 gate/up 一致
    ASSERT(out->shape() == gate->shape(),
           "SwiGLU: out.shape must equal gate.shape.");

    ASSERT(out->shape() == up->shape(),
           "SwiGLU: out.shape must equal up.shape.");

    // 当前 CPU 实现按连续内存处理
    ASSERT(out->isContiguous() &&
               gate->isContiguous() &&
               up->isContiguous(),
           "SwiGLU: all tensors must be contiguous.");

    const size_t numel = out->numel();

    // CPU
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu(
            out->data(),
            gate->data(),
            up->data(),
            gate->dtype(),
            numel);
    }

    // 切换当前设备
    llaisys::core::context().setDevice(
        out->deviceType(),
        out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(
            out->data(),
            gate->data(),
            up->data(),
            gate->dtype(),
            numel);

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
