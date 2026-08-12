#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);

    // 当前任务要求：
    // in   : 2-D (M, K)
    // weight: 2-D (O, K)   -- 未转置，计算时处理转置
    // out  : 2-D (M, O)
    // bias : 1-D (O,), 可选
    ASSERT(in->ndim() == 2,
           "Linear: in must be a 2D tensor.");

    ASSERT(weight->ndim() == 2,
           "Linear: weight must be a 2D tensor.");

    ASSERT(out->ndim() == 2,
           "Linear: out must be a 2D tensor.");

    // 所有张量 dtype 必须一致
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());

    const size_t M = in->shape()[0];
    const size_t K = in->shape()[1];
    const size_t O = weight->shape()[0];

    // 输出 shape:
    //
    // out.shape[0] == in.shape[0]
    // out.shape[1] == weight.shape[0]
    ASSERT(weight->shape()[1] == K,
           "Linear: weight.shape[1] must equal in.shape[1].");

    ASSERT(out->shape()[0] == M,
           "Linear: out.shape[0] must equal in.shape[0].");

    ASSERT(out->shape()[1] == O,
           "Linear: out.shape[1] must equal weight.shape[0].");

    // 可选偏置：1-D (O,)，与 weight 同 device / dtype
    if (bias) {
        CHECK_SAME_DEVICE(bias, out);
        CHECK_SAME_DTYPE(bias->dtype(), weight->dtype());

        ASSERT(bias->ndim() == 1,
               "Linear: bias must be a 1D tensor.");

        ASSERT(bias->shape()[0] == O,
               "Linear: bias.shape[0] must equal weight.shape[0].");

        ASSERT(bias->isContiguous(),
               "Linear: bias must be contiguous.");
    }

    // 当前 CPU 实现按连续内存处理
    ASSERT(out->isContiguous() &&
               in->isContiguous() &&
               weight->isContiguous(),
           "Linear: out/in/weight must be contiguous.");

    const std::byte *bias_data = bias ? bias->data() : nullptr;

    // CPU
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(
            out->data(),
            in->data(),
            weight->data(),
            bias_data,
            in->dtype(),
            M, K, O);
    }

    // 切换当前设备
    llaisys::core::context().setDevice(
        out->deviceType(),
        out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(
            out->data(),
            in->data(),
            weight->data(),
            bias_data,
            in->dtype(),
            M, K, O);

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
