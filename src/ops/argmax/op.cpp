#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"

namespace llaisys::ops {

void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);

    // 当前作业只要求支持 1D vals
    ASSERT(vals->ndim() == 1, "Argmax: vals must be a 1D tensor.");

    // max_idx 和 max_val 都只保存一个元素
    ASSERT(max_idx->ndim() == 1 && max_idx->numel() == 1,
           "Argmax: max_idx must be a 1D tensor with one element.");

    ASSERT(max_val->ndim() == 1 && max_val->numel() == 1,
           "Argmax: max_val must be a 1D tensor with one element.");

    // 最大值的数据类型应该和输入一致
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());

    // 当前实现先只处理 contiguous tensor
    ASSERT(
        max_idx->isContiguous() &&
            max_val->isContiguous() &&
            vals->isContiguous(),
        "Argmax: all tensors must be contiguous.");

    ASSERT(vals->numel() > 0, "Argmax: vals must not be empty.");

    /*
     * 如果测试里的 max_idx 确实是 i64，
     * 建议加上这个检查。
     */
    ASSERT(
        max_idx->dtype() == LLAISYS_DTYPE_I64,
        "Argmax: max_idx dtype must be int64.");

    // CPU
    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(
            max_idx->data(),
            max_val->data(),
            vals->data(),
            vals->dtype(),
            vals->numel());
    }

    /*
     * 非 CPU 情况先切换当前 device。
     */
    llaisys::core::context().setDevice(
        vals->deviceType(),
        vals->deviceId());

    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(
            max_idx->data(),
            max_val->data(),
            vals->data(),
            vals->dtype(),
            vals->numel());

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