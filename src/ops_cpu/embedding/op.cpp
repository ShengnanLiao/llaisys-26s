#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {

void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);

    // 当前任务要求：
    // index: 1-D
    // weight: 2-D
    // out: 2-D
    ASSERT(index->ndim() == 1,
           "Embedding: index must be a 1D tensor.");

    ASSERT(weight->ndim() == 2,
           "Embedding: weight must be a 2D tensor.");

    ASSERT(out->ndim() == 2,
           "Embedding: out must be a 2D tensor.");

    // index 必须是 Int64
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64,
           "Embedding: index dtype must be int64.");

    // out 和 weight dtype 必须一致
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());

    // 输出 shape:
    //
    // out.shape[0] == index.shape[0]
    // out.shape[1] == weight.shape[1]
    ASSERT(out->shape()[0] == index->shape()[0],
           "Embedding: out.shape[0] must equal index.shape[0].");

    ASSERT(out->shape()[1] == weight->shape()[1],
           "Embedding: out.shape[1] must equal weight.shape[1].");

    // 当前 CPU 实现按连续内存处理
    ASSERT(out->isContiguous() &&
               index->isContiguous() &&
               weight->isContiguous(),
           "Embedding: all tensors must be contiguous.");

    // CPU
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(
            out->data(),
            index->data(),
            weight->data(),
            weight->dtype(),
            index->numel(),
            weight->shape()[1]);
    }

    // 切换当前设备
    llaisys::core::context().setDevice(
        out->deviceType(),
        out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(
            out->data(),
            index->data(),
            weight->data(),
            weight->dtype(),
            index->numel(),
            weight->shape()[1]);

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