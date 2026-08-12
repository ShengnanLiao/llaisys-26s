#include "../runtime_api.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace llaisys::device::nvidia {

namespace runtime_api {

/*
 * ============================================================
 * CUDA 错误检查
 * ============================================================
 */
static void checkCuda(
    cudaError_t err,
    const char *expr) {

    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA Runtime error in ") +
            expr +
            ": " +
            cudaGetErrorString(err));
    }
}


/*
 * ============================================================
 * LLAISYS memcpy kind -> CUDA memcpy kind
 * ============================================================
 */
static cudaMemcpyKind toCudaMemcpyKind(
    llaisysMemcpyKind_t kind) {

    switch (kind) {

    case LLAISYS_MEMCPY_H2H:
        return cudaMemcpyHostToHost;

    case LLAISYS_MEMCPY_H2D:
        return cudaMemcpyHostToDevice;

    case LLAISYS_MEMCPY_D2H:
        return cudaMemcpyDeviceToHost;

    case LLAISYS_MEMCPY_D2D:
        return cudaMemcpyDeviceToDevice;

    default:
        throw std::invalid_argument(
            "Invalid llaisysMemcpyKind_t");
    }
}


/*
 * ============================================================
 * Device
 * ============================================================
 */

int getDeviceCount() {

    int count = 0;

    checkCuda(
        cudaGetDeviceCount(&count),
        "cudaGetDeviceCount");

    return count;
}


void setDevice(int device_id) {

    checkCuda(
        cudaSetDevice(device_id),
        "cudaSetDevice");
}


void deviceSynchronize() {

    checkCuda(
        cudaDeviceSynchronize(),
        "cudaDeviceSynchronize");
}


/*
 * ============================================================
 * Stream
 * ============================================================
 */

llaisysStream_t createStream() {

    cudaStream_t stream = nullptr;

    checkCuda(
        cudaStreamCreate(&stream),
        "cudaStreamCreate");

    return reinterpret_cast<llaisysStream_t>(
        stream);
}


void destroyStream(
    llaisysStream_t stream) {

    if (stream == nullptr) {
        return;
    }

    cudaStream_t cuda_stream =
        reinterpret_cast<cudaStream_t>(
            stream);

    checkCuda(
        cudaStreamDestroy(cuda_stream),
        "cudaStreamDestroy");
}


void streamSynchronize(
    llaisysStream_t stream) {

    cudaStream_t cuda_stream =
        reinterpret_cast<cudaStream_t>(
            stream);

    checkCuda(
        cudaStreamSynchronize(cuda_stream),
        "cudaStreamSynchronize");
}


/*
 * ============================================================
 * Device Memory
 * ============================================================
 */

void *mallocDevice(size_t size) {

    void *ptr = nullptr;

    checkCuda(
        cudaMalloc(&ptr, size),
        "cudaMalloc");

    return ptr;
}


void freeDevice(void *ptr) {

    if (ptr == nullptr) {
        return;
    }

    checkCuda(
        cudaFree(ptr),
        "cudaFree");
}


/*
 * ============================================================
 * Host Memory
 *
 * 使用 pinned/page-locked host memory。
 * 后续 cudaMemcpyAsync 也可以真正异步工作。
 * ============================================================
 */

void *mallocHost(size_t size) {

    void *ptr = nullptr;

    checkCuda(
        cudaMallocHost(&ptr, size),
        "cudaMallocHost");

    return ptr;
}


void freeHost(void *ptr) {

    if (ptr == nullptr) {
        return;
    }

    checkCuda(
        cudaFreeHost(ptr),
        "cudaFreeHost");
}


/*
 * ============================================================
 * Synchronous memcpy
 * ============================================================
 */

void memcpySync(
    void *dst,
    const void *src,
    size_t size,
    llaisysMemcpyKind_t kind) {

    checkCuda(
        cudaMemcpy(
            dst,
            src,
            size,
            toCudaMemcpyKind(kind)),
        "cudaMemcpy");
}


/*
 * ============================================================
 * Asynchronous memcpy
 * 必须和 LlaisysRuntimeAPI 中 memcpy_async_api 完全一致。
 * ============================================================
 */

void memcpyAsync(
    void *dst,
    const void *src,
    size_t size,
    llaisysMemcpyKind_t kind,
    llaisysStream_t stream) {

    cudaStream_t cuda_stream =
        reinterpret_cast<cudaStream_t>(
            stream);

    checkCuda(
        cudaMemcpyAsync(
            dst,
            src,
            size,
            toCudaMemcpyKind(kind),
            cuda_stream),
        "cudaMemcpyAsync");
}


/*
 * ============================================================
 * Runtime API table
 * ============================================================
 */

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,

    &createStream,
    &destroyStream,
    &streamSynchronize,

    &mallocDevice,
    &freeDevice,

    &mallocHost,
    &freeHost,

    &memcpySync,
    &memcpyAsync
};

} // namespace runtime_api


const LlaisysRuntimeAPI *getRuntimeAPI() {

    return &runtime_api::RUNTIME_API;
}

} // namespace llaisys::device::nvidia