#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);

    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }

    TensorMeta meta{dtype, shape, strides};

    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU &&
        core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage =
            core::context().runtime().allocateHostStorage(total_elems * dtype_size);

        return std::shared_ptr<Tensor>(
            new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);

        auto storage =
            core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);

        return std::shared_ptr<Tensor>(
            new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(
        _meta.shape.begin(),
        _meta.shape.end(),
        size_t(1),
        std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";

    for (auto s : this->shape()) {
        ss << s << " ";
    }

    ss << "] strides[ ";

    for (auto s : this->strides()) {
        ss << s << " ";
    }

    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data,
                const std::vector<size_t> &shape,
                const std::vector<ptrdiff_t> &strides,
                size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> ||
                          std::is_same_v<T, fp16_t>) {
                std::cout
                    << utils::cast<float>(data[i * strides[dim]])
                    << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }

        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(
                data + i * strides[dim],
                shape,
                strides,
                dim + 1);
        }
    }
}

void debug_print(const std::byte *data,
                 const std::vector<size_t> &shape,
                 const std::vector<ptrdiff_t> &strides,
                 llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(
            reinterpret_cast<const char *>(data),
            shape,
            strides,
            0);

    case LLAISYS_DTYPE_BOOL:
        return print_data(
            reinterpret_cast<const bool *>(data),
            shape,
            strides,
            0);

    case LLAISYS_DTYPE_I8:
        return print_data(
            reinterpret_cast<const int8_t *>(data),
            shape,
            strides,
            0);

    case LLAISYS_DTYPE_I16:
        return print_data(
            reinterpret_cast<const int16_t *>(data),
            shape,
            strides,
            0);

    case LLAISYS_DTYPE_I32:
        return print_data(
            reinterpret_cast<const int32_t *>(data),
            shape,
            strides,
            0);

    case LLAISYS_DTYPE_I64:
        return print_data(
            reinterpret_cast<const int64_t *>(data),
            shape,
            strides,
            0);

    case LLAISYS_DTYPE_U8:
        return print_data(
            reinterpret_cast<const uint8_t *>(data),
            shape,
            strides,
            0);

    case LLAISYS_DTYPE_U16:
        return print_data(
            reinterpret_cast<const uint16_t *>(data),
            shape,
            strides,
            0);

    case LLAISYS_DTYPE_U32:
        return print_data(
            reinterpret_cast<const uint32_t *>(data),
            shape,
            strides,
            0);

    case LLAISYS_DTYPE_U64:
        return print_data(
            reinterpret_cast<const uint64_t *>(data),
            shape,
            strides,
            0);

    case LLAISYS_DTYPE_F16:
        return print_data(
            reinterpret_cast<const fp16_t *>(data),
            shape,
            strides,
            0);

    case LLAISYS_DTYPE_F32:
        return print_data(
            reinterpret_cast<const float *>(data),
            shape,
            strides,
            0);

    case LLAISYS_DTYPE_F64:
        return print_data(
            reinterpret_cast<const double *>(data),
            shape,
            strides,
            0);

    case LLAISYS_DTYPE_BF16:
        return print_data(
            reinterpret_cast<const bf16_t *>(data),
            shape,
            strides,
            0);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(
        this->deviceType(),
        this->deviceId());

    core::context()
        .runtime()
        .api()
        ->device_synchronize();

    std::cout << this->info() << std::endl;

    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(
            this->data(),
            this->shape(),
            this->strides(),
            this->dtype());
    } else {
        auto tmp_tensor =
            create({this->_storage->size()}, this->dtype());

        core::context()
            .runtime()
            .api()
            ->memcpy_sync(
                tmp_tensor->data(),
                this->data(),
                this->numel() * this->elementSize(),
                LLAISYS_MEMCPY_D2H);

        debug_print(
            tmp_tensor->data(),
            this->shape(),
            this->strides(),
            this->dtype());
    }
}

/*
 * Task 1.2
 *
 * 判断 tensor 是否连续。
 *
 * C-style / row-major tensor:
 *
 * shape   = [2, 3, 5]
 * strides = [15, 5, 1]
 *
 * 就是连续的。
 *
 * size == 1 的维度不会真正跨越内存，因此它的 stride
 * 不影响连续性。
 */
bool Tensor::isContiguous() const {
    if (_meta.shape.empty()) {
        return true;
    }

    ptrdiff_t expected_stride = 1;

    for (ptrdiff_t i =
             static_cast<ptrdiff_t>(_meta.shape.size()) - 1;
         i >= 0;
         --i) {

        /*
         * 长度为 1 的维度不会影响实际内存布局。
         *
         * 比如：
         *
         * shape   = [2, 1, 3]
         * strides = [3, 100, 1]
         *
         * 实际数据依然可以是连续的。
         */
        if (_meta.shape[i] == 1) {
            continue;
        }

        if (_meta.strides[i] != expected_stride) {
            return false;
        }

        expected_stride *=
            static_cast<ptrdiff_t>(_meta.shape[i]);
    }

    return true;
}

/*
 * Task 1.4
 *
 * permute 只改变 tensor metadata。
 *
 * 不复制数据。
 *
 * 示例：
 *
 * shape   = [2, 3, 5]
 * strides = [15, 5, 1]
 *
 * order = [1, 0, 2]
 *
 * =>
 *
 * shape   = [3, 2, 5]
 * strides = [5, 15, 1]
 */
tensor_t Tensor::permute(
    const std::vector<size_t> &order) const {

    if (order.size() != ndim()) {
        throw std::invalid_argument(
            "permute: order size must equal tensor ndim");
    }

    std::vector<bool> visited(ndim(), false);

    std::vector<size_t> new_shape(ndim());
    std::vector<ptrdiff_t> new_strides(ndim());

    for (size_t i = 0; i < ndim(); ++i) {
        size_t old_dim = order[i];

        if (old_dim >= ndim()) {
            throw std::invalid_argument(
                "permute: dimension out of range");
        }

        if (visited[old_dim]) {
            throw std::invalid_argument(
                "permute: duplicated dimension");
        }

        visited[old_dim] = true;

        new_shape[i] =
            _meta.shape[old_dim];

        new_strides[i] =
            _meta.strides[old_dim];
    }

    TensorMeta meta{
        _meta.dtype,
        std::move(new_shape),
        std::move(new_strides)
    };

    /*
     * 注意：
     *
     * 必须继续使用原 storage 和原 offset。
     *
     * 因为 permute 本质是 view。
     */
    return std::shared_ptr<Tensor>(
        new Tensor(
            std::move(meta),
            _storage,
            _offset));
}

/*
 * Task 1.3
 *
 * view 不复制数据，只修改 shape 和 strides。
 *
 * 但是只有内存布局兼容时才能 view。
 *
 * 例如：
 *
 * shape   = [2, 3, 5]
 * strides = [15, 5, 1]
 *
 * 可以：
 *
 * [2, 3, 5] -> [2, 15]
 *
 *
 * 但是：
 *
 * shape   = [2, 3, 5]
 * strides = [30, 10, 1]
 *
 * 不能：
 *
 * [2, 3, 5] -> [2, 15]
 *
 * 因为最后两个维度在物理内存中不是连续块：
 *
 * stride[1] = 10
 *
 * 但
 *
 * stride[2] * shape[2]
 * = 1 * 5
 * = 5
 */
tensor_t Tensor::view(
    const std::vector<size_t> &shape) const {

    /*
     * 1. 检查元素数量。
     */
    size_t new_numel = std::accumulate(
        shape.begin(),
        shape.end(),
        size_t(1),
        std::multiplies<size_t>());

    if (new_numel != numel()) {
        throw std::invalid_argument(
            "view: number of elements does not match");
    }

    /*
     * scalar tensor。
     */
    if (shape.empty()) {
        if (numel() != 1) {
            throw std::invalid_argument(
                "view: only one-element tensor "
                "can be viewed as scalar");
        }

        TensorMeta meta{
            _meta.dtype,
            {},
            {}
        };

        return std::shared_ptr<Tensor>(
            new Tensor(
                std::move(meta),
                _storage,
                _offset));
    }

    /*
     * 对于 0 元素 Tensor，
     * 没有实际数据需要保持，因此直接建立标准 contiguous stride。
     */
    if (numel() == 0) {
        std::vector<ptrdiff_t> new_strides(shape.size());

        ptrdiff_t stride = 1;

        for (ptrdiff_t i =
                 static_cast<ptrdiff_t>(shape.size()) - 1;
             i >= 0;
             --i) {

            new_strides[i] = stride;

            stride *=
                static_cast<ptrdiff_t>(shape[i]);
        }

        TensorMeta meta{
            _meta.dtype,
            shape,
            std::move(new_strides)
        };

        return std::shared_ptr<Tensor>(
            new Tensor(
                std::move(meta),
                _storage,
                _offset));
    }

    /*
     * 如果原 tensor 是 scalar，
     * 元素数量只能是 1。
     *
     * 这种情况下新的所有维度都只能乘起来等于 1，
     * 直接生成 contiguous stride 即可。
     */
    if (_meta.shape.empty()) {
        std::vector<ptrdiff_t> new_strides(shape.size());

        ptrdiff_t stride = 1;

        for (ptrdiff_t i =
                 static_cast<ptrdiff_t>(shape.size()) - 1;
             i >= 0;
             --i) {

            new_strides[i] = stride;

            stride *=
                static_cast<ptrdiff_t>(shape[i]);
        }

        TensorMeta meta{
            _meta.dtype,
            shape,
            std::move(new_strides)
        };

        return std::shared_ptr<Tensor>(
            new Tensor(
                std::move(meta),
                _storage,
                _offset));
    }

    /*
     * 2. 根据原来的 shape + stride 分析“连续 chunk”。
     *
     * 这是 view 最重要的部分。
     *
     * 一个 chunk 内的维度可以自由拆分/合并。
     *
     * chunk 之间不能跨越。
     */
    std::vector<ptrdiff_t> new_strides(
        shape.size(),
        0);

    ptrdiff_t view_dim =
        static_cast<ptrdiff_t>(shape.size()) - 1;

    size_t chunk_numel = 1;
    size_t view_numel = 1;

    ptrdiff_t chunk_base_stride =
        _meta.strides.back();

    /*
     * 从最后一个原始维度开始向前扫描。
     */
    for (ptrdiff_t tensor_dim =
             static_cast<ptrdiff_t>(_meta.shape.size()) - 1;
         tensor_dim >= 0;
         --tensor_dim) {

        chunk_numel *=
            _meta.shape[tensor_dim];

        /*
         * 判断当前连续 chunk 是否结束。
         *
         * chunk 在以下情况结束：
         *
         * 1. 已经到第 0 维；
         *
         * 2. 前一个维度不是 size=1，并且 stride 关系不满足：
         *
         * old_stride[d - 1]
         * !=
         * chunk_numel * chunk_base_stride
         */
        bool end_of_chunk = false;

        if (tensor_dim == 0) {
            end_of_chunk = true;
        } else {
            size_t previous_size =
                _meta.shape[tensor_dim - 1];

            ptrdiff_t previous_stride =
                _meta.strides[tensor_dim - 1];

            if (previous_size != 1 &&
                previous_stride !=
                    static_cast<ptrdiff_t>(chunk_numel) *
                        chunk_base_stride) {
                end_of_chunk = true;
            }
        }

        if (!end_of_chunk) {
            continue;
        }

        /*
         * 用 new shape 的若干维度，
         * 去填充当前这个连续 chunk。
         */
        while (view_dim >= 0 &&
               (view_numel < chunk_numel ||
                shape[view_dim] == 1)) {

            new_strides[view_dim] =
                static_cast<ptrdiff_t>(view_numel) *
                chunk_base_stride;

            view_numel *=
                shape[view_dim];

            --view_dim;
        }

        /*
         * 必须刚好把 chunk 填满。
         *
         * 如果填不满/超过，说明 view 不兼容。
         */
        if (view_numel != chunk_numel) {
            throw std::invalid_argument(
                "view: requested shape is incompatible "
                "with tensor strides");
        }

        /*
         * 开始分析下一个 chunk。
         */
        if (tensor_dim > 0) {
            chunk_base_stride =
                _meta.strides[tensor_dim - 1];

            chunk_numel = 1;
            view_numel = 1;
        }
    }

    /*
     * 理论上所有新维度都必须已经匹配。
     *
     * 剩余的维度如果存在，只能是 size == 1。
     */
    while (view_dim >= 0) {
        if (shape[view_dim] != 1) {
            throw std::invalid_argument(
                "view: requested shape is incompatible "
                "with tensor strides");
        }

        ptrdiff_t stride = 1;

        if (view_dim + 1 <
            static_cast<ptrdiff_t>(shape.size())) {

            stride =
                new_strides[view_dim + 1] *
                static_cast<ptrdiff_t>(
                    shape[view_dim + 1]);
        }

        new_strides[view_dim] = stride;

        --view_dim;
    }

    TensorMeta meta{
        _meta.dtype,
        shape,
        std::move(new_strides)
    };

    return std::shared_ptr<Tensor>(
        new Tensor(
            std::move(meta),
            _storage,
            _offset));
}

/*
 * Task 1.5
 *
 * slice 也是零拷贝操作。
 *
 * 只需要：
 *
 * 1. 修改 shape
 * 2. 保留 stride
 * 3. 修改 offset
 *
 * offset 单位是 byte。
 */
tensor_t Tensor::slice(
    size_t dim,
    size_t start,
    size_t end) const {

    if (dim >= ndim()) {
        throw std::invalid_argument(
            "slice: dimension out of range");
    }

    if (start > end) {
        throw std::invalid_argument(
            "slice: start must be <= end");
    }

    if (end > _meta.shape[dim]) {
        throw std::invalid_argument(
            "slice: end exceeds dimension size");
    }

    std::vector<size_t> new_shape =
        _meta.shape;

    new_shape[dim] =
        end - start;

    /*
     * slice 不改变 stride。
     */
    TensorMeta meta{
        _meta.dtype,
        std::move(new_shape),
        _meta.strides
    };

    /*
     * stride 的单位是 element。
     *
     * _offset 的单位是 byte。
     *
     * 所以：
     *
     * byte offset =
     *
     * start * stride * elementSize()
     */
    size_t offset_delta =
        start *
        static_cast<size_t>(_meta.strides[dim]) *
        elementSize();

    /*
     * 注意一定是：
     *
     * 原 offset + 新 offset
     *
     * 这样才能正确支持：
     *
     * tensor.slice(...).slice(...)
     */
    return std::shared_ptr<Tensor>(
        new Tensor(
            std::move(meta),
            _storage,
            _offset + offset_delta));
}

/*
 * Task 1.1
 *
 * 从 host 内存加载数据。
 *
 * src_ 永远是 CPU pointer。
 *
 * Tensor 自己可能位于 CPU，也可能位于设备上。
 */
void Tensor::load(const void *src_) {
    if (src_ == nullptr) {
        throw std::invalid_argument(
            "load: source pointer is nullptr");
    }

    /*
     * load 本质是连续内存 memcpy。
     *
     * 对非连续 view 直接 memcpy 会破坏 tensor 语义，
     * 所以这里要求 tensor contiguous。
     */
    if (!isContiguous()) {
        throw std::runtime_error(
            "load: loading into a non-contiguous tensor "
            "is not supported");
    }

    /*
     * 切换到 Tensor 所属设备的 runtime。
     */
    core::context().setDevice(
        this->deviceType(),
        this->deviceId());

    size_t bytes =
        this->numel() *
        this->elementSize();

    /*
     * src_ 在 host。
     *
     * destination 是当前 tensor。
     *
     * 因此使用 H2D。
     *
     * CPU runtime 下这个接口同样负责普通 host memcpy。
     */
    core::context()
        .runtime()
        .api()
        ->memcpy_sync(
            this->data(),
            src_,
            bytes,
            LLAISYS_MEMCPY_H2D);
}

/*
 * 后续挑战任务。
 *
 * 当前作业 #1 不实现。
 */
tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();

    return std::shared_ptr<Tensor>(
        new Tensor(
            _meta,
            _storage));
}

tensor_t Tensor::reshape(
    const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();

    return std::shared_ptr<Tensor>(
        new Tensor(
            _meta,
            _storage));
}

tensor_t Tensor::to(
    llaisysDeviceType_t device_type,
    int device) const {
    TO_BE_IMPLEMENTED();

    return std::shared_ptr<Tensor>(
        new Tensor(
            _meta,
            _storage));
}

} // namespace llaisys