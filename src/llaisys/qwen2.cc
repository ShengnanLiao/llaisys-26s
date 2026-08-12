#include "../../include/llaisys/models/qwen2.h"
#include "../../include/llaisys/ops.h"
#include "../../include/llaisys/runtime.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
static constexpr size_t QWEN2_CACHE_CAPACITY = 4096;

static size_t qwen2_dtype_size(llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F16:
        return 2;
    case LLAISYS_DTYPE_BF16:
        return 2;
    case LLAISYS_DTYPE_F32:
        return 4;
    default:
        throw std::runtime_error(
            "Qwen2: unsupported model dtype");
    }
}
/*
 * Qwen2 模型在 C++ 后端中的实际结构。
 *
 * include/llaisys/models/qwen2.h 中只有：
 *
 *      struct LlaisysQwen2Model;
 *
 * 是一个前向声明。
 *
 * 真正的数据结构放在这里，对 Python/C API 隐藏。
 */
struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    LlaisysQwen2Weights weights;
    llaisysDeviceType_t device;
    int device_id;
    // 已经写入 KV Cache 的 token 数
    size_t past_len;
    // 实际分配的 KV Cache 长度
    size_t cache_capacity;
    // 每层一个 K Cache / V Cache
    // shape:
    // [cache_capacity, nkvh, dh]
    llaisysTensor_t *k_cache;
    llaisysTensor_t *v_cache;
};
/*
 * ============================================================
 * 辅助函数：创建 Tensor
 * ============================================================
 */
static llaisysTensor_t create_tensor(
    const std::vector<size_t> &shape,
    llaisysDataType_t dtype,
    llaisysDeviceType_t device,
    int device_id) {
    /*
     * tensorCreate 接收 size_t *，
     * 因此这里把 vector.data() 传进去。
     */
    return tensorCreate(
        const_cast<size_t *>(shape.data()),
        shape.size(),
        dtype,
        device,
        device_id);
}
/*
 * ============================================================
 * 辅助函数：安全销毁单个 Tensor
 * ============================================================
 */
static void destroy_tensor(llaisysTensor_t &tensor) {
    if (tensor != nullptr) {
        tensorDestroy(tensor);
        tensor = nullptr;
    }
}
/*
 * ============================================================
 * 辅助函数：销毁 Tensor 数组中的所有 Tensor
 * ============================================================
 */
static void destroy_tensor_array(
    llaisysTensor_t *&array,
    size_t n) {
    if (array == nullptr) {
        return;
    }
    for (size_t i = 0; i < n; ++i) {
        if (array[i] != nullptr) {
            tensorDestroy(array[i]);
            array[i] = nullptr;
        }
    }
    delete[] array;
    array = nullptr;
}
/*
 * ============================================================
 * C API
 * ============================================================
 */
__C {
/*
 * 创建 Qwen2 模型。
 *
 * 当前阶段主要做：
 *
 * 1. 保存模型 Meta
 * 2. 创建所有模型权重 Tensor
 * 3. 初始化模型状态
 *
 * 注意：
 *
 * 这里只“分配”权重空间。
 *
 * 真正 safetensors 中的数据，
 * 后面由 Python 通过 tensorLoad() 加载。
 */
struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
    const LlaisysQwen2Meta *meta,
    llaisysDeviceType_t device,
    int *device_ids,
    int ndevice) {
    if (meta == nullptr) {
        throw std::invalid_argument(
            "llaisysQwen2ModelCreate: meta is nullptr");
    }
    if (ndevice <= 0 || device_ids == nullptr) {
        throw std::invalid_argument(
            "llaisysQwen2ModelCreate: invalid device configuration");
    }
    /*
     * 当前先只支持单设备。
     *
     * 你的作业现在跑 CPU，
     * 后面如果需要多卡再扩展。
     */
    if (ndevice != 1) {
        throw std::invalid_argument(
            "llaisysQwen2ModelCreate: only one device is supported for now");
    }
    /*
     * Qwen2 中：
     *
     * hidden_size = num_attention_heads * head_dim
     */
    if (meta->nh * meta->dh != meta->hs) {
        throw std::invalid_argument(
            "llaisysQwen2ModelCreate: nh * dh must equal hs");
    }
    /*
     * 创建模型对象。
     *
     * {} 可以保证 weights 内所有指针初始为 nullptr，
     * 这样 Destroy 更安全。
     */
    auto *model = new LlaisysQwen2Model{};
    model->meta = *meta;
    model->device = device;
    model->device_id = device_ids[0];
    model->past_len = 0;
    model->cache_capacity =
    std::min(
        static_cast<size_t>(QWEN2_CACHE_CAPACITY),
        meta->maxseq);
    auto &w = model->weights;
    const size_t nlayer = meta->nlayer;
    const size_t hs = meta->hs;
    /*
     * Q projection 输出维度：
     *
     * nh * dh
     *
     * 对你的模型：
     *
     * 12 * 128 = 1536
     */
    const size_t q_size =
        meta->nh * meta->dh;
    /*
     * K/V projection 输出维度：
     *
     * nkvh * dh
     *
     * 对你的模型：
     *
     * 2 * 128 = 256
     */
    const size_t kv_size =
        meta->nkvh * meta->dh;
    /*
     * ========================================================
     * 1. 全局权重
     * ========================================================
     */
    /*
     * model.embed_tokens.weight
     *
     * [vocab_size, hidden_size]
     *
     * 你的模型：
     *
     * [151936, 1536]
     */
    w.in_embed = create_tensor(
        {meta->voc, hs},
        meta->dtype,
        device,
        model->device_id);
    /*
     * lm_head.weight
     *
     * [vocab_size, hidden_size]
     *
     * 注意：
     *
     * 你的模型 tie_word_embeddings=false，
     * 因此 lm_head.weight 和 embedding 是独立权重。
     */
    w.out_embed = create_tensor(
        {meta->voc, hs},
        meta->dtype,
        device,
        model->device_id);
    /*
     * model.norm.weight
     *
     * [hidden_size]
     */
    w.out_norm_w = create_tensor(
        {hs},
        meta->dtype,
        device,
        model->device_id);
    /*
     * ========================================================
     * 2. 为每层权重创建指针数组
     * ========================================================
     *
     * 使用 {} 初始化为 nullptr。
     */
    w.attn_norm_w =
        new llaisysTensor_t[nlayer]{};
    w.attn_q_w =
        new llaisysTensor_t[nlayer]{};
    w.attn_q_b =
        new llaisysTensor_t[nlayer]{};
    w.attn_k_w =
        new llaisysTensor_t[nlayer]{};
    w.attn_k_b =
        new llaisysTensor_t[nlayer]{};
    w.attn_v_w =
        new llaisysTensor_t[nlayer]{};
    w.attn_v_b =
        new llaisysTensor_t[nlayer]{};
    w.attn_o_w =
        new llaisysTensor_t[nlayer]{};
    w.mlp_norm_w =
        new llaisysTensor_t[nlayer]{};
    w.mlp_gate_w =
        new llaisysTensor_t[nlayer]{};
    w.mlp_up_w =
        new llaisysTensor_t[nlayer]{};
    w.mlp_down_w =
        new llaisysTensor_t[nlayer]{};
    /*
     * ========================================================
     * 3. 创建每一层的具体权重
     * ========================================================
     */
    for (size_t i = 0; i < nlayer; ++i) {
        /*
         * model.layers.i.input_layernorm.weight
         *
         * [hidden_size]
         */
        w.attn_norm_w[i] =
            create_tensor(
                {hs},
                meta->dtype,
                device,
                model->device_id);
        /*
         * self_attn.q_proj.weight
         *
         * [nh * dh, hidden_size]
         *
         * 你的模型：
         *
         * [1536, 1536]
         */
        w.attn_q_w[i] =
            create_tensor(
                {q_size, hs},
                meta->dtype,
                device,
                model->device_id);
        /*
         * self_attn.q_proj.bias
         *
         * [nh * dh]
         *
         * [1536]
         */
        w.attn_q_b[i] =
            create_tensor(
                {q_size},
                meta->dtype,
                device,
                model->device_id);
        /*
         * self_attn.k_proj.weight
         *
         * [nkvh * dh, hidden_size]
         *
         * [256, 1536]
         */
        w.attn_k_w[i] =
            create_tensor(
                {kv_size, hs},
                meta->dtype,
                device,
                model->device_id);
        /*
         * self_attn.k_proj.bias
         *
         * [256]
         */
        w.attn_k_b[i] =
            create_tensor(
                {kv_size},
                meta->dtype,
                device,
                model->device_id);
        /*
         * self_attn.v_proj.weight
         *
         * [256, 1536]
         */
        w.attn_v_w[i] =
            create_tensor(
                {kv_size, hs},
                meta->dtype,
                device,
                model->device_id);
        /*
         * self_attn.v_proj.bias
         *
         * [256]
         */
        w.attn_v_b[i] =
            create_tensor(
                {kv_size},
                meta->dtype,
                device,
                model->device_id);
        /*
         * self_attn.o_proj.weight
         *
         * [hidden_size, hidden_size]
         *
         * [1536, 1536]
         */
        w.attn_o_w[i] =
            create_tensor(
                {hs, q_size},
                meta->dtype,
                device,
                model->device_id);
        /*
         * model.layers.i.post_attention_layernorm.weight
         *
         * [hidden_size]
         */
        w.mlp_norm_w[i] =
            create_tensor(
                {hs},
                meta->dtype,
                device,
                model->device_id);
        /*
         * mlp.gate_proj.weight
         *
         * [intermediate_size, hidden_size]
         *
         * [8960, 1536]
         */
        w.mlp_gate_w[i] =
            create_tensor(
                {meta->di, hs},
                meta->dtype,
                device,
                model->device_id);
        /*
         * mlp.up_proj.weight
         *
         * [8960, 1536]
         */
        w.mlp_up_w[i] =
            create_tensor(
                {meta->di, hs},
                meta->dtype,
                device,
                model->device_id);
        /*
         * mlp.down_proj.weight
         *
         * [hidden_size, intermediate_size]
         *
         * [1536, 8960]
         */
        w.mlp_down_w[i] =
            create_tensor(
                {hs, meta->di},
                meta->dtype,
                device,
                model->device_id);
    }
    /*
     * ========================================================
     * 4. 创建 KV Cache
     * ========================================================
     *
     * 每层：
     *   K Cache [cache_capacity, nkvh, dh]
     *   V Cache [cache_capacity, nkvh, dh]
     *
     * 当前先固定最多缓存 4096 token（同时不超过 maxseq）。
     */
    model->k_cache =
        new llaisysTensor_t[nlayer]{};

    model->v_cache =
        new llaisysTensor_t[nlayer]{};

    for (size_t i = 0; i < nlayer; ++i) {
        model->k_cache[i] =
            create_tensor(
                {
                    model->cache_capacity,
                    meta->nkvh,
                    meta->dh
                },
                meta->dtype,
                device,
                model->device_id);

        model->v_cache[i] =
            create_tensor(
                {
                    model->cache_capacity,
                    meta->nkvh,
                    meta->dh
                },
                meta->dtype,
                device,
                model->device_id);
    }

    return model;
}
/*
 * ============================================================
 * 销毁 Qwen2 模型
 * ============================================================
 */
void llaisysQwen2ModelDestroy(
    struct LlaisysQwen2Model *model) {
    if (model == nullptr) {
        return;
    }
    auto &w = model->weights;
    /*
     * 全局权重
     */
    destroy_tensor(w.in_embed);
    destroy_tensor(w.out_embed);
    destroy_tensor(w.out_norm_w);
    /*
     * 每层权重
     */
    const size_t nlayer =
        model->meta.nlayer;
    destroy_tensor_array(
        w.attn_norm_w,
        nlayer);
    destroy_tensor_array(
        w.attn_q_w,
        nlayer);
    destroy_tensor_array(
        w.attn_q_b,
        nlayer);
    destroy_tensor_array(
        w.attn_k_w,
        nlayer);
    destroy_tensor_array(
        w.attn_k_b,
        nlayer);
    destroy_tensor_array(
        w.attn_v_w,
        nlayer);
    destroy_tensor_array(
        w.attn_v_b,
        nlayer);
    destroy_tensor_array(
        w.attn_o_w,
        nlayer);
    destroy_tensor_array(
        w.mlp_norm_w,
        nlayer);
    destroy_tensor_array(
        w.mlp_gate_w,
        nlayer);
    destroy_tensor_array(
        w.mlp_up_w,
        nlayer);
    destroy_tensor_array(
        w.mlp_down_w,
        nlayer);
    /*
     * KV Cache
     */
    destroy_tensor_array(
        model->k_cache,
        nlayer);

    destroy_tensor_array(
        model->v_cache,
        nlayer);

    delete model;
}
/*
 * ============================================================
 * 获取模型权重结构
 * ============================================================
 *
 * Python 调用这个函数以后，
 * 可以拿到每个 Tensor，
 * 然后通过 tensorLoad() 加载 safetensors 数据。
 */
struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(
    struct LlaisysQwen2Model *model) {
    if (model == nullptr) {
        return nullptr;
    }
    return &model->weights;
}
/*
 * ============================================================
 * 模型推理
 * ============================================================
 *
 * 下一阶段再实现。
 */
int64_t llaisysQwen2ModelInfer(
    struct LlaisysQwen2Model *model,
    int64_t *token_ids,
    size_t ntoken) {
    if (model == nullptr) {
        throw std::invalid_argument(
            "llaisysQwen2ModelInfer: model is nullptr");
    }
    if (token_ids == nullptr) {
        throw std::invalid_argument(
            "llaisysQwen2ModelInfer: token_ids is nullptr");
    }
    if (ntoken == 0) {
        throw std::invalid_argument(
            "llaisysQwen2ModelInfer: ntoken must be > 0");
    }
    const auto &meta = model->meta;
    auto &w = model->weights;

    const size_t N = ntoken;
    const size_t total_len = model->past_len + N;

    if (total_len > model->cache_capacity) {
        throw std::runtime_error(
            "llaisysQwen2ModelInfer: KV cache capacity exceeded");
    }

    const size_t HS = meta.hs;
    const size_t NH = meta.nh;
    const size_t NKVH = meta.nkvh;
    const size_t DH = meta.dh;
    const size_t DI = meta.di;
    const size_t VOC = meta.voc;
    const size_t QSIZE = NH * DH;
    const size_t KVSIZE = NKVH * DH;
    const auto dtype = meta.dtype;
    const auto device = model->device;
    const int device_id = model->device_id;

    const LlaisysRuntimeAPI *runtime =
        llaisysGetRuntimeAPI(device);

    if (runtime == nullptr) {
        throw std::runtime_error(
            "llaisysQwen2ModelInfer: failed to get runtime API");
    }
    /*
     * --------------------------------------------------------
     * 1. token ids
     * --------------------------------------------------------
     *
     * shape: [N]
     * dtype: I64
     */
    size_t token_shape[] = {N};
    llaisysTensor_t tokens =
        tensorCreate(
            token_shape,
            1,
            LLAISYS_DTYPE_I64,
            device,
            device_id);
    tensorLoad(
        tokens,
        token_ids);
    /*
     * --------------------------------------------------------
     * 2. position ids
     * --------------------------------------------------------
     *
     * 第一版不使用 KV Cache，
     * 每次都是从 position 0 开始重新计算。
     *
     * [0, 1, 2, ..., N-1]
     */
    std::vector<int64_t> pos_data(N);
    for (size_t i = 0; i < N; ++i) {
        pos_data[i] =
            static_cast<int64_t>(
                model->past_len + i);
    }
    llaisysTensor_t pos_ids =
        tensorCreate(
            token_shape,
            1,
            LLAISYS_DTYPE_I64,
            device,
            device_id);
    tensorLoad(
        pos_ids,
        pos_data.data());
    /*
     * --------------------------------------------------------
     * 3. Embedding
     * --------------------------------------------------------
     *
     * [N]
     *   ↓
     * [N, HS]
     */
    size_t hidden_shape[] = {
        N,
        HS
    };
    llaisysTensor_t hidden =
        tensorCreate(
            hidden_shape,
            2,
            dtype,
            device,
            device_id);
    llaisysEmbedding(
        hidden,
        tokens,
        w.in_embed);
    /*
     * ========================================================
     * Transformer Layers
     * ========================================================
     */
    for (size_t layer = 0;
         layer < meta.nlayer;
         ++layer) {
        /*
         * ----------------------------------------------------
         * Attention RMSNorm
         * ----------------------------------------------------
         */
        llaisysTensor_t norm1 =
            tensorCreate(
                hidden_shape,
                2,
                dtype,
                device,
                device_id);
        llaisysRmsNorm(
            norm1,
            hidden,
            w.attn_norm_w[layer],
            meta.epsilon);
        /*
         * ----------------------------------------------------
         * Q projection
         *
         * [N, HS]
         *   ↓
         * [N, NH * DH]
         * ----------------------------------------------------
         */
        size_t q2_shape[] = {
            N,
            QSIZE
        };
        llaisysTensor_t q2 =
            tensorCreate(
                q2_shape,
                2,
                dtype,
                device,
                device_id);
        llaisysLinear(
            q2,
            norm1,
            w.attn_q_w[layer],
            w.attn_q_b[layer]);
        /*
         * ----------------------------------------------------
         * K projection
         *
         * [N, HS]
         *   ↓
         * [N, NKVH * DH]
         * ----------------------------------------------------
         */
        size_t kv2_shape[] = {
            N,
            KVSIZE
        };
        llaisysTensor_t k2 =
            tensorCreate(
                kv2_shape,
                2,
                dtype,
                device,
                device_id);
        llaisysLinear(
            k2,
            norm1,
            w.attn_k_w[layer],
            w.attn_k_b[layer]);
        /*
         * ----------------------------------------------------
         * V projection
         * ----------------------------------------------------
         */
        llaisysTensor_t v2 =
            tensorCreate(
                kv2_shape,
                2,
                dtype,
                device,
                device_id);
        llaisysLinear(
            v2,
            norm1,
            w.attn_v_w[layer],
            w.attn_v_b[layer]);
        /*
         * ----------------------------------------------------
         * View Q
         *
         * [N, QSIZE]
         * ->
         * [N, NH, DH]
         * ----------------------------------------------------
         */
        size_t q3_shape[] = {
            N,
            NH,
            DH
        };
        llaisysTensor_t q =
            tensorView(
                q2,
                q3_shape,
                3);
        /*
         * ----------------------------------------------------
         * View K/V
         *
         * [N, KVSIZE]
         * ->
         * [N, NKVH, DH]
         * ----------------------------------------------------
         */
        size_t kv3_shape[] = {
            N,
            NKVH,
            DH
        };
        llaisysTensor_t k =
            tensorView(
                k2,
                kv3_shape,
                3);
        llaisysTensor_t v =
            tensorView(
                v2,
                kv3_shape,
                3);
        /*
         * ----------------------------------------------------
         * RoPE
         *
         * 注意不能原地调用。
         * 单独创建输出 Tensor 更安全。
         * ----------------------------------------------------
         */
        llaisysTensor_t q_rope =
            tensorCreate(
                q3_shape,
                3,
                dtype,
                device,
                device_id);
        llaisysTensor_t k_rope =
            tensorCreate(
                kv3_shape,
                3,
                dtype,
                device,
                device_id);
        llaisysROPE(
            q_rope,
            q,
            pos_ids,
            meta.theta);
        llaisysROPE(
            k_rope,
            k,
            pos_ids,
            meta.theta);

        /*
         * ----------------------------------------------------
         * 将本次新产生的 K/V 写入 KV Cache
         *
         * K Cache 保存 RoPE 后的 K。
         * V Cache 保存原始 V。
         *
         * 写入区间：
         * [past_len, past_len + N)
         * ----------------------------------------------------
         */
        const size_t element_size =
            qwen2_dtype_size(dtype);

        const size_t one_token_kv_bytes =
            NKVH * DH * element_size;

        const size_t new_kv_bytes =
            N * one_token_kv_bytes;

        auto *k_cache_ptr =
            reinterpret_cast<std::byte *>(
                tensorGetData(
                    model->k_cache[layer]));

        auto *v_cache_ptr =
            reinterpret_cast<std::byte *>(
                tensorGetData(
                    model->v_cache[layer]));

        const auto *new_k_ptr =
            reinterpret_cast<const std::byte *>(
                tensorGetData(k_rope));

        const auto *new_v_ptr =
            reinterpret_cast<const std::byte *>(
                tensorGetData(v));

        const llaisysMemcpyKind_t kv_copy_kind =
            (device == LLAISYS_DEVICE_CPU)
                ? LLAISYS_MEMCPY_H2H
                : LLAISYS_MEMCPY_D2D;

        runtime->memcpy_sync(
            k_cache_ptr +
                model->past_len * one_token_kv_bytes,
            new_k_ptr,
            new_kv_bytes,
            kv_copy_kind);

        runtime->memcpy_sync(
            v_cache_ptr +
                model->past_len * one_token_kv_bytes,
            new_v_ptr,
            new_kv_bytes,
            kv_copy_kind);

        /*
         * 当前层 Attention 使用：
         *
         * K/V:
         * [0, total_len)
         */
        llaisysTensor_t cached_k =
            tensorSlice(
                model->k_cache[layer],
                0,
                0,
                total_len);

        llaisysTensor_t cached_v =
            tensorSlice(
                model->v_cache[layer],
                0,
                0,
                total_len);

        /*
         * ----------------------------------------------------
         * Self Attention
         *
         * Q:
         * [N, NH, DH]
         *
         * Cached K/V:
         * [total_len, NKVH, DH]
         *
         * output:
         * [N, NH, DH]
         * ----------------------------------------------------
         */
        llaisysTensor_t attn3 =
            tensorCreate(
                q3_shape,
                3,
                dtype,
                device,
                device_id);

        const float scale =
            1.0f /
            std::sqrt(
                static_cast<float>(DH));

        llaisysSelfAttention(
            attn3,
            q_rope,
            cached_k,
            cached_v,
            scale);
        /*
         * Attention output:
         *
         * [N, NH, DH]
         * ->
         * [N, HS]
         */
        llaisysTensor_t attn2 =
            tensorView(
                attn3,
                hidden_shape,
                2);
        /*
         * ----------------------------------------------------
         * O projection
         * ----------------------------------------------------
         */
        llaisysTensor_t attn_out =
            tensorCreate(
                hidden_shape,
                2,
                dtype,
                device,
                device_id);
        llaisysLinear(
            attn_out,
            attn2,
            w.attn_o_w[layer],
            nullptr);
        /*
         * ----------------------------------------------------
         * First residual
         *
         * hidden =
         * hidden + attention_out
         * ----------------------------------------------------
         */
        llaisysTensor_t hidden_after_attn =
            tensorCreate(
                hidden_shape,
                2,
                dtype,
                device,
                device_id);
        llaisysAdd(
            hidden_after_attn,
            hidden,
            attn_out);
        /*
         * ----------------------------------------------------
         * MLP RMSNorm
         * ----------------------------------------------------
         */
        llaisysTensor_t norm2 =
            tensorCreate(
                hidden_shape,
                2,
                dtype,
                device,
                device_id);
        llaisysRmsNorm(
            norm2,
            hidden_after_attn,
            w.mlp_norm_w[layer],
            meta.epsilon);
        /*
         * ----------------------------------------------------
         * gate_proj / up_proj
         *
         * [N, HS]
         * ->
         * [N, DI]
         * ----------------------------------------------------
         */
        size_t intermediate_shape[] = {
            N,
            DI
        };
        llaisysTensor_t gate =
            tensorCreate(
                intermediate_shape,
                2,
                dtype,
                device,
                device_id);
        llaisysTensor_t up =
            tensorCreate(
                intermediate_shape,
                2,
                dtype,
                device,
                device_id);
        llaisysLinear(
            gate,
            norm2,
            w.mlp_gate_w[layer],
            nullptr);
        llaisysLinear(
            up,
            norm2,
            w.mlp_up_w[layer],
            nullptr);
        /*
         * ----------------------------------------------------
         * SwiGLU
         * ----------------------------------------------------
         */
        llaisysTensor_t swiglu =
            tensorCreate(
                intermediate_shape,
                2,
                dtype,
                device,
                device_id);
        llaisysSwiGLU(
            swiglu,
            gate,
            up);
        /*
         * ----------------------------------------------------
         * down_proj
         *
         * [N, DI]
         * ->
         * [N, HS]
         * ----------------------------------------------------
         */
        llaisysTensor_t down =
            tensorCreate(
                hidden_shape,
                2,
                dtype,
                device,
                device_id);
        llaisysLinear(
            down,
            swiglu,
            w.mlp_down_w[layer],
            nullptr);
        /*
         * ----------------------------------------------------
         * Second residual
         * ----------------------------------------------------
         */
        llaisysTensor_t new_hidden =
            tensorCreate(
                hidden_shape,
                2,
                dtype,
                device,
                device_id);
        llaisysAdd(
            new_hidden,
            hidden_after_attn,
            down);
        /*
         * ----------------------------------------------------
         * 当前层结束：
         *
         * new_hidden 成为下一层输入。
         * ----------------------------------------------------
         */
        tensorDestroy(hidden);
        hidden = new_hidden;
        /*
         * ----------------------------------------------------
         * 释放当前层临时 Tensor
         * ----------------------------------------------------
         *
         * 注意 view 对象本身需要 destroy，
         * 但底层 storage 是 shared_ptr，
         * 所以不会把 q2/k2/v2 的 storage 提前释放掉。
         */
        tensorDestroy(norm1);
        tensorDestroy(q);
        tensorDestroy(k);
        tensorDestroy(v);
        tensorDestroy(q2);
        tensorDestroy(k2);
        tensorDestroy(v2);
        tensorDestroy(q_rope);
        tensorDestroy(k_rope);
        tensorDestroy(cached_k);
        tensorDestroy(cached_v);
        tensorDestroy(attn2);
        tensorDestroy(attn3);
        tensorDestroy(attn_out);
        tensorDestroy(hidden_after_attn);
        tensorDestroy(norm2);
        tensorDestroy(gate);
        tensorDestroy(up);
        tensorDestroy(swiglu);
        tensorDestroy(down);
    }
    /*
     * ========================================================
     * Final RMSNorm
     * ========================================================
     */
    llaisysTensor_t final_norm =
        tensorCreate(
            hidden_shape,
            2,
            dtype,
            device,
            device_id);
    llaisysRmsNorm(
        final_norm,
        hidden,
        w.out_norm_w,
        meta.epsilon);
    /*
     * ========================================================
     * 只取最后一个 token
     * ========================================================
     *
     * final_norm:
     *
     * [N, HS]
     *
     * slice dim 0:
     *
     * [1, HS]
     */
    llaisysTensor_t last_hidden =
        tensorSlice(
            final_norm,
            0,
            N - 1,
            N);
    /*
     * ========================================================
     * LM Head
     *
     * [1, HS]
     *   ×
     * [VOC, HS]^T
     *
     * ->
     *
     * [1, VOC]
     * ========================================================
     */
    size_t logits2_shape[] = {
        1,
        VOC
    };
    llaisysTensor_t logits2 =
        tensorCreate(
            logits2_shape,
            2,
            dtype,
            device,
            device_id);
    llaisysLinear(
        logits2,
        last_hidden,
        w.out_embed,
        nullptr);
    /*
     * Argmax 当前要求 vals 是 1-D，
     * 所以：
     *
     * [1, VOC]
     * ->
     * [VOC]
     */
    size_t logits1_shape[] = {
        VOC
    };
    llaisysTensor_t logits =
        tensorView(
            logits2,
            logits1_shape,
            1);
    /*
     * max_idx:
     * [1] I64
     *
     * max_val:
     * [1] model dtype
     */
    size_t one_shape[] = {
        1
    };
    llaisysTensor_t max_idx =
        tensorCreate(
            one_shape,
            1,
            LLAISYS_DTYPE_I64,
            device,
            device_id);
    llaisysTensor_t max_val =
        tensorCreate(
            one_shape,
            1,
            dtype,
            device,
            device_id);
    llaisysArgmax(
        max_idx,
        max_val,
        logits);
    int64_t next_token = 0;

    const llaisysMemcpyKind_t token_copy_kind =
        (device == LLAISYS_DEVICE_CPU)
            ? LLAISYS_MEMCPY_H2H
            : LLAISYS_MEMCPY_D2H;

    runtime->memcpy_sync(
        &next_token,
        tensorGetData(max_idx),
        sizeof(next_token),
        token_copy_kind);
    /*
     * ========================================================
     * Cleanup
     * ========================================================
     */
    tensorDestroy(tokens);
    tensorDestroy(pos_ids);
    tensorDestroy(hidden);
    tensorDestroy(final_norm);
    tensorDestroy(last_hidden);
    tensorDestroy(logits);
    tensorDestroy(logits2);
    tensorDestroy(max_idx);
    tensorDestroy(max_val);

    /*
     * 本次 Infer 的所有层都已经完成，
     * 现在才能统一推进 KV Cache 的有效长度。
     */
    model->past_len = total_len;

    return next_token;
}
} // __C