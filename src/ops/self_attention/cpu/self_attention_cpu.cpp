#include "self_attention_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <vector>

template <typename T>
void self_attention_(
    T *attn_val,
    const T *q,
    const T *k,
    const T *v,
    float scale,
    size_t seq_len,
    size_t total_len,
    size_t nhead,
    size_t nkvhead,
    size_t d,
    size_t dv,
    float *scores_buf)
{
    const size_t past_len = total_len - seq_len;
    std::vector<float> out_buf(dv);

    for (size_t i = 0; i < seq_len; ++i)
    {
        for (size_t h = 0; h < nhead; ++h)
        {
            const size_t kv_head = h * nkvhead / nhead;

            const T *q_vec = q + (i * nhead + h) * d;
            const T *k_head = k + kv_head * d;
            const T *v_head = v + kv_head * dv;

            const size_t eff_kv_len = past_len + i + 1;
            float max_score = -INFINITY;

            // QK^T 计算 + scale
            for (size_t j = 0; j < eff_kv_len; ++j)
            {
                const T *k_vec = k_head + j * nkvhead * d;
                float score = 0.0f;
                for (size_t t = 0; t < d; ++t)
                {
                    float q_val = llaisys::utils::cast<float>(q_vec[t]);
                    float k_val = llaisys::utils::cast<float>(k_vec[t]);
                    score += q_val * k_val;
                }
                score *= scale;
                scores_buf[j] = score;
                if (score > max_score)
                    max_score = score;
            }

            // Softmax 数值稳定版
            float sum_exp = 0.0f;
            for (size_t j = 0; j < eff_kv_len; ++j)
            {
                float exp_val = std::exp(scores_buf[j] - max_score);
                scores_buf[j] = exp_val;
                sum_exp += exp_val;
            }

            // Softmax 权重乘V 加权求和
            std::fill(out_buf.begin(), out_buf.end(), 0.0f);
            for (size_t j = 0; j < eff_kv_len; ++j)
            {
                float weight = scores_buf[j] / sum_exp;
                const T *v_vec = v_head + j * nkvhead * dv;
                for (size_t c = 0; c < dv; ++c)
                {
                    out_buf[c] += weight * llaisys::utils::cast<float>(v_vec[c]);
                }
            }

            // 写回输出张量
            T *out_vec = attn_val + (i * nhead + h) * dv;
            for (size_t c = 0; c < dv; ++c)
            {
                out_vec[c] = llaisys::utils::cast<T>(out_buf[c]);
            }
        }
    }
}

namespace llaisys::ops::cpu
{

void self_attention(
    std::byte *attn_val,
    const std::byte *q,
    const std::byte *k,
    const std::byte *v,
    llaisysDataType_t type,
    float scale,
    size_t seq_len,
    size_t total_len,
    size_t nhead,
    size_t nkvhead,
    size_t d,
    size_t dv)
{
    std::vector<float> scores_buf(total_len);
    float *scores = scores_buf.data();

    switch (type)
    {
    case LLAISYS_DTYPE_F32:
        self_attention_(
            reinterpret_cast<float *>(attn_val),
            reinterpret_cast<const float *>(q),
            reinterpret_cast<const float *>(k),
            reinterpret_cast<const float *>(v),
            scale, seq_len, total_len, nhead, nkvhead, d, dv, scores);
        break;

    case LLAISYS_DTYPE_F16:
        self_attention_(
            reinterpret_cast<llaisys::fp16_t *>(attn_val),
            reinterpret_cast<const llaisys::fp16_t *>(q),
            reinterpret_cast<const llaisys::fp16_t *>(k),
            reinterpret_cast<const llaisys::fp16_t *>(v),
            scale, seq_len, total_len, nhead, nkvhead, d, dv, scores);
        break;

    case LLAISYS_DTYPE_BF16:
        self_attention_(
            reinterpret_cast<llaisys::bf16_t *>(attn_val),
            reinterpret_cast<const llaisys::bf16_t *>(q),
            reinterpret_cast<const llaisys::bf16_t *>(k),
            reinterpret_cast<const llaisys::bf16_t *>(v),
            scale, seq_len, total_len, nhead, nkvhead, d, dv, scores);
        break;

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu