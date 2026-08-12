# LLAISYS 实验记录

> 实验环境：AutoDL 平台租用的 RTX 2080Ti 服务器  

---

## 0. 环境信息

### 0.1 硬件配置

| 项目 | 详情 |
|------|------|
| **GPU** | NVIDIA GeForce RTX 2080 Ti（11GB 显存） |
| **CUDA Version** | 13.0 |
| **Driver Version** | 580.105.08 |
| **CPU 核心数** | 12 |
| **内存** | 总计 375Gi，可用 325Gi |
| **Swap** | 0B（未启用） |

### 0.2 GPU 状态

```
Fan: 32%    Temp: 31°C    Power: 20W / 250W
Memory: 0MiB / 11264MiB    GPU-Util: 0%
```

> 实验开始时 GPU 处于空闲状态，无运行中的进程。

---

## 1. 作业 1：Tensor 操作测试

### 命令

```bash
python test/test_tensor.py
```

### 测试结果

#### ✅ Test load

Tensor shape `[3, 4, 5]`，strides `[20, 5, 1]`，dtype=6，数据从 0 到 59 连续填充。

#### ✅ Test view

将 `[3, 4, 5]` 的 Tensor reshape 为 `[6, 10]`，strides `[10, 1]`，数据保持一致。

#### ✅ Test permute

对 `[3, 4, 5]` 的 Tensor 进行维度转置，得到 `[5, 3, 4]`，strides `[1, 20, 5]`。

#### ✅ Test slice

对原始 Tensor 进行切片操作，得到 `[3, 4, 3]`，strides `[20, 5, 1]`。

### 结论

```
✅ Test passed!
```

所有 Tensor 基础操作（load、view、permute、slice）均通过测试。

---

## 2. 作业 2：算子测试

### 命令

```bash
cd /root/autodl-tmp/llaisys/test/ops
for f in *.py; do echo "===== 正在运行 $f ====="; python3 "$f"; done
```

### 测试结果

共测试 **8 个算子**，均在 CPU 上运行，覆盖 `f32`、`f16`、`bf16` 三种精度：

| # | 算子 | 测试规模 | 状态 |
|---|------|----------|------|
| 1 | **add** | 小规模 (2,3) + 大规模 (512,4096) | ✅ Passed |
| 2 | **argmax** | 小规模 (4,) + 大规模 (4096,) | ✅ Passed |
| 3 | **embedding** | 小规模 idx(1,) embd(2,3) + 大规模 idx(50,) embd(512,4096) | ✅ Passed |
| 4 | **linear** | 小规模 (2,4)→(2,3) + 大规模 (512,4096)→(512,4096) | ✅ Passed |
| 5 | **rms_norm** | 小规模 (1,4) + 大规模 (512,4096) | ✅ Passed |
| 6 | **rope** | 小规模 (2,1,4) + 大规模 (512,4,4096) | ✅ Passed |
| 7 | **self_attention** | 小规模 qlen=2,kvlen=2 + 大规模 qlen=5,kvlen=11 | ✅ Passed |
| 8 | **swiglu** | 小规模 (2,3) + 大规模 (512,4096) | ✅ Passed |

### 结论

所有算子测试通过，覆盖了 LLM 推理中的关键操作：加法、argmax、embedding、线性层、RMSNorm、RoPE 位置编码、自注意力和 SwiGLU 激活函数。

---

## 3. 纯 CPU 推理

### 命令

```bash
python test/test_infer.py \
  --model ./DeepSeek-R1-Distill-Qwen-1.5B \
  --test \
  --prompt "A very very simple answer to who you are"
```

### 测试说明

使用 **DeepSeek-R1-Distill-Qwen-1.5B** 模型，在纯 CPU 模式下进行推理测试，对比参考实现与自研实现的结果。

### 推理结果

**Prompt：** *"A very very simple answer to who you are"*

**模型输出：**

> Greetings! I'm DeepSeek-R1, an artificial intelligence assistant created by DeepSeek. I'm at your service and would be delighted to assist you with any inquiries or tasks you may have.

### 性能对比

| 指标 | 参考实现 | 自研实现 |
|------|----------|----------|
| **Token 序列** | 一致 ✅ | 一致 ✅ |
| **输出内容** | 一致 ✅ | 一致 ✅ |
| **耗时** | 9.05s | 42.06s |

### 结论

```
✅ Test passed!
```

自研实现的推理结果与参考实现完全一致，Token 序列和输出内容均匹配。CPU 模式下自研实现耗时约为参考实现的 4.6 倍，后续可通过 GPU 加速优化。

---

## 4. GPU CUDA 算子加速推理

### 4.1 测试 Runtime

#### 命令

```bash
python test/test_runtime.py --device nvidia
```

#### 结果

```
Found 1 nvidia devices
Testing device {i}...
     Passed

✅ Test passed!
```

成功检测到 1 个 NVIDIA 设备，Runtime 测试通过。

---

### 4.2 GPU 推理

#### 命令

```bash
python test/test_infer.py \
  --model /root/autodl-tmp/llaisys/DeepSeek-R1-Distill-Qwen-1.5B \
  --test \
  --device nvidia
```

#### 推理结果

**Prompt：** *"Who are you?"*

**模型输出：**

> Greetings! I'm DeepSeek-R1, an artificial intelligence assistant created by DeepSeek. I'm at your service and would be delighted to assist you with any inquiries or tasks you may have.

#### 性能对比

| 指标 | 参考实现 | 自研实现 |
|------|----------|----------|
| **Token 序列** | 一致 ✅ | 一致 ✅ |
| **输出内容** | 一致 ✅ | 一致 ✅ |
| **耗时** | 3.20s | 9.56s |

#### 结论

```
✅ Test passed!
```

GPU 加速后自研实现推理结果与参考实现完全一致。相比 CPU 模式（42.06s），GPU 推理耗时降至 9.56s，加速比约 **4.4x**。

---

## 总结

| 阶段 | 内容 | 结果 |
|------|------|------|
| **环境** | AutoDL RTX 2080Ti, 12核 CPU, 375GB 内存 | ✅ 就绪 |
| **作业 1** | Tensor 基础操作（load/view/permute/slice） | ✅ 全部通过 |
| **作业 2** | 8 个算子测试（add/argmax/embedding/linear/rms_norm/rope/self_attention/swiglu） | ✅ 全部通过 |
| **CPU 推理** | DeepSeek-R1-Distill-Qwen-1.5B 纯 CPU 推理 | ✅ 结果一致，耗时 42.06s |
| **GPU 推理** | CUDA 算子加速推理 | ✅ 结果一致，耗时 9.56s |

> **关键结论：** 自研 LLAISYS 框架的推理结果在 CPU 和 GPU 模式下均与参考实现完全一致；GPU 加速相比 CPU 推理获得约 4.4 倍的性能提升。

---

// 后续会在天数、摩尔、沐曦卡上适配。
