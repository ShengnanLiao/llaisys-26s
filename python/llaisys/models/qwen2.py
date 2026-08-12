from typing import Sequence

from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType
from ..libllaisys import DataType
from ..libllaisys import LlaisysQwen2Meta

from pathlib import Path

import ctypes
import json
import re
import torch
import safetensors


class Qwen2:

    def __init__(
        self,
        model_path,
        device: DeviceType = DeviceType.CPU,
    ):
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model path does not exist: {model_path}"
            )

        # ======================================================
        # 1. 读取 config.json
        # ======================================================

        config_path = model_path / "config.json"

        if not config_path.exists():
            raise FileNotFoundError(
                f"config.json not found: {config_path}"
            )

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # ======================================================
        # 2. 根据 config 构造模型 Meta
        # ======================================================

        hidden_size = config["hidden_size"]
        num_heads = config["num_attention_heads"]

        head_dim = hidden_size // num_heads

        dtype_name = config.get(
            "torch_dtype",
            "bfloat16",
        )

        if dtype_name == "bfloat16":
            dtype = DataType.BF16
        elif dtype_name == "float16":
            dtype = DataType.F16
        elif dtype_name == "float32":
            dtype = DataType.F32
        else:
            raise ValueError(
                f"Unsupported model dtype: {dtype_name}"
            )

        self.meta = LlaisysQwen2Meta(
            dtype=dtype,
            nlayer=config["num_hidden_layers"],
            hs=hidden_size,
            nh=num_heads,
            nkvh=config["num_key_value_heads"],
            dh=head_dim,
            di=config["intermediate_size"],
            maxseq=config["max_position_embeddings"],
            voc=config["vocab_size"],
            epsilon=config["rms_norm_eps"],
            theta=config["rope_theta"],
            end_token=config["eos_token_id"],
        )

        self.device = device
        self.nlayer = self.meta.nlayer

        # ======================================================
        # 3. 创建 C++ 模型
        # ======================================================

        device_ids = (ctypes.c_int * 1)(0)

        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(self.meta),
            device,
            device_ids,
            1,
        )

        if not self._model:
            raise RuntimeError(
                "Failed to create Qwen2 model"
            )

        # ======================================================
        # 4. 获取 C++ 模型权重
        # ======================================================

        weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(
            self._model
        )

        if not weights_ptr:
            raise RuntimeError(
                "Failed to get Qwen2 model weights"
            )

        self._weights = weights_ptr.contents

        # ======================================================
        # 5. safetensors name -> C tensor
        # ======================================================

        def get_target_tensor(name: str):

            if name == "model.embed_tokens.weight":
                return self._weights.in_embed

            if name == "model.norm.weight":
                return self._weights.out_norm_w

            if name == "lm_head.weight":
                return self._weights.out_embed

            match = re.match(
                r"model\.layers\.(\d+)\.(.+)",
                name,
            )

            if match is None:
                raise KeyError(
                    f"Unknown model weight: {name}"
                )

            layer = int(match.group(1))
            suffix = match.group(2)

            if layer < 0 or layer >= self.nlayer:
                raise IndexError(
                    f"Layer index out of range: {layer}"
                )

            mapping = {
                "input_layernorm.weight":
                    self._weights.attn_norm_w,

                "self_attn.q_proj.weight":
                    self._weights.attn_q_w,

                "self_attn.q_proj.bias":
                    self._weights.attn_q_b,

                "self_attn.k_proj.weight":
                    self._weights.attn_k_w,

                "self_attn.k_proj.bias":
                    self._weights.attn_k_b,

                "self_attn.v_proj.weight":
                    self._weights.attn_v_w,

                "self_attn.v_proj.bias":
                    self._weights.attn_v_b,

                "self_attn.o_proj.weight":
                    self._weights.attn_o_w,

                "post_attention_layernorm.weight":
                    self._weights.mlp_norm_w,

                "mlp.gate_proj.weight":
                    self._weights.mlp_gate_w,

                "mlp.up_proj.weight":
                    self._weights.mlp_up_w,

                "mlp.down_proj.weight":
                    self._weights.mlp_down_w,
            }

            if suffix not in mapping:
                raise KeyError(
                    f"Unknown layer weight: {name}"
                )

            return mapping[suffix][layer]

        # ======================================================
        # 6. 逐个加载 safetensors
        # ======================================================

        loaded = 0

        for file in sorted(
            model_path.glob("*.safetensors")
        ):
            print(f"Loading {file.name}")
        
            with safetensors.safe_open(
                file,
                framework="pt",
                device="cpu",
            ) as data_:
        
                for name_ in data_.keys():
        
                    tensor = data_.get_tensor(name_).contiguous()
        
                    target = get_target_tensor(name_)
        
                    LIB_LLAISYS.tensorLoad(
                        target,
                        ctypes.c_void_p(tensor.data_ptr()),
                    )
        
                    loaded += 1
        
                    if loaded % 20 == 0:
                        print(
                            f"  loaded {loaded} tensors"
                        )

    def __del__(self):
        model = getattr(
            self,
            "_model",
            None,
        )

        if model:
            try:
                LIB_LLAISYS.llaisysQwen2ModelDestroy(
                    model
                )
            except Exception:
                pass

            self._model = None

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        if max_new_tokens is None:
            max_new_tokens = 128
    
        tokens = list(inputs)
    
        for _ in range(max_new_tokens):
    
            arr = (ctypes.c_int64 * len(tokens))(
                *tokens
            )
    
            next_token = (
                LIB_LLAISYS.llaisysQwen2ModelInfer(
                    self._model,
                    arr,
                    len(tokens),
                )
            )
    
            next_token = int(next_token)
    
            tokens.append(next_token)
    
            if next_token == self.meta.end_token:
                break
    
        return tokens