# SPDX-License-Identifier: Apache-2.0
"""Per-tensor precision metadata for Inkling2 on Volta (sm_70).

Only the routed experts are quantized (NVFP4 on layers 0..19, MXFP8 on layers 20..23);
all other linears run in FP16 with FP32 accumulation and FP32 residual stream.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import torch

from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)

from .configs import INKLING2_ARCHITECTURES

_COMPUTE_DTYPES = ("float32", "bfloat16", "float16", "mxfp8", "nvfp4")


class Inkling2QuantConfig(QuantizationConfig):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.mxfp8_weights: set[str] = set()
        for group in config["model"]["parameter_groups"]:
            compute = group["compute_dtype"]
            if compute not in _COMPUTE_DTYPES:
                raise ValueError(f"Unsupported Inkling2 compute dtype: {compute}")
            if compute == "mxfp8":
                self.mxfp8_weights.update(group["tensors"])

    @classmethod
    def get_name(cls) -> str:
        return "inkling2"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["hf_quant_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Inkling2QuantConfig:
        return cls(config)

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
        hf_config: Any = None,
    ) -> str | None:
        architectures = getattr(hf_config, "architectures", None) or []
        if INKLING2_ARCHITECTURES & set(architectures):
            # Released Inkling2 checkpoints declare a minimal ModelOpt stub in
            # config.json (`quantization_config`) while the full per-tensor
            # precision map lives in `hf_quant_config.json`. Populate both
            # `hf_quant_cfg` (`model_arch_config` copy) and
            # `hf_config.quantization_config` so 1Cat-vLLM's get_quant_config
            # always passes the full map to from_config.
            full_cfg: dict[str, Any] | None = None
            if isinstance(hf_quant_cfg, dict) and "model" in hf_quant_cfg:
                full_cfg = hf_quant_cfg
            else:
                model_path = getattr(hf_config, "_name_or_path", None)
                if model_path:
                    p = pathlib.Path(model_path) / "hf_quant_config.json"
                    if p.is_file():
                        full_cfg = json.loads(p.read_text())
            if full_cfg is not None:
                if isinstance(hf_quant_cfg, dict) and hf_quant_cfg is not full_cfg:
                    hf_quant_cfg.update(full_cfg)
                if hf_config is not None:
                    cur = getattr(hf_config, "quantization_config", None)
                    if isinstance(cur, dict):
                        cur.update(full_cfg)
                    else:
                        setattr(hf_config, "quantization_config", dict(full_cfg))
            return "inkling2"
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        return None

    def is_mxfp8_layer(self, layer_id: int) -> bool:
        name = f"model.llm.layers.{layer_id}.mlp.experts.w13_weight"
        return name in self.mxfp8_weights
