"""Volta (sm_70) plugin for serving thinkingmachines/Inkling-2-Mini on 1Cat-vLLM v1.5.4.

Zero edits to 1Cat-vLLM site-packages; zero interference with Qwen3.8-27B / q27fuse.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger("inkling_v100")
_REGISTERED = False


def register() -> None:
    global _REGISTERED
    if _REGISTERED or os.environ.get("INKLING_V100", "1") == "0":
        return
    _REGISTERED = True

    from vllm.model_executor.layers.quantization import register_quantization_config
    from vllm.model_executor.models.registry import ModelRegistry
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    from .configs import (
        INKLING2_ARCHITECTURE,
        INKLING2_LOOP_ARCHITECTURE,
        Inkling2DFly2Config,
        Inkling2DSparkConfig,
        Inkling2LoopMMConfig,
        Inkling2MMConfig,
        Inkling2ModelConfig,
    )
    from .quant_config import Inkling2QuantConfig

    # 1. Register HF config classes for AutoConfig / vLLM get_config.
    _CONFIG_REGISTRY["inkling2_loop"] = Inkling2LoopMMConfig
    _CONFIG_REGISTRY["inkling2"] = Inkling2MMConfig
    _CONFIG_REGISTRY["inkling2_model"] = Inkling2ModelConfig
    _CONFIG_REGISTRY["inkling2_loop_text"] = Inkling2ModelConfig
    _CONFIG_REGISTRY["inkling2_dspark"] = Inkling2DSparkConfig
    _CONFIG_REGISTRY["inkling2_dfly2_kv"] = Inkling2DFly2Config

    # 2. Register quantization config (takes precedence over config.json's ModelOpt block).
    register_quantization_config("inkling2")(Inkling2QuantConfig)

    # 3. Register model architectures (lazy module path to avoid early torch/CUDA init).
    ModelRegistry.register_model(
        INKLING2_LOOP_ARCHITECTURE,
        "inkling_v100.model:Inkling2LoopForConditionalGeneration",
    )
    ModelRegistry.register_model(
        INKLING2_ARCHITECTURE,
        "inkling_v100.model:Inkling2ForConditionalGeneration",
    )
    ModelRegistry.register_model(
        "DFlashInkling2DSparkModel",
        "inkling_v100.dspark:Inkling2DSparkModel",
    )
    ModelRegistry.register_model(
        "I2MDFly2DraftModel",
        "inkling_v100.dfly2:Inkling2DFly2DraftModel",
    )
    ModelRegistry.register_model(
        "DFlashI2MDFly2DraftModel",
        "inkling_v100.dfly2:Inkling2DFly2DraftModel",
    )

    # 4. Route `inkling2_dspark` and `inkling2_dfly2_kv` draft configs without
    # editing 1Cat-vLLM site-packages.
    _install_dspark_speculator_hook()
    from .conv_state import install_ring_conv_hook
    from .spec_k import apply_spec_k

    install_ring_conv_hook()
    apply_spec_k()
    log.info(
        "inkling_v100: registered Inkling2Loop/Inkling2/DSpark/DFly2 configs, quant, models, and speculator hook for sm_70"
    )


def _is_inkling2_dfly2(vllm_config) -> bool:
    spec = getattr(vllm_config, "speculative_config", None)
    if spec is None:
        return False
    if os.environ.get("INKLING_DFLY2_CKPT", "").strip():
        mc_archs = getattr(vllm_config.model_config.hf_config, "architectures", None) or ()
        if any("Inkling2" in a for a in mc_archs):
            return True
    draft_cfg = getattr(spec, "draft_model_config", None)
    hf_cfg = getattr(draft_cfg, "hf_config", None)
    if hf_cfg is None:
        return False
    if getattr(hf_cfg, "model_type", None) == "inkling2_dfly2_kv":
        return True
    inner = getattr(hf_cfg, "model", None)
    if getattr(inner, "model_type", None) == "inkling2_dfly2_kv":
        return True
    archs = getattr(hf_cfg, "architectures", None) or ()
    return ("I2MDFly2DraftModel" in archs) or ("DFlashI2MDFly2DraftModel" in archs)


def _is_inkling2_dspark(vllm_config) -> bool:
    spec = getattr(vllm_config, "speculative_config", None)
    if spec is None or getattr(spec, "method", None) != "dflash":
        return False
    draft_cfg = getattr(spec, "draft_model_config", None)
    hf_cfg = getattr(draft_cfg, "hf_config", None)
    if hf_cfg is None:
        return False
    if getattr(hf_cfg, "model_type", None) == "inkling2_dspark":
        return True
    inner = getattr(hf_cfg, "model", None)
    if getattr(inner, "model_type", None) == "inkling2_dspark":
        return True
    archs = getattr(hf_cfg, "architectures", None) or ()
    return "DFlashInkling2DSparkModel" in archs


def _install_dspark_speculator_hook() -> None:
    import sys
    import vllm.v1.worker.gpu.spec_decode as sd

    if getattr(sd.init_speculator, "_inkling_v100_wrapped", False):
        return
    orig_init = sd.init_speculator

    def _wrapped_init_speculator(vllm_config, device):
        if _is_inkling2_dfly2(vllm_config):
            from .dfly2 import Inkling2DFly2Speculator

            return Inkling2DFly2Speculator(vllm_config, device)
        if _is_inkling2_dspark(vllm_config):
            from .speculator import Inkling2DSparkSpeculator

            return Inkling2DSparkSpeculator(vllm_config, device)
        return orig_init(vllm_config, device)

    _wrapped_init_speculator._inkling_v100_wrapped = True  # type: ignore[attr-defined]
    sd.init_speculator = _wrapped_init_speculator
    mr = sys.modules.get("vllm.v1.worker.gpu.model_runner")
    if mr is not None and getattr(mr, "init_speculator", None) is orig_init:
        mr.init_speculator = _wrapped_init_speculator
