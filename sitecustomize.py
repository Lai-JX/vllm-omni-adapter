# sitecustomize.py
# Auto-executed by Python at startup (when this directory is in PYTHONPATH).
# Registers alpamayo_reasoning_vla model type so vLLM can load Alpamayo models.
try:
    from transformers import AutoConfig
    from alpamayo1_5.models.base_model import ReasoningVLAConfig

    try:
        AutoConfig.register("alpamayo_reasoning_vla", ReasoningVLAConfig)
    except ValueError:
        pass  # Already registered
except Exception:
    pass  # Best-effort; don't block Python startup
