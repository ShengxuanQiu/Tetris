import json
import os


def resolve_model_config_path(path: str) -> str:
    if os.path.exists(path):
        return path
    raise FileNotFoundError(f"Model config not found: {path}")


def load_model_config(path: str) -> dict:
    resolved = resolve_model_config_path(path)
    with open(resolved, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if "head_dim" not in cfg:
        if "v_head_dim" in cfg:
            cfg["head_dim"] = cfg["v_head_dim"]
        else:
            cfg["head_dim"] = cfg["hidden_size"] // cfg["num_attention_heads"]

    if "num_experts" not in cfg:
        routed = cfg.get("n_routed_experts", 0)
        shared = cfg.get("n_shared_experts", 0)
        cfg["num_experts"] = routed + shared

    if "moe_intermediate_size" not in cfg:
        cfg["moe_intermediate_size"] = cfg.get("intermediate_size")

    required = [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "vocab_size",
    ]
    for key in required:
        if key not in cfg:
            raise ValueError(f"Missing required model config key: {key}")
    return cfg
