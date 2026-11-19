"""Emit an SGLang model override for a local DeepSeek-V3 checkpoint."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re


def add_routing_args(parser):
    routing = parser.add_mutually_exclusive_group()
    routing.add_argument("--top-k", type=int, help="Routed experts per MoE layer")
    routing.add_argument("--topk-list", type=Path, help="JSON array indexed by decoder layer")


def visible_devices():
    devices = os.getenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    indices = devices.split(",")
    if len(set(indices)) != 8 or len(indices) != 8 or not all(
        re.fullmatch(r"0|[1-9][0-9]*", index) for index in indices
    ):
        raise ValueError("set CUDA_VISIBLE_DEVICES to eight distinct CUDA device indices")
    return devices


def require_layerwise_patch():
    package = importlib.util.find_spec("sglang")
    if package is not None and package.origin is not None:
        model = Path(package.origin).parent / "srt/models/deepseek_v2.py"
        if model.is_file() and "ASET v0.4 adds optional per-decoder-layer" in model.read_text(
            encoding="utf-8"
        ):
            return
    raise ValueError("apply layerwise-topk.patch to the SGLang checkout used by this Python")


def build_override(checkpoint, top_k, schedule_path):
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("model_type") != "deepseek_v3":
        raise ValueError("expected a DeepSeek-V3 config.json")
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or "DeepseekV3ForCausalLM" not in architectures:
        raise ValueError("expected the DeepseekV3ForCausalLM architecture")
    original_k = config["num_experts_per_tok"]
    if type(original_k) is not int or original_k < 1:
        raise ValueError("invalid num_experts_per_tok in the checkpoint")

    if schedule_path is None:
        k = original_k if top_k is None else top_k
        if not 1 <= k <= original_k:
            raise ValueError(f"top-k must be from 1 to {original_k}")
        return {"num_experts_per_tok": k, "topk_list": None}

    topk_list = json.loads(schedule_path.read_text(encoding="utf-8"))
    num_layers = config["num_hidden_layers"]
    first_moe = config["first_k_dense_replace"]
    moe_freq = config["moe_layer_freq"]
    if (
        type(num_layers) is not int
        or type(first_moe) is not int
        or type(moe_freq) is not int
        or not 0 <= first_moe < num_layers
        or moe_freq < 1
    ):
        raise ValueError("invalid decoder layer configuration")
    if not isinstance(topk_list, list) or len(topk_list) != num_layers:
        raise ValueError(f"expected a list with {num_layers} decoder entries")
    for layer, k in enumerate(topk_list):
        if layer < first_moe or layer % moe_freq != 0:
            if k is not None:
                raise ValueError(f"dense layer {layer} must have a null entry")
        elif type(k) is not int or not 1 <= k <= original_k:
            raise ValueError(f"MoE layer {layer} must have an integer k from 1 to {original_k}")
    require_layerwise_patch()
    return {"topk_list": topk_list}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    add_routing_args(parser)
    args = parser.parse_args()

    try:
        visible_devices()
        override = build_override(args.checkpoint, args.top_k, args.topk_list)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    print(json.dumps(override))


if __name__ == "__main__":
    main()
