"""Evaluate the ASET router on OLMoE with lm-eval."""

import argparse
import json
import math
from pathlib import Path


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    if config.get("model") != "skip":
        raise ValueError("model must be 'skip'")
    tasks = config.get("tasks")
    if not isinstance(tasks, (str, list)) or not tasks:
        raise ValueError("tasks must be a task name or a non-empty list of tasks")
    if isinstance(tasks, list) and any(task is None for task in tasks):
        raise ValueError("tasks must not contain null entries")

    model_args = config.get("model_args")
    if not isinstance(model_args, dict):
        raise ValueError("model_args must be an object")
    pretrained = model_args.get("pretrained")
    if pretrained is None:
        raise ValueError("model_args.pretrained is required")
    for key in ("device", "batch_size", "num_fewshot", "gen_kwargs"):
        if key not in config:
            raise ValueError(f"{key} is required")
    for key in ("device", "batch_size"):
        if key in model_args:
            raise ValueError(f"{key} must be set only at the top level")
    if isinstance(pretrained, str) and not isinstance(config["device"], str):
        raise ValueError("device must be a string for a checkpoint name or path")
    batch_size = config["batch_size"]
    if batch_size is None or (
        isinstance(pretrained, str) and not isinstance(batch_size, (int, str))
    ):
        raise ValueError("batch_size must be an integer or a string")

    theta_min, theta_max, gamma = (model_args[key] for key in ("tau_min", "tau_max", "beta"))
    if not 0 <= theta_min <= theta_max <= 1:
        raise ValueError("expected 0 <= tau_min <= tau_max <= 1")
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("beta must be finite and positive")


def run_evaluation(config: dict, limit: int | None) -> dict:
    validate_config(config)
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    from routing import install_aset

    hf_model_args = config["model_args"].copy()
    theta_min = hf_model_args.pop("tau_min")
    theta_max = hf_model_args.pop("tau_max")
    gamma = hf_model_args.pop("beta")
    lm_eval_model = HFLM(
        **hf_model_args,
        batch_size=config["batch_size"],
        device=config["device"],
    )
    layer_routing_stats = install_aset(lm_eval_model.model, theta_min, theta_max, gamma)
    results = simple_evaluate(
        model=lm_eval_model,
        tasks=config["tasks"],
        num_fewshot=config["num_fewshot"],
        gen_kwargs=config["gen_kwargs"],
        limit=limit,
        log_samples=False,
    )
    layer_mean_activations = [
        stats.num_experts_activated.item() / stats.num_tokens
        for stats in layer_routing_stats
        if stats.num_tokens
    ]
    if not layer_mean_activations:
        raise RuntimeError("Evaluation did not route any tokens")
    # Match the archived evaluator's mean of the per-layer activation means.
    results["TOPK"] = sum(layer_mean_activations) / len(layer_mean_activations)
    results["aset"] = config
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--limit", type=int, help="Maximum examples per task")
    parser.add_argument("--output-path", type=Path, default=Path("aset-results.json"))
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    try:
        config = load_config(args.config)
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))

    if args.check_config:
        print(json.dumps(config, indent=2))
        return

    from lm_eval.utils import handle_non_serializable

    results = run_evaluation(config, args.limit)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as output:
        json.dump(results, output, indent=2, default=handle_non_serializable)
        output.write("\n")
    print(f"TOPK={results['TOPK']:.6f}  {args.output_path}")


if __name__ == "__main__":
    main()
