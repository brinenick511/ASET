"""Evaluate the six DeepSeek-V3 tasks through a local SGLang engine."""

import argparse
import json
import os
from pathlib import Path

from routing_override import add_routing_args, build_override, visible_devices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path, help="New JSON result file")
    add_routing_args(parser)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()

    try:
        override = build_override(checkpoint, args.top_k, args.topk_list)
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices()
        if args.output.exists() or not args.output.parent.is_dir():
            raise ValueError("provide a new output file in an existing directory")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    from lm_eval import simple_evaluate
    from lm_eval.utils import handle_non_serializable

    result = simple_evaluate(
        model="sglang",
        model_args={
            "pretrained": str(checkpoint),
            "trust_remote_code": True,
            "enable_p2p_check": True,
            "disable_cuda_graph": True,
            "tp_size": 8,
            "mem_fraction_static": 0.75,
            "json_model_override_args": json.dumps(override),
        },
        tasks=["arc_challenge", "arc_easy", "boolq", "openbookqa", "rte", "winogrande"],
        num_fewshot=0,
        batch_size=32,
        log_samples=False,
    )
    payload = json.dumps(result, indent=2, default=handle_non_serializable)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(payload + "\n")


if __name__ == "__main__":
    main()
