# ASET

Adaptive Skipping with Entropy-Penalized Thresholding for OLMoE. [`routing.py`](routing.py) implements the candidate prefix, normalized entropy and per-token threshold in §4.3 and Algorithm 1. It dispatches only the retained experts within OLMoE's original top-8 budget. [`evaluate.py`](evaluate.py) uses lm-eval to score the six tasks from the archived OLMoE run.

## Run

Use Python 3.11 and one CUDA GPU with BF16 support. From this directory, run the configuration check and then the evaluation. The evaluation downloads the pinned OLMoE checkpoint and the benchmark datasets.

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python evaluate.py --config configs/olmoe.example.json --check-config
python evaluate.py --config configs/olmoe.example.json --limit 2 --output-path aset-results.json
```

The configuration check validates required fields and prints the parsed JSON. The evaluation writes task metrics in `results` and the mean of the per-layer expert counts per routed position in `TOPK`. For this checkpoint, `TOPK` is between 1 and 8. The count includes padding positions, so compare runs with matching batching and padding. Omit `--limit` to score each full task.

The CLI accepts `0 <= tau_min <= tau_max <= 1` and a finite positive `beta`. These keys represent the paper's θ_min, θ_max and γ.

**Ubuntu status.** The historical run recorded Ubuntu 22.04.4 LTS with PyTorch 2.4.1 and CUDA 12.1. This release has not been run on Ubuntu or with the full checkpoint.

## Cite and trace

Haoqi Yang, Luohe Shi, Qiwei Li, Zuchao Li, Ping Wang, Hao Huang, and Hai Zhao. 2026. *Faster MoE LLM Inference for Extremely Large Models*. Findings of the Association for Computational Linguistics: ACL 2026. [Paper](https://aclanthology.org/2026.findings-acl.2140/)

In the experiment archive, `data/olmoe.tar.gz` contains `olmoe/fag_4/0.2_0.7_2025-07-30T03-50-03.951557.json`. That result records the six zero-shot tasks, generation settings, checkpoint name, BF16 and `tau_min=0.2`, `tau_max=0.7`, `beta=4` used in the sample. The archived run used a local checkpoint path, batch size 16 and FlashAttention 2, and reported `TOPK=3.373717576265335`. The sample pins a Hub revision and uses batch size 1 and SDPA. The release router follows the paper and the archived OLMoE block interface. Figure 6 reproduction remains unverified.

- Recorded evaluation checkout [lm_eval_aaai at `34b7fc72`](https://github.com/brinenick511/lm_eval_aaai/tree/34b7fc7257f1399bdefd091da0cff72017a29fdc).
- OLMoE block adapted from the archived `calibration/models/debugging_olmoe.py` at `7f7643879155bbe748489c7cb628a4810c34dd3e` and [Transformers 4.51.3](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/olmoe/modeling_olmoe.py#L576).
- Evaluation dependencies [lm-evaluation-harness 0.4.9](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.9) and [AllenAI OLMoE revision `b89a7c4`](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct/tree/b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e).
- [ASET MIT license](LICENSE), [OLMoE Apache 2.0 terms](licenses/Transformers-Apache-2.0.txt), and [EleutherAI MIT notice](licenses/lm-evaluation-harness-MIT.txt).
