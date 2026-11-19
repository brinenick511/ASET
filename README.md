# ASET

[![ACL Anthology](https://img.shields.io/badge/ACL_Anthology-2026.findings--acl.2140-1f6feb?style=flat)](https://aclanthology.org/2026.findings-acl.2140/)

Code for *Faster MoE LLM Inference for Extremely Large Models* (Findings of ACL 2026). Adaptive Skipping with Entropy-Penalized Thresholding (ASET) chooses how many of OLMoE's top eight experts to activate for each token. The repository also includes static expert skipping in SGLang for DeepSeek-V3 on eight H200s. [Paper PDF](docs/paper.pdf) · [Poster PDF](docs/poster.pdf) · [DeepSeek-V3 experiment](experiments/deepseek-v3/README.md)

## OLMoE evaluation

Use Python 3.11 and one CUDA GPU with BF16 support. The evaluation downloads the pinned OLMoE checkpoint and benchmark datasets.

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python evaluate.py --config configs/olmoe.example.json --check-config
python evaluate.py --config configs/olmoe.example.json --limit 2 --output-path aset-results.json
```

Omit `--limit` to run all six tasks in full. The output includes task metrics in `results` and mean experts per routed position in `TOPK`. The count includes padding, so use matching batching and padding across comparisons.

[`routing.py`](routing.py) implements §4.3 and Algorithm 1. The config keys `tau_min`, `tau_max` and `beta` correspond to the paper's θ_min, θ_max and γ. Use `0 <= tau_min <= tau_max <= 1` and a finite positive `beta`.

The sample uses BF16, batch size 1 and SDPA. The archived six-task run used a local checkpoint, batch size 16 and FlashAttention 2, and recorded `TOPK=3.373717576265335` at `tau_min=0.2`, `tau_max=0.7` and `beta=4`. Figure 6 reproduction is pending. [Archived evaluator](https://github.com/brinenick511/lm_eval_aaai/tree/34b7fc7257f1399bdefd091da0cff72017a29fdc) · [Pinned OLMoE checkpoint](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct/tree/b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e)

## Licenses

The OLMoE block adapts [Transformers 4.51.3](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/olmoe/modeling_olmoe.py#L576). Evaluation uses [lm-evaluation-harness 0.4.9](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.9). [ASET MIT license](LICENSE) · [Transformers Apache 2.0 terms](licenses/Transformers-Apache-2.0.txt) · [EleutherAI MIT notice](licenses/lm-evaluation-harness-MIT.txt)

The published paper is available under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). [Publication record](https://aclanthology.org/2026.findings-acl.2140/)

## Citation

```bibtex
@inproceedings{yang-etal-2026-faster,
  title     = {Faster {M}o{E} {LLM} Inference for Extremely Large Models},
  author    = {Yang, Haoqi and Shi, Luohe and Li, Qiwei and Li, Zuchao and
               Wang, Ping and Huang, Hao and Zhao, Hai},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  year      = {2026},
  pages     = {43133--43151},
  doi       = {10.18653/v1/2026.findings-acl.2140},
  url       = {https://aclanthology.org/2026.findings-acl.2140/}
}
```
