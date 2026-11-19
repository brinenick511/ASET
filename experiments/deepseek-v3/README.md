# DeepSeek-V3 static expert skipping

This ASET v0.4 supplement runs fixed and layerwise routed top-k on a single Linux node with eight H200 141 GB GPUs and TP8. It uses a local FP8 DeepSeek-V3 checkpoint, including its tokenizer and configuration files. [DeepSeek-V3 checkpoint](https://huggingface.co/deepseek-ai/DeepSeek-V3) · [SGLang DeepSeek setup](https://github.com/sgl-project/sglang/blob/ad4e58bf67ec833ff4d036af5129ec6e1633efc4/docs/references/deepseek.md)

## Install

Use an isolated Python 3.11 environment on the Linux x86-64 GPU host with the CUDA 12.4 toolkit and a C++ compiler. Run from the ASET root. Build `sgl-kernel==0.0.5.post3` from the pinned checkout because that version is absent from PyPI. The patch is required for layerwise schedules. [Pinned kernel source](https://github.com/sgl-project/sglang/tree/ad4e58bf67ec833ff4d036af5129ec6e1633efc4/sgl-kernel)

```bash
exp="$PWD/experiments/deepseek-v3"
git clone --filter=blob:none --no-checkout https://github.com/sgl-project/sglang.git sglang
git -C sglang checkout --detach ad4e58bf67ec833ff4d036af5129ec6e1633efc4
git -C sglang apply "$exp/layerwise-topk.patch"
git -C sglang submodule update --init --recursive
python3 -m pip install 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cu124
python3 -m pip install 'setuptools>=75' wheel ninja scikit-build-core
python3 -m pip install --no-build-isolation ./sglang/sgl-kernel
python3 -m pip install -e "./sglang/python[all]" \
  --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
```

The paper identifies this exact commit as the SGLang 0.4.4.post1 base. The public archive fork pins the original `topk_list` mechanism. The supplement uses SGLang's config override for fixed k and a small patch for layerwise k. It passes the selected width to the existing grouped router and dispatch path with the full expert pool. [Base](https://github.com/sgl-project/sglang/commit/ad4e58bf67ec833ff4d036af5129ec6e1633efc4) · [Archive fork at `71b12e17fc6adccbf0978a10cef14d4c5805eb97`](https://github.com/brinenick511/sglang-colm/tree/71b12e17fc6adccbf0978a10cef14d4c5805eb97) · [Apache 2.0 license and SGLang notice](LICENSE-SGLang.txt)

## Serve and benchmark

Set `model` to the local checkpoint directory in both terminals. Set `sharegpt` to a local ShareGPT JSON file with at least 2,048 usable conversations with two or more turns. Keep the checkpoint and dataset fixed across runs.

```bash
# Terminal 1, from the ASET root
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash experiments/deepseek-v3/serve.sh "$model" --top-k 8

# Terminal 2, from the ASET root
bash experiments/deepseek-v3/bench.sh "$model" "$sharegpt" throughput-k8.jsonl --top-k 8
```

Repeat for `--top-k 6`, `4` and `3` in separate server runs. Pass the same setting to both scripts and use new output paths. The benchmark checks the checkpoint, TP size and routing override and writes the verified settings to `<output>.routing.json`. Set `PORT` in both terminals to change the loopback port and `READY_TIMEOUT` to change the 900-second startup wait. The server uses TP attention and stock compile settings by default. Set `DP_SIZE=2`, `4` or `8` to enable DP attention.

For layerwise routing, pass `--topk-list schedule.json` in place of `--top-k`. The JSON array has one entry per zero-based decoder layer. Use `null` for dense layers and integers from 1 to the checkpoint's original k for MoE layers. For the 61-layer V3 configuration, this example gives the first 29 MoE layers k=4 and the last 29 k=6.

```bash
python3 -c 'import json; print(json.dumps([None] * 3 + [4] * 29 + [6] * 29))' > schedule.json
```

## Evaluate tasks

The task runner starts its own SGLang engine on all eight devices. Run it after freeing the GPUs used by the server. It saves the six individual zero-shot task results at batch size 32 with the selected routing configuration. Datasets must be available from the Hub or in the local cache. The runner uses the checkpoint's native FP8 weights and SGLang's default KV-cache type. [Pinned lm-evaluation-harness 0.4.8](https://github.com/EleutherAI/lm-evaluation-harness/commit/1afb190ce95340453d60aaccce9c3e5ad7888805)

```bash
python3 -m pip install 'git+https://github.com/EleutherAI/lm-evaluation-harness.git@1afb190ce95340453d60aaccce9c3e5ad7888805'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python3 experiments/deepseek-v3/evaluate_tasks.py "$model" tasks-example.json \
    --topk-list schedule.json
```

For a reference run, use `--top-k 8` and a separate output file. For a uniform schedule with the same mean k=5 as the two-block example, use `--top-k 5`. Both are comparison settings for new runs. The archived per-task metric choices for the paper's six-task aggregate remain unverified.

## Experiment record

- The paper's §5 and Appendix D.2 specify eight H200s and 1024 input / 1024 output tokens. The serving sweep here follows Table 8 with k=8, 6, 4, 3 and concurrency 2–128. Figure 4b labels the last series k=2. Figure 5b and Table 10 report the six-task layerwise sweep. The historical V3 launch arguments, checkpoint and dataset revisions, and per-layer arrays are unverified. [Paper PDF](https://aclanthology.org/2026.findings-acl.2140.pdf)
- The archived `sh_0328/o/server_0328.sh` and `sglang-colm/scripts/h200_0324.sh` name DeepSeek-V2-Lite. The serving client template `sh_0328/h200_0328.sh` supplies the ShareGPT workload, ratio 1 and 16 requests per concurrency slot used here. The fork's `scripts/utils/modify_config.py` supplies the layer-list convention. [Archived task script](https://github.com/brinenick511/sglang-colm/blob/71b12e17fc6adccbf0978a10cef14d4c5805eb97/scripts/h200_0324.sh) · [Schedule source](https://github.com/brinenick511/sglang-colm/blob/71b12e17fc6adccbf0978a10cef14d4c5805eb97/scripts/utils/modify_config.py)
- Compare the client's end-to-end `output_throughput` with k=8 at the same concurrency. It divides generated tokens by wall-clock time, including prefill. The paper excludes prefill, and the historical timing calculation remains unavailable. This supplement has static verification only. [Pinned benchmark source](https://github.com/sgl-project/sglang/blob/ad4e58bf67ec833ff4d036af5129ec6e1633efc4/python/sglang/bench_serving.py)
