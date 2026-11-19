#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 )); then
    echo "Usage: $0 CHECKPOINT [--top-k K | --topk-list FILE]" >&2
    exit 2
fi
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null && pwd -P)
checkpoint=$(cd -- "$1" >/dev/null && pwd -P)
shift
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0,1,2,3,4,5,6,7}"
override=$(python3 "$script_dir/routing_override.py" "$checkpoint" "$@")

port=${PORT:-30000}
if [[ ! $port =~ ^[1-9][0-9]{0,4}$ ]] || (( port > 65535 )); then
    echo "PORT must be an integer from 1 to 65535." >&2
    exit 2
fi
dp_size=${DP_SIZE:-1}
set --
case $dp_size in
    1) ;;
    2|4|8) set -- --enable-dp-attention --dp-size "$dp_size" ;;
    *) echo "DP_SIZE must be 1, 2, 4 or 8 for TP8." >&2; exit 2 ;;
esac

printf 'DeepSeek-V3 on TP8, DP_SIZE=%s, routing override %s\n' "$dp_size" "$override" >&2
export HF_HUB_OFFLINE=1
exec python3 -m sglang.launch_server \
    --model-path="$checkpoint" --trust-remote-code \
    --host 127.0.0.1 --port "$port" --device cuda --tp-size 8 \
    --enable-p2p-check \
    --json-model-override-args "$override" \
    "$@"
