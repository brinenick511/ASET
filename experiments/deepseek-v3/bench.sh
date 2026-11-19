#!/usr/bin/env bash
set -euo pipefail

if (( $# != 5 )); then
    echo "Usage: $0 CHECKPOINT SHAREGPT_JSON OUTPUT_JSONL (--top-k K | --topk-list FILE)" >&2
    exit 2
fi
checkpoint=$(cd -- "$1" >/dev/null && pwd -P)
dataset=$2
output=$3
shift 3
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null && pwd -P)
expected_override=$(python3 "$script_dir/routing_override.py" "$checkpoint" "$@")
port=${PORT:-30000}
timeout=${READY_TIMEOUT:-900}
if [[ ! -r "$checkpoint/config.json" || ! -f "$dataset" || ! -r "$dataset" ]]; then
    echo "Provide a local checkpoint and readable ShareGPT JSON file." >&2
    exit 2
fi
if [[ ! $port =~ ^[1-9][0-9]{0,4}$ ]] || (( port > 65535 )); then
    echo "PORT must be an integer from 1 to 65535." >&2
    exit 2
fi
if [[ ! $timeout =~ ^[1-9][0-9]{0,5}$ ]]; then
    echo "READY_TIMEOUT must be a positive integer in seconds (at most 999999)." >&2
    exit 2
fi

url="http://127.0.0.1:$port"
deadline=$((SECONDS + timeout))
printf 'Waiting for SGLang at %s\n' "$url" >&2
while :; do
    remaining=$((deadline - SECONDS))
    if (( remaining <= 0 )); then
        echo "SGLang did not become ready at $url." >&2
        exit 1
    fi
    probe_time=$((remaining < 30 ? remaining : 30))
    if curl --noproxy '*' --fail --silent --max-time "$probe_time" \
        --output /dev/null "$url/health_generate"; then
        break
    fi
    sleep 1
done

curl --noproxy '*' --fail --silent --show-error --max-time 5 "$url/get_model_info" |
    python3 -c '
import json
from pathlib import Path
import sys

model = json.load(sys.stdin)
if not model["is_generation"] or Path(model["model_path"]).resolve() != Path(sys.argv[1]):
    sys.exit("The server is not serving the requested checkpoint.")
' "$checkpoint"

if [[ -e "$output" || -e "$output.routing.json" ]]; then
    echo "Choose a new output JSONL path in an existing directory." >&2
    exit 2
fi

verified_routing=$(
    curl --noproxy '*' --fail --silent --show-error --max-time 5 "$url/get_server_info" |
        python3 -c '
import json
import sys

server = json.load(sys.stdin)
actual = json.loads(server["json_model_override_args"])
if actual != json.loads(sys.argv[1]) or server["tp_size"] != 8:
    sys.exit("The server routing override or tensor-parallel size does not match this run.")
print(json.dumps({"tensor_parallel_size": 8, "routing_override": actual}))
' "$expected_override"
)
(
    set -o noclobber
    : > "$output"
    printf '%s\n' "$verified_routing" > "$output.routing.json"
)

export HF_HUB_OFFLINE=1
for concurrency in 2 4 8 16 32 64 128; do
    prompts=$((16 * concurrency))
    python3 -m sglang.bench_serving \
        --backend sglang --host 127.0.0.1 --port "$port" \
        --model="$checkpoint" --dataset-name random --dataset-path="$dataset" \
        --random-input-len 1024 --random-output-len 1024 --random-range-ratio 1 \
        --num-prompts "$prompts" --max-concurrency "$concurrency" --output-file="$output"
    tail -n 1 -- "$output" | python3 -c '
import json
import sys

row = json.load(sys.stdin)
if row["max_concurrency"] != int(sys.argv[1]) or row["completed"] != int(sys.argv[2]):
    sys.exit("Benchmark did not complete the requested prompts.")
' "$concurrency" "$prompts"
done
