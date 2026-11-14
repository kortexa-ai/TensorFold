#!/usr/bin/env bash
# One foreground rank. Run on both Sparks with the same master address and context.
set -euo pipefail
rank=${RANK:?Set RANK to 0 on the API Spark or 1 on its peer}
master=${MASTER_ADDR:?Set MASTER_ADDR to the API Spark IPv4 address reachable by both hosts}
model=${MODEL_DIR:?'Set MODEL_DIR to this rank download directory'}
[[ $rank == 0 || $rank == 1 ]] || { echo 'RANK must be 0 or 1.' >&2; exit 2; }
[[ $(uname -s) == Linux && $(uname -m) == aarch64 ]] || exit 2
python3 - "$master" "$model" "$rank" <<'PY'
import ipaddress, json, pathlib, sys
ipaddress.IPv4Address(sys.argv[1])
p = pathlib.Path(sys.argv[2])
r = json.loads((p / 'rank.json').read_text())
assert r['rank'] == int(sys.argv[3]) and r['world'] == 2
assert r['revision'] == '05c6cd0dff9df5bf20b10419c663a1e2adc7eb64'
assert (p / 'config.json').is_file(), 'Run download.sh to completion first'
PY
gpu=${GPU_UUID:-$(nvidia-smi --query-gpu=uuid --format=csv,noheader)}
[[ $gpu == GPU-* && $gpu != *$'\n'* ]] || { echo 'Set GPU_UUID to one GB10.' >&2; exit 2; }
[[ $(nvidia-smi -i "$gpu" --query-gpu=name --format=csv,noheader) == 'NVIDIA GB10' ]] || exit 2
if [[ -z ${NCCL_SOCKET_IFNAME:-} ]]; then
    NCCL_SOCKET_IFNAME=$(python3 - "$master" "$rank" <<'PY'
import json, subprocess, sys
master, rank = sys.argv[1:]
if rank == '0':
    links = json.loads(subprocess.check_output(['ip', '-j', 'address']))
    print(next(l['ifname'] for l in links if any(a.get('local') == master for a in l['addr_info'])))
else:
    print(json.loads(subprocess.check_output(['ip', '-j', 'route', 'get', master]))[0]['dev'])
PY
)
fi
name=${INKLING_CONTAINER:-tensorfold-inkling2-rank$rank}
image=${INKLING_IMAGE:-tensorfold-inkling2:spark}
if docker container inspect "$name" >/dev/null 2>&1; then
    echo "Container $name already exists; inspect and stop its owning process before starting again." >&2
    exit 2
fi
rdma=()
[[ ! -d /dev/infiniband ]] || rdma=(--device /dev/infiniband)
exec docker run --rm --name "$name" --network host --ipc host --gpus "device=$gpu" \
    --ulimit memlock=-1 --cap-add IPC_LOCK "${rdma[@]}" \
    --label ai.kortexa.inkling2.native=1 --label "ai.kortexa.inkling2.rank=$rank" \
    -e "TF_INKLING_RANK=$rank" -e TF_INKLING_WORLD=2 \
    -e "NCCL_COMM_ID=$master:${MASTER_PORT:-29597}" -e "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME" \
    -e NCCL_RUNTIME_CONNECT=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_CUMEM_HOST_ENABLE=0 \
    -e "TF_ROCE=${TF_ROCE:-1}" -e "TENSORFOLD_MEMORY_RESERVE_GIB=${RESERVE_GIB:-6}" \
    -v "$(cd "$model" && pwd):/model:ro" "$image" \
    serve /model --backend cuda --device 0 --context "${CONTEXT:-524288}" \
    --parallel 1 --no-drafts --no-thinking --temperature 0 --prompt-cache-gib 0 \
    --host "${API_HOST:-127.0.0.1}" --port "${API_PORT:-8000}" --name inkling-2-small \
    --no-update-check "$@"
