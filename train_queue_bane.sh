#!/bin/bash
# Queue jobs per GPU (bane, 2 GPUs). Fill GPU0/GPU1; entry = "<command> | <log name>".
# Empty list -> that GPU is left idle. Logs go to logs/<name>.log.
# tmux new -s train; bash train_queue_bane.sh; Ctrl+B then D to detach.

set -uo pipefail
mkdir -p logs

GPU0=(
    "python train/solve_trajectory.py --config plants/four_pendulum/config.py | solve_four"
)

GPU1=(
    "python train/solve_trajectory.py --config plants/triple_pendulum/config.py | solve_triple"
)

trim() { local s="$*"; s="${s#"${s%%[![:space:]]*}"}"; printf '%s' "${s%"${s##*[![:space:]]}"}"; }

run_gpu() {
    local gpu=$1; shift
    local jobs=("$@")
    local n=${#jobs[@]} i=0
    for entry in "${jobs[@]}"; do
        local cmd; cmd="$(trim "${entry%%|*}")"
        local name; name="$(trim "${entry##*|}")"
        local log="logs/${name}.log"
        i=$((i + 1))
        echo "[GPU $gpu] [$(date +%H:%M:%S)] $i/$n start $name"
        local start; start=$(date +%s)
        CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
            $cmd > "$log" 2>&1
        local status=$?
        local elapsed=$(( $(date +%s) - start ))
        if [ $status -eq 0 ]; then
            echo "[GPU $gpu] [$(date +%H:%M:%S)] done $name in $((elapsed / 60))m$((elapsed % 60))s"
        else
            echo "[GPU $gpu] [$(date +%H:%M:%S)] FAILED $name exit $status after $((elapsed / 60))m$((elapsed % 60))s"
        fi
    done
}

echo "=== queue started: $(date) ==="
total_start=$(date +%s)

pids=()
[ ${#GPU0[@]} -gt 0 ] && { run_gpu 0 "${GPU0[@]}" & pids+=($!); }
[ ${#GPU1[@]} -gt 0 ] && { run_gpu 1 "${GPU1[@]}" & pids+=($!); }
[ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"

total=$(( $(date +%s) - total_start ))
echo "=== queue finished: $(date) (total $((total / 3600))h$(((total % 3600) / 60))m) ==="
