#!/bin/bash
# Queue jobs per GPU (bloodseeker, 2 GPUs). Fill GPU0/GPU1; entry = "<command> | <log name>".
# Entries run sequentially per GPU; a job is its setup (solve or cp trajectories) then train.
# Empty list -> that GPU is left idle. Logs go to logs/<name>.log.
# tmux new -s train; bash train_queue_bloodseeker.sh; Ctrl+B then D to detach.

set -uo pipefail
mkdir -p logs

GPU0=(
    "mkdir -p outputs/kinova/h1 | kinova_h1"
    "cp outputs/kinova/config/trajectories.npz outputs/kinova/h1/trajectories.npz | kinova_h1"
    "python train/train_pure_rnn.py --config plants/kinova/h1.py | kinova_h1"

    "mkdir -p outputs/kinova/h50 | kinova_h50"
    "cp outputs/kinova/config/trajectories.npz outputs/kinova/h50/trajectories.npz | kinova_h50"
    "python train/train_pure_rnn.py --config plants/kinova/h50.py | kinova_h50"

    "mkdir -p outputs/kinova/h300 | kinova_h300"
    "cp outputs/kinova/config/trajectories.npz outputs/kinova/h300/trajectories.npz | kinova_h300"
    "python train/train_pure_rnn.py --config plants/kinova/h300.py | kinova_h300"
)

GPU1=(
    "mkdir -p outputs/kinova/h10 | kinova_h10"
    "cp outputs/kinova/config/trajectories.npz outputs/kinova/h10/trajectories.npz | kinova_h10"
    "python train/train_pure_rnn.py --config plants/kinova/h10.py | kinova_h10"

    "mkdir -p outputs/kinova/h100 | kinova_h100"
    "cp outputs/kinova/config/trajectories.npz outputs/kinova/h100/trajectories.npz | kinova_h100"
    "python train/train_pure_rnn.py --config plants/kinova/h100.py | kinova_h100"

    "mkdir -p outputs/kinova/h200 | kinova_h200"
    "cp outputs/kinova/config/trajectories.npz outputs/kinova/h200/trajectories.npz | kinova_h200"
    "python train/train_pure_rnn.py --config plants/kinova/h200.py | kinova_h200"
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
        CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 \
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
