#!/bin/bash
# Queue jobs on GPU0 (blob, 1 GPU). Fill GPU0; entry = "<command> | <log name>".
# Empty list -> nothing runs. Logs go to logs/<name>.log.
# tmux new -s train; bash train_queue_blob.sh; Ctrl+B then D to detach.

set -uo pipefail
mkdir -p logs

GPU0=(
    "mkdir -p outputs/four_pendulum/h1000 | four_h1000"
    "cp outputs/four_pendulum/config/trajectories.npz outputs/four_pendulum/h1000/trajectories.npz | four_h1000"
    "python train/train_pure_rnn.py --config plants/four_pendulum/h1000.py | four_h1000"

    "mkdir -p outputs/four_pendulum/h900 | four_h900"
    "cp outputs/four_pendulum/config/trajectories.npz outputs/four_pendulum/h900/trajectories.npz | four_h900"
    "python train/train_pure_rnn.py --config plants/four_pendulum/h900.py | four_h900"

    "mkdir -p outputs/six_pendulum/h700 | six_h700"
    "cp outputs/six_pendulum/config/trajectories.npz outputs/six_pendulum/h700/trajectories.npz | six_h700"
    "python train/train_pure_rnn.py --config plants/six_pendulum/h700.py | six_h700"

    "mkdir -p outputs/six_pendulum/h200 | six_h200"
    "cp outputs/six_pendulum/config/trajectories.npz outputs/six_pendulum/h200/trajectories.npz | six_h200"
    "python train/train_pure_rnn.py --config plants/six_pendulum/h200.py | six_h200"
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

[ ${#GPU0[@]} -gt 0 ] && run_gpu 0 "${GPU0[@]}"

total=$(( $(date +%s) - total_start ))
echo "=== queue finished: $(date) (total $((total / 3600))h$(((total % 3600) / 60))m) ==="
