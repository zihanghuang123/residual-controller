#!/bin/bash
# Queue jobs per GPU (bloodseeker, 2 GPUs). Fill GPU0/GPU1; entry = "<command> | <log name>".
# Entries run sequentially per GPU, so each job is listed as mkdir / cp trajectories / train.
# Empty list -> that GPU is left idle. Logs go to logs/<name>.log.
# tmux new -s train; bash train_queue_bloodseeker.sh; Ctrl+B then D to detach.

set -uo pipefail
mkdir -p logs

GPU0=(
    "mkdir -p outputs/four_pendulum/h1200 | four_h1200"
    "cp outputs/four_pendulum/config/trajectories.npz outputs/four_pendulum/h1200/trajectories.npz | four_h1200"
    "python train/train_pure_rnn.py --config plants/four_pendulum/h1200.py | four_h1200"

    "mkdir -p outputs/four_pendulum/h700 | four_h700"
    "cp outputs/four_pendulum/config/trajectories.npz outputs/four_pendulum/h700/trajectories.npz | four_h700"
    "python train/train_pure_rnn.py --config plants/four_pendulum/h700.py | four_h700"

    "mkdir -p outputs/six_pendulum/h500 | six_h500"
    "cp outputs/six_pendulum/config/trajectories.npz outputs/six_pendulum/h500/trajectories.npz | six_h500"
    "python train/train_pure_rnn.py --config plants/six_pendulum/h500.py | six_h500"

    "mkdir -p outputs/six_pendulum/h400 | six_h400"
    "cp outputs/six_pendulum/config/trajectories.npz outputs/six_pendulum/h400/trajectories.npz | six_h400"
    "python train/train_pure_rnn.py --config plants/six_pendulum/h400.py | six_h400"
)

GPU1=(
    "mkdir -p outputs/four_pendulum/h1100 | four_h1100"
    "cp outputs/four_pendulum/config/trajectories.npz outputs/four_pendulum/h1100/trajectories.npz | four_h1100"
    "python train/train_pure_rnn.py --config plants/four_pendulum/h1100.py | four_h1100"

    "mkdir -p outputs/four_pendulum/h800 | four_h800"
    "cp outputs/four_pendulum/config/trajectories.npz outputs/four_pendulum/h800/trajectories.npz | four_h800"
    "python train/train_pure_rnn.py --config plants/four_pendulum/h800.py | four_h800"

    "mkdir -p outputs/six_pendulum/h600 | six_h600"
    "cp outputs/six_pendulum/config/trajectories.npz outputs/six_pendulum/h600/trajectories.npz | six_h600"
    "python train/train_pure_rnn.py --config plants/six_pendulum/h600.py | six_h600"

    "mkdir -p outputs/six_pendulum/h300 | six_h300"
    "cp outputs/six_pendulum/config/trajectories.npz outputs/six_pendulum/h300/trajectories.npz | six_h300"
    "python train/train_pure_rnn.py --config plants/six_pendulum/h300.py | six_h300"
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
