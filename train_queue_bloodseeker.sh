#!/bin/bash
# train_queue_bloodseeker.sh — H-sweep, slots B+C on bloodseeker (2 GPUs).
#
# Slot B (GPU 0): five_H2500 + four_H1000  (~7.4 hr)
# Slot C (GPU 1): six_H2000 + triple_H1500 + triple_H500  (~8.0 hr)
#
# Usage:
#   tmux new -s sweep
#   bash train_queue_bloodseeker.sh
#   Ctrl+B then D   (detach)
#   tmux attach -t sweep

set -uo pipefail
mkdir -p logs

run_track() {
    local gpu=$1
    shift
    local jobs=("$@")
    local n=${#jobs[@]}
    local i=0
    for job in "${jobs[@]}"; do
        read -r script config tag <<< "$job"
        local log="logs/${tag}.log"
        i=$((i + 1))
        echo "[GPU $gpu] [$(date +%H:%M:%S)] $i/$n starting $tag"
        local start=$(date +%s)
        CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
            python "$script" --config "$config" > "$log" 2>&1
        local status=$?
        local elapsed=$(( $(date +%s) - start ))
        local mm=$((elapsed / 60))
        local ss=$((elapsed % 60))
        if [ $status -eq 0 ]; then
            echo "[GPU $gpu] [$(date +%H:%M:%S)] done $tag in ${mm}m${ss}s"
        else
            echo "[GPU $gpu] [$(date +%H:%M:%S)] FAILED $tag exit $status after ${mm}m${ss}s"
        fi
    done
}

# Slot B — GPU 0
SLOT_B=(
    "train/train_supervised.py        plants/six_pendulum/config1.py  train_supervised_six_1"
    "train/train_pure.py              plants/six_pendulum/config1.py  train_pure_six_1"
)

# Slot C — GPU 1
SLOT_C=(
    "train/train_supervised.py        plants/six_pendulum/config.py  train_supervised_six"
    "train/train_pure.py              plants/six_pendulum/config.py  train_pure_six"
)

echo "=== bloodseeker queue started: $(date) ==="
total_start=$(date +%s)

run_track 0 "${SLOT_B[@]}" &
pid_b=$!
run_track 1 "${SLOT_C[@]}" &
pid_c=$!

wait $pid_b $pid_c

total_elapsed=$(( $(date +%s) - total_start ))
total_h=$((total_elapsed / 3600))
total_m=$(((total_elapsed % 3600) / 60))
echo "=== bloodseeker queue finished: $(date) (total ${total_h}h${total_m}m) ==="