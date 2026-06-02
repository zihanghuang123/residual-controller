#!/bin/bash
# train_queue_blob.sh — H-sweep, slot A on blob (1 GPU).
#
# Slot A jobs: six_H2500 (~6.7 hr) + triple_H1000 (~1.3 hr) = ~8 hr wall time.
#
# Usage:
#   tmux new -s sweep
#   bash train_queue_blob.sh
#   Ctrl+B then D   (detach)
#   tmux attach -t sweep   (reattach)

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
        CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 \
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

# Slot A — long job first (LPT order).
SLOT_A=(
    "train/solve_trajectory.py plants/six_pendulum/config_H2500.py    solve_six_H2500"
    "train/train_pure.py       plants/six_pendulum/config_H2500.py    pure_six_H2500"
    "train/solve_trajectory.py plants/triple_pendulum/config_H1000.py solve_triple_H1000"
    "train/train_pure.py       plants/triple_pendulum/config_H1000.py pure_triple_H1000"
)

echo "=== blob queue started: $(date) ==="
total_start=$(date +%s)

run_track 0 "${SLOT_A[@]}"

total_elapsed=$(( $(date +%s) - total_start ))
total_h=$((total_elapsed / 3600))
total_m=$(((total_elapsed % 3600) / 60))
echo "=== blob queue finished: $(date) (total ${total_h}h${total_m}m) ==="
