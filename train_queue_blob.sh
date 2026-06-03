#!/bin/bash
# Curriculum BPTT for six_pendulum: 500 -> 1000 -> 1500 -> 2000 -> 2500.
# All stages share OUTPUT_DIR, so each warm-starts from the previous via pure_params.pkl.
#
# Usage:
#   tmux new -s curr
#   bash train_queue_blob.sh
#   Ctrl+B then D
#   tmux attach -t curr

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

CURRICULUM=(
    "train/train_pure.py plants/six_pendulum/config_curr_H500.py  curr_H500"
    "train/train_pure.py plants/six_pendulum/config_curr_H1000.py curr_H1000"
    "train/train_pure.py plants/six_pendulum/config_curr_H1500.py curr_H1500"
    "train/train_pure.py plants/six_pendulum/config_curr_H2000.py curr_H2000"
    "train/train_pure.py plants/six_pendulum/config_curr_H2500.py curr_H2500"
)

echo "=== blob curriculum started: $(date) ==="
total_start=$(date +%s)

run_track 0 "${CURRICULUM[@]}"

total_elapsed=$(( $(date +%s) - total_start ))
total_h=$((total_elapsed / 3600))
total_m=$(((total_elapsed % 3600) / 60))
echo "=== blob curriculum finished: $(date) (total ${total_h}h${total_m}m) ==="
