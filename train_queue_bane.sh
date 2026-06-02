#!/bin/bash
# train_queue_bane.sh — H-sweep, slots D+E on bane (2 GPUs).
#
# Slot D (GPU 0): five_H2000 + four_H1500 + double_H750            (~7.8 hr)
# Slot E (GPU 1): four_H2000 + five_H1500 + double_H500 + double_H250  (~7.6 hr)
#
# Usage:
#   tmux new -s sweep
#   bash train_queue_bane.sh
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

# Slot D — GPU 0
SLOT_D=(
    "train/solve_trajectory.py plants/five_pendulum/config_H2000.py   solve_five_H2000"
    "train/train_pure.py       plants/five_pendulum/config_H2000.py   pure_five_H2000"
    "train/solve_trajectory.py plants/four_pendulum/config_H1500.py   solve_four_H1500"
    "train/train_pure.py       plants/four_pendulum/config_H1500.py   pure_four_H1500"
    "train/solve_trajectory.py plants/double_pendulum/config_H750.py  solve_double_H750"
    "train/train_pure.py       plants/double_pendulum/config_H750.py  pure_double_H750"
)

# Slot E — GPU 1
SLOT_E=(
    "train/solve_trajectory.py plants/four_pendulum/config_H2000.py   solve_four_H2000"
    "train/train_pure.py       plants/four_pendulum/config_H2000.py   pure_four_H2000"
    "train/solve_trajectory.py plants/five_pendulum/config_H1500.py   solve_five_H1500"
    "train/train_pure.py       plants/five_pendulum/config_H1500.py   pure_five_H1500"
    "train/solve_trajectory.py plants/double_pendulum/config_H500.py  solve_double_H500"
    "train/train_pure.py       plants/double_pendulum/config_H500.py  pure_double_H500"
    "train/solve_trajectory.py plants/double_pendulum/config_H250.py  solve_double_H250"
    "train/train_pure.py       plants/double_pendulum/config_H250.py  pure_double_H250"
)

echo "=== bane queue started: $(date) ==="
total_start=$(date +%s)

run_track 0 "${SLOT_D[@]}" &
pid_d=$!
run_track 1 "${SLOT_E[@]}" &
pid_e=$!

wait $pid_d $pid_e

total_elapsed=$(( $(date +%s) - total_start ))
total_h=$((total_elapsed / 3600))
total_m=$(((total_elapsed % 3600) / 60))
echo "=== bane queue finished: $(date) (total ${total_h}h${total_m}m) ==="
