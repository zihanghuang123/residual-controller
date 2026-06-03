#!/bin/bash
# train_queue_bloodseeker.sh — run jobs sequentially, surviving disconnect.
#
# Usage:
#   tmux new -s train
#   bash train_queue_bloodseeker.sh
#   Ctrl+B then D
#   tmux attach -t train

set -uo pipefail

mkdir -p logs

JOBS=(
    "train/solve_trajectory.py        plants/six_pendulum/config.py  solve_six"
    "train/build_supervised_dataset.py plants/six_pendulum/config.py  build_supervised_six"
    "eval/plot_residual_check.py      plants/six_pendulum/config.py  validate_dataset_six"
    "train/train_supervised.py        plants/six_pendulum/config.py  train_supervised_six"
    "train/train_pure.py              plants/six_pendulum/config.py  train_pure_six"
)

echo "=== Queue started: $(date) ==="
echo "${#JOBS[@]} jobs to run sequentially"
echo

total_start=$(date +%s)

for i in "${!JOBS[@]}"; do
    read -r script config tag <<< "${JOBS[$i]}"
    log="logs/${tag}.log"
    n=$((i + 1))

    echo "----------------------------------------"
    echo "[$(date +%H:%M:%S)] Job $n/${#JOBS[@]}: $tag"
    echo "  cmd: python $script --config $config"
    echo "  log: $log"

    start=$(date +%s)
    PYTHONUNBUFFERED=1 python "$script" --config "$config" > "$log" 2>&1
    status=$?
    elapsed=$(( $(date +%s) - start ))
    mm=$((elapsed / 60))
    ss=$((elapsed % 60))

    if [ $status -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] done in ${mm}m${ss}s"
    else
        echo "[$(date +%H:%M:%S)] FAILED (exit $status) after ${mm}m${ss}s — continuing"
    fi
done

total_elapsed=$(( $(date +%s) - total_start ))
total_mm=$((total_elapsed / 60))
echo
echo "=== Queue finished: $(date) (total ${total_mm}m) ==="
