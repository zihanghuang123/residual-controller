#!/bin/bash
# train_queue.sh — run training jobs sequentially, surviving disconnect.
#
# Usage:
#   tmux new -s train               # so it survives SSH disconnect
#   bash train_queue.sh
#   Ctrl+B then D                   # detach without stopping
#   # ... go to sleep ...
#   tmux attach -t train            # re-attach next morning
#
# Or with nohup if you don't want tmux:
#   nohup bash train_queue.sh > queue.log 2>&1 &
#   disown

set -uo pipefail

mkdir -p logs

# Each line: "script  config_path  tag"  (tag is used for log filename + status messages).
# Edit this list to change what runs and in what order.
JOBS=(
    "train/train_pure.py   triple_pendulum/config.py   pure_small"
    "train/train_oracle.py triple_pendulum/config.py   oracle_small"
    "train/train_pure.py   triple_pendulum/config1.py  pure_med"
    "train/train_oracle.py triple_pendulum/config1.py  oracle_med"
    "train/train_pure.py   triple_pendulum/config2.py  pure_big"
    "train/train_oracle.py triple_pendulum/config2.py  oracle_big"
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
