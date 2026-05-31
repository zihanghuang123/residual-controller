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
    # --- triple pendulum (3 links) ---
    "train/train_pure.py   plants/triple_pendulum/config.py  pure_triple_huge"
    "train/train_oracle.py plants/triple_pendulum/config.py  oracle_triple_huge"

    # --- four pendulum (4 links) ---
    "train/train_pure.py   plants/four_pendulum/config.py  pure_four_huge"
    "train/train_oracle.py plants/four_pendulum/config.py  oracle_four_huge"

    # --- five pendulum (5 links) ---
    "train/train_pure.py   plants/five_pendulum/config.py  pure_five_huge"
    "train/train_oracle.py plants/five_pendulum/config.py  oracle_five_huge"

    # --- six pendulum (6 links) ---
    "train/train_pure.py   plants/six_pendulum/config.py  pure_six_huge"
    "train/train_oracle.py plants/six_pendulum/config.py  oracle_six_huge"

    # --- seven pendulum (7 links) ---
    "train/train_pure.py   plants/seven_pendulum/config.py  pure_seven_huge"
    "train/train_oracle.py plants/seven_pendulum/config.py  oracle_seven_huge"
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
