#!/bin/bash
# Crocoddyl trajectory generation on bane — four + triple in parallel.
set -uo pipefail
mkdir -p logs

run_plant() {
    local plant=$1
    local log="logs/solve_${plant}.log"
    echo "[$(date +%H:%M:%S)] starting solve $plant"
    local start=$(date +%s)
    PYTHONUNBUFFERED=1 python train/solve_trajectory.py \
        --config "plants/${plant}_pendulum/config.py" > "$log" 2>&1
    local status=$?
    local mm=$(( ($(date +%s) - start) / 60 ))
    local ss=$(( ($(date +%s) - start) % 60 ))
    if [ $status -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] done $plant in ${mm}m${ss}s"
    else
        echo "[$(date +%H:%M:%S)] FAILED $plant exit $status after ${mm}m${ss}s"
    fi
}

echo "=== bane solve queue started: $(date) ==="
total_start=$(date +%s)

run_plant four &
pid_a=$!
run_plant triple &
pid_b=$!
wait $pid_a $pid_b

total=$(( $(date +%s) - total_start ))
echo "=== bane queue finished: $(date) (total $((total / 60))m$((total % 60))s) ==="
