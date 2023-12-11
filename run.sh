#!/bin/bash

declare -a children=()

cleanup() {
    echo "Terminating all child processes..."
    for pid in "${children[@]}"; do
        echo "Killing process $pid"
        pkill -P "$pid"  # Attempt to kill child processes of each child
        kill "$pid"      # Kill the child process itself
    done
    exit 0
}

trap 'cleanup' SIGINT SIGTERM

function run_python() {
    python3 -u ./run.py "$@" &
    children+=($!)
}

for lambd in 1 3 5 10
do
    # First-Price
    run_python --method ES --mode train --cfg ./config/es.yaml --lambd "$lambd" --burn_flag 'non'

    # # Log-Burn
    # for a in 1 0.01 100
    # do
    #     run_python --method ES --mode train --cfg ./config/es.yaml --lambd "$lambd" --burn_flag 'log' --a "$a"
    # done
    
    # # Poly-Burn
    # for a in 0.5 0.01
    # do
    #     run_python --method ES --mode train --cfg ./config/es.yaml --lambd "$lambd" --burn_flag 'poly' --a "$a"
    # done
done

wait
