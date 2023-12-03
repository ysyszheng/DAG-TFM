#!/bin/bash

cleanup() {
    echo "Terminating all child processes..."
    pkill -P $$
    exit 0
}

trap 'cleanup' SIGINT SIGTERM

for lambd in 1 2 3 5 10
do
    # First-Price
    python3 -u ./run.py --method ES --mode train --cfg ./config/es.yaml --lambd $lambd --is_burn 0 --a &

    # Log-Burn
    for a in 1 0.01 100
    do
        python3 -u ./run.py --method ES --mode train --cfg ./config/es.yaml --lambd $lambd --is_burn 1 --a $a &
    done
    
    # Poly-Burn
    for a in 0.5 0.01
    do
        python3 -u ./run.py --method ES --mode train --cfg ./config/es.yaml --lambd $lambd --is_burn 2 --a $a &
    done
done

wait
