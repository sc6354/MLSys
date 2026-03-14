#!/bin/bash
# Run all benchmarks with a 60s cooldown between each to avoid rate limits

BENCHMARKS=(1 5 9 13 17)

for i in "${!BENCHMARKS[@]}"; do
    n=${BENCHMARKS[$i]}
    echo "=== Running benchmark $n ==="
    python agent.py benchmarks/mlsys-2026-$n.json output_$n.json
    if [ $i -lt $((${#BENCHMARKS[@]} - 1)) ]; then
        echo "Cooling down 120s before next benchmark..."
        sleep 120
    fi
done

echo "All benchmarks done."
