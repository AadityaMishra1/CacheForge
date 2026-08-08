#!/bin/bash

POLICY_FILE="new_policies/real_data/006_hawk_stride_tinylfu_ensemble__hst_e___v0.cc"
POLICY_NAME="HST-E"

# Compile
echo "Compiling ${POLICY_NAME}..."
g++ -Wall -std=c++17 \
    -I./inc \
    -I./champ_repl_pol \
    "${POLICY_FILE}" \
    lib/config2.a \
    -o "hst_e_test.out"

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Compilation successful!"

# Workloads and their trace files
declare -A WORKLOADS=(
    ["astar"]="traces_gz/astar_313B.trace.gz"
    ["calculix"]="traces_gz/calculix_2670B.trace.gz"
    ["gcc"]="traces_gz/gcc_39B.trace.gz"
    ["hmmer"]="traces_gz/hmmer_397B.trace.gz"
    ["lbm"]="traces_gz/lbm_564B.trace.gz"
    ["mcf"]="traces_gz/mcf_46B.trace.gz"
    ["milc"]="traces_gz/milc_409B.trace.gz"
    ["namd"]="traces_gz/namd_591B.trace.gz"
    ["omnetpp"]="traces_gz/omnetpp_340B.trace.gz"
    ["zeusmp"]="traces_gz/zeusmp_100B.trace.gz"
)

# Run on all workloads
mkdir -p test_results
for workload in "${!WORKLOADS[@]}"; do
    trace="${WORKLOADS[$workload]}"
    echo ""
    echo "========================================="
    echo "Running ${POLICY_NAME} on ${workload}..."
    echo "========================================="
    
    ./hst_e_test.out \
        -warmup_instructions 1000000 \
        -simulation_instructions 100000000 \
        -traces "${trace}" \
        > "test_results/${workload}_output.txt" 2>&1
    
    # Extract IPC
    ipc=$(grep "CPU 0 cumulative IPC:" "test_results/${workload}_output.txt" | awk '{print $5}')
    llc_hit=$(grep "LLC TOTAL" "test_results/${workload}_output.txt" | grep "HIT" | awk '{print $6}')
    
    echo "${workload}: IPC=${ipc}, LLC_HIT=${llc_hit}"
    echo "${workload},${ipc},${llc_hit}" >> test_results/summary.csv
done

echo ""
echo "========================================="
echo "All tests complete! Results in test_results/"
echo "========================================="
