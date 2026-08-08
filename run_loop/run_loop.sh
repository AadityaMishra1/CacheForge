#!/bin/bash
#BSUB -n 11
#BSUB -R "span[hosts=1]"
#BSUB -W 2:00
#BSUB -o ../logs/%J.out
#BSUB -e ../logs/%J.err
#BSUB -R "rusage[mem=4.00/task]"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

source ~/.bashrc
conda activate /share/csc491006f25/conda_env/HW1_env/

python3 run_loop.py