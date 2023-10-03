#!/bin/bash
#$ -cwd

NAME=$1
BACKEND=$2
TASKS=$3

# Our NumPy is compiled with OpenBLAS and the job scheduler assumes one
# thread per job. This might need to be adjusted on another machine.
export OPENBLAS_NUM_THREADS=1

source .venv/bin/activate
python run.py --name ${NAME} --backend ${BACKEND} --tasks ${TASKS}
