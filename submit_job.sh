#!/bin/bash

set -euo pipefail

# Get RESULT_DATABASE_PATH and EXPERIMENT_NAME from db_info.sh
source db_info.sh

# Get current date and time and store in variable DATE
# Used to name the subdirectory (within the main experiment directory)
# where outputs are stored.
# printf -v DATE '%(%Y-%m-%d_%H:%M:%S)T\n' -1

save_dir="${RESULT_DATABASE_PATH}/${EXPERIMENT_NAME}/${SLURM_JOB_ID}"
json="${RESULT_DATABASE_PATH}/${EXPERIMENT_NAME}/${SLURM_JOB_ID}/result.json"

if [ ! -d "${save_dir}" ]; then
    mkdir -p "${save_dir}"
fi

module load julia/1.11.5 git

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

executable=$1

x=$2
Q=$3
beta=$4
xmax=$5
NEVAL=$6
Q0=$7
x0=$8
lambda=$9

# Run the task:
echo "Partition: $SLURM_JOB_PARTITION"
echo "Number of threads: $SLURM_CPUS_PER_TASK" #Useful to output this for running strong scaling experiments etc

starttime=$(date +%s%N)
echo "Job started at: $(date)"

julia $executable ${x} ${Q} ${beta} ${xmax} --neval ${NEVAL} --Q0 ${Q0} --x0 ${x0} --lambda ${lambda} --save_dir $save_dir --json $json

endtime=$(date +%s%N)
echo "Job finished at: $(date)"
elapsedtime=$((endtime - starttime))
printf "Job duration: %s.%s seconds\n" "${elapsedtime:0: -9}" "${elapsedtime: -9:3}"


