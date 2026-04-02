#!/bin/bash

#set -euo pipefail

# SLURM parameters
PARTITION="small"
GRES="gpu:v100:1"
ACCOUNT="lappi"
NTASKS=1
CPUS_PER_TASK=8
TIME="00:12:00"  # HH:MM:SS format
DRYRUN=1  # Set to 1 for dry run mode

# Parameters for generating beta values corresponding to uniform spacing in MX
M_PI=0.135
M_MIN=$(echo "2*$M_PI" | bc -l)
M_MIN_SQ=$(echo "$M_MIN*$M_MIN" | bc -l)
XPMAX=0.1

# Parameters for generating beta values for fixed slices in x_Bj
XPOM_MAX=0.01 #XPOM_MIN will be set to x_Bj

# Parameter lists
XBJ_LIST=(1e-3 1e-4)

XPOM_LIST=(1e-3 1e-4)
BETA_LIST=()
Q_LIST=(2.0)
XMAX_LIST=(40.0)

# Get EXPERIMENT_NAME and SLRM_OUTPUT_DIR from db_info.sh
source db_info.sh

# Other parameters
NEVAL=4e10
MEMORY="4G"
LANGUAGE="python"
EXEC="photon_T_grid.py"
DIPOLE="bk_kcbk_pd_map.dat"
PARAM_FILE="HERA_kinematic_points.txt" # File with Q, beta and xpom values
JOB_SCRIPT="submit_job.sh" #The job script takes code name, parameters etc as arguments

# File to store submitted job IDs (for watcher script)
JOBID_FILE="$RESULT_DATABASE_PATH/$EXPERIMENT_NAME/submitted_jobs.txt"
: > "$JOBID_FILE"   # truncate / create new

if [ "$DRYRUN" -eq 1 ]; then
	echo "DRY RUN:"
	echo ""
fi
if [ ! -f "$PARAM_FILE" ]; then
    echo "Error: $PARAM_FILE not found!"
    exit 1
fi

# Loop over each line in the parameter file
for XMAX in "${XMAX_LIST[@]}"; do
    while read -r Q_VAL BETA XPOM; do
        
        Q=$Q_VAL 

        echo "Submitting: Q=$Q_VAL, BETA=$BETA, XPOM=$XPOM"

        CMD="sbatch --parsable --partition=$PARTITION --account=$ACCOUNT --ntasks=$NTASKS --cpus-per-task=$CPUS_PER_TASK --mem=$MEMORY --gres=$GRES --time=$TIME --job-name=\"${EXPERIMENT_NAME}\" --output=\"${SLRM_OUTPUT_DIR}/${EXPERIMENT_NAME}_%j.out\" --error=\"${SLRM_OUTPUT_DIR}/${EXPERIMENT_NAME}_%j.err\" ${JOB_SCRIPT} ${LANGUAGE} ${EXEC} $Q $BETA $XPOM $XMAX $NEVAL $DIPOLE"

        if [ "$DRYRUN" -eq 1 ]; then
            echo "$CMD"
        else
            jid=$(eval "$CMD")
            echo "$jid" >> "$JOBID_FILE"
        fi

    done < "$PARAM_FILE"
done
echo "All jobs submitted. Job IDs written to $JOBID_FILE"
echo "Run ./wait_and_collect.sh to wait for completion and run the collector."
