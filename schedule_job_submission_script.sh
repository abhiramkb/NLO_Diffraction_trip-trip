#!/bin/bash

set -euo pipefail

# SLURM parameters
PARTITION="large"
ACCOUNT="lappi"
NTASKS=1
CPUS_PER_TASK=8
TIME="23:00:00"  # HH:MM:SS format
DRYRUN=0  # Set to 1 for dry run mode

# Parameter lists
BETA_LIST=(0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
Q_LIST=(3.1622776601683795 4.47213595499958 7.0710678118654755 10.0 14.142135623730951 22.360679774997898)
XMAX_LIST=(40.0)

# Get EXPERIMENT_NAME and SLRM_OUTPUT_DIR from db_info.sh
source db_info.sh

# Other parameters
NEVAL=1e9
MEMORY="1G"
EXEC="photon_T.jl"
JOB_SCRIPT="submit_job.sh" #The job script takes code name, parameters etc as arguments

# File to store submitted job IDs (for watcher script)
JOBID_FILE="$RESULT_DATABASE_PATH/$EXPERIMENT_NAME/submitted_jobs.txt"
: > "$JOBID_FILE"   # truncate / create new

if [ "$DRYRUN" -eq 1 ]; then
	echo "DRY RUN:"
	echo ""
fi

# Loop over parameter combinations
for XMAX in "${XMAX_LIST[@]}"; do
    for BETA in "${BETA_LIST[@]}"; do
        for Q in "${Q_LIST[@]}"; do
            echo "Submitting job in experiment $EXPERIMENT_NAME with xmax = $XMAX, Q=$Q and BETA=$BETA"
            CMD="sbatch --parsable --partition=$PARTITION --account=$ACCOUNT --ntasks=$NTASKS --cpus-per-task=$CPUS_PER_TASK --mem=$MEMORY --time=$TIME --job-name=\"${EXPERIMENT_NAME}\" --output=\"${SLRM_OUTPUT_DIR}/${EXPERIMENT_NAME}_%j.out\" --error=\"${SLRM_OUTPUT_DIR}/${EXPERIMENT_NAME}_%j.err\" ${JOB_SCRIPT} ${EXEC} $Q $BETA $XMAX $NEVAL"
        
            if [ "$DRYRUN" -eq 1 ]; then
                echo "$CMD"
            else
                jid=$(eval "$CMD")  # --parsable ensures only job ID is returned
                echo "Submitted job $jid"
                echo "$jid" >> "$JOBID_FILE"
            fi
        done
    done
done

echo "All jobs submitted. Job IDs written to $JOBID_FILE"
echo "Run ./wait_and_collect.sh to wait for completion and run the collector."
