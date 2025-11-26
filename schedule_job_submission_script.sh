#!/bin/bash

set -euo pipefail

# SLURM parameters
PARTITION="small"
ACCOUNT="lappi"
NTASKS=1
CPUS_PER_TASK=16
TIME="1-23:00:00"  # HH:MM:SS format
DRYRUN=0  # Set to 1 for dry run mode

# Parameter lists
X_LIST=(1e-2)
BETA_LIST=(0.9821012055292299 0.521811049609881 0.35529279679383774 0.26934158775519274 0.2168757529655048 0.1815174404225726 0.15607222398237006 0.13688377916950556 0.1218970347936801 0.10986810608534718 0.1)
Q_LIST=(2.0)
XMAX_LIST=(40.0)
Q0_LIST=(1.0)
X0_LIST=(3.04e-4)
LAMBDA_LIST=(0.288)

# Get EXPERIMENT_NAME and SLRM_OUTPUT_DIR from db_info.sh
source db_info.sh

# Other parameters
NEVAL=4e10
MEMORY="4G"
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
for Q0 in "${Q0_LIST[@]}"; do
	for X0 in "${X0_LIST[@]}"; do
		for LAMBDA in "${LAMBDA_LIST[@]}"; do
			for X in "${X_LIST[@]}"; do
				for XMAX in "${XMAX_LIST[@]}"; do
					for BETA in "${BETA_LIST[@]}"; do
						for Q in "${Q_LIST[@]}"; do
							echo "Submitting job in experiment $EXPERIMENT_NAME with xmax = $XMAX, x=$X, Q=$Q and BETA=$BETA"
							CMD="sbatch --parsable --partition=$PARTITION --account=$ACCOUNT --ntasks=$NTASKS --cpus-per-task=$CPUS_PER_TASK --mem=$MEMORY --time=$TIME --job-name=\"${EXPERIMENT_NAME}\" --output=\"${SLRM_OUTPUT_DIR}/${EXPERIMENT_NAME}_%j.out\" --error=\"${SLRM_OUTPUT_DIR}/${EXPERIMENT_NAME}_%j.err\" ${JOB_SCRIPT} ${EXEC} $X $Q $BETA $XMAX $NEVAL $Q0 $X0 $LAMBDA"
					
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
			done
		done
	done
done
echo "All jobs submitted. Job IDs written to $JOBID_FILE"
echo "Run ./wait_and_collect.sh to wait for completion and run the collector."
