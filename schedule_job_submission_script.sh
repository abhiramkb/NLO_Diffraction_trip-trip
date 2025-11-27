#!/bin/bash

#set -euo pipefail

# SLURM parameters
PARTITION="small"
ACCOUNT="lappi"
NTASKS=1
CPUS_PER_TASK=16
TIME="1-23:00:00"  # HH:MM:SS format
DRYRUN=1  # Set to 1 for dry run mode

# Parameters for generating beta values corresponding to uniform spacing in MX
M_PI=0.135
M_MIN=$(echo "2*$M_PI" | bc -l)
M_MIN_SQ=$(echo "$M_MIN*$M_MIN" | bc -l)
XPMAX=0.1

# Parameter lists
X_LIST=(1e-2)
BETA_LIST=()
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
			for XMAX in "${XMAX_LIST[@]}"; do
				for X in "${X_LIST[@]}"; do
					for Q in "${Q_LIST[@]}"; do
						# Convert scientific-notation X → decimal so bc can read it
						X_DEC=$(printf "%.15f" "$X")
						QSQ=$(echo "$Q*$Q" | bc -l)

        					# Compute M_max^2
						MMAX_SQ=$(echo "$QSQ * ($XPMAX/$X_DEC - 1)" | bc -l)

						BETA_LIST=()
		
						for i in $(seq 0 0.1 0.2); do
					        	Msq=$(echo "$M_MIN_SQ + ($MMAX_SQ - $M_MIN_SQ)*$i" | bc -l)
					        	beta=$(echo "$QSQ / ($QSQ + $Msq)" | bc -l)
					        	BETA_LIST+=("$beta")    # append to bash array
					        done
        					echo
        					for BETA in "${BETA_LIST[@]}"; do						
							echo "Submitting job in experiment $EXPERIMENT_NAME with xmax = $XMAX, x=$X, Q=$Q and BETA=$BETA"
							CMD="sbatch --parsable --partition=$PARTITION --account=$ACCOUNT --ntasks=$NTASKS --cpus-per-task=$CPUS_PER_TASK --mem=$MEMORY --time=$TIME --job-name=\"${EXPERIMENT_NAME}\" --output=\"${SLRM_OUTPUT_DIR}/${EXPERIMENT_NAME}_%j.out\" --error=\"${SLRM_OUTPUT_DIR}/${EXPERIMENT_NAME}_%j.err\" ${JOB_SCRIPT} ${EXEC} $Q $BETA $X $XMAX $NEVAL $Q0 $X0 $LAMBDA"
					
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
