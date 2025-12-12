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

# Parameters for generating beta values for fixed slices in x_Bj
XPOM_MAX=0.01 #XPOM_MIN will be set to x_Bj

# Parameter lists
XBJ_LIST=(1e-3 1e-4)

XPOM_LIST=(1e-3 1e-4)
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
				for Q in "${Q_LIST[@]}"; do
					for XBJ in "${XBJ_LIST[@]}"; do
						XPOM_MIN=XBJ
						xPvals=()
						for t in $(seq 0 0.1 1); do
	    						logxP=$(awk -v xmin="$xPmin" -v xmax="$xPmax" -v t="$t" \
							    'BEGIN{print log(xmin) + (log(xmax) - log(xmin)) * t}')
							    xP=$(awk -v v="$logxP" 'BEGIN{print exp(v)}')
							    xPvals+=("$xP")
						done
						
						# --- Rounding step: xPvals = round.(xPvals .* 1e15) ./ 1e15 ---
						rounded_xPvals=()
						for v in "${xPvals[@]}"; do
						    rounded=$(awk -v x="$v" 'BEGIN{printf "%.15f", (round(x*1e15)/1e15)}')
						    rounded_xPvals+=("$rounded")
						done
						
						# --- Compute beta_vals = x ./ xPvals ---
						BETA_LIST=()
						for xp in "${rounded_xPvals[@]}"; do
						    beta=$(awk -v x="$x" -v xp="$xp" 'BEGIN{print x/xp}')
						    BETA_LIST+=("$beta")
						done
						
						for BETA in "${BETA_LIST[@]}"; do
							XPOM=$(awk -v x="$XBJ" -v b="$BETA" 'BEGIN{print x/b}')		
							
							echo "Submitting job in experiment $EXPERIMENT_NAME with xmax = $XMAX, x_pom=$XPOM, Q=$Q and BETA=$BETA"
							CMD="sbatch --parsable --partition=$PARTITION --account=$ACCOUNT --ntasks=$NTASKS --cpus-per-task=$CPUS_PER_TASK --mem=$MEMORY --time=$TIME --job-name=\"${EXPERIMENT_NAME}\" --output=\"${SLRM_OUTPUT_DIR}/${EXPERIMENT_NAME}_%j.out\" --error=\"${SLRM_OUTPUT_DIR}/${EXPERIMENT_NAME}_%j.err\" ${JOB_SCRIPT} ${EXEC} $Q $BETA $XPOM $XMAX $NEVAL $Q0 $X0 $LAMBDA"
						
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
