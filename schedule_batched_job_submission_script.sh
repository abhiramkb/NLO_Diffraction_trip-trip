#!/bin/bash

DRYRUN=0 

# --- Standard Parameters ---
PARTITION="gputest"
GRES="gpu:v100:1"
ACCOUNT="lappi"
NTASKS=1
CPUS_PER_TASK=8
TIME="00:15:00"
MEMORY="10G"
NEVAL=1e9

EXEC="sf_batched_nlo_sdaw_trip_T.py"
DIPOLE="median_bk.dat"
PARAM_FILE="test_batched_kinematic_points.txt"

XMAX=160.0

source db_info.sh
JOBID_FILE="$RESULT_DATABASE_PATH/$EXPERIMENT_NAME/submitted_jobs.txt"

if [ "$DRYRUN" -ne 1 ]; then : > "$JOBID_FILE"; fi

# --- Batching Arrays ---
CURRENT_CHUNK_ID=""
Q_BATCH=()
BETA_BATCH=()
XPOM_BATCH=()

submit_batch() {
    if [ ${#BETA_BATCH[@]} -eq 0 ]; then return; fi
    
    Q_STR="${Q_BATCH[0]}"
    BETAS_STR="${BETA_BATCH[*]}"
    XPOM_STR="${XPOM_BATCH[0]}"
    
    # Define job-specific paths
    JOB_NAME="${EXPERIMENT_NAME}"
    
    # This is the actual command string
    # We use --wrap to keep it simple, or a heredoc for complex multi-line logic
    if [ "$DRYRUN" -eq 1 ]; then
        echo "------------------------------------------------"
        echo "CHUNK ID: $CURRENT_CHUNK_ID (Size: ${#BETA_BATCH[@]})"
        echo "Q:    $Q_STR"
        echo "XPOM: $XPOM_STR"
        echo "BETAS: $BETAS_STR"
        echo "WOULD RUN: mkdir, module load, and python $EXEC"
        echo ""
    else
        # 1. Create a descriptive temporary filename using the kinematics
        SUBMIT_SCRIPT="${SLRM_OUTPUT_DIR}/${JOB_NAME}_Q_${Q_STR}_xpom_${XPOM_STR}_beta_${BETA_BATCH[0]}.sh"
        # 2. Write the heredoc to this file
        cat <<EOF > "$SUBMIT_SCRIPT"
#!/bin/bash
#SBATCH --partition=$PARTITION
#SBATCH --account=$ACCOUNT
#SBATCH --ntasks=$NTASKS
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --mem=$MEMORY
#SBATCH --gres=$GRES
#SBATCH --time=$TIME
#SBATCH --job-name="$JOB_NAME"
#SBATCH --output="${SLRM_OUTPUT_DIR}/${JOB_NAME}_%j.out"
#SBATCH --error="${SLRM_OUTPUT_DIR}/${JOB_NAME}_%j.err"

set -euo pipefail

# Setup Directories
save_dir="${RESULT_DATABASE_PATH}/${EXPERIMENT_NAME}/\${SLURM_JOB_ID}"
mkdir -p "\${save_dir}"

# Environment
module load julia/1.11.5 git
module load tensorflow
export JULIA_NUM_THREADS=\$SLURM_CPUS_PER_TASK

echo "Partition: $PARTITION"
echo "Number of threads: $CPUS_PER_TASK" #Useful to output this for running strong scaling experiments etc

starttime=$(date +%s%N)
echo "Job started at: $(date)"

# Execution
python $EXEC --Q $Q_STR --beta $BETAS_STR --x $XPOM_STR \
             --xmax $XMAX --neval $NEVAL --dipole_path $DIPOLE \
             --save_dir "\${save_dir}"

endtime=$(date +%s%N)
echo "Job finished at: $(date)"
elapsedtime=$((endtime - starttime))
printf "Job duration: %s.%s seconds\n" "${elapsedtime:0: -9}" "${elapsedtime: -9:3}"             
EOF

        # 3. Submit the script to sbatch
        jid=$(sbatch --parsable "$SUBMIT_SCRIPT")

        # 4. Append the real Slurm Job ID to the end of the filename 
        #    This preserves the kinematics in the name while linking it to your logs
        mv "$SUBMIT_SCRIPT" "${SLRM_OUTPUT_DIR}/${JOB_NAME}_Q_${Q_STR}_xpom_${XPOM_STR}_beta_${BETA_BATCH[0]}_${jid}.sh"

        echo "Chunk $CURRENT_CHUNK_ID submitted: $jid"
        echo "$jid" >> "$JOBID_FILE"
    fi
}

# --- Main Loop ---
while read -r Q_VAL BETA_VAL XPOM_VAL CHUNK_ID || [[ -n $Q_VAL ]]; do
    [[ -z "$Q_VAL" || "$Q_VAL" == "#"* ]] && continue

    if [[ -n "$CURRENT_CHUNK_ID" && "$CHUNK_ID" != "$CURRENT_CHUNK_ID" ]]; then
        submit_batch
        Q_BATCH=()
        BETA_BATCH=()
        XPOM_BATCH=()
    fi
    
    CURRENT_CHUNK_ID=$CHUNK_ID
    Q_BATCH+=("$Q_VAL")
    BETA_BATCH+=("$BETA_VAL")
    XPOM_BATCH+=("$XPOM_VAL")
done < "$PARAM_FILE"

submit_batch
