#!/bin/bash

DRYRUN=1  

# --- Standard Parameters ---
PARTITION="small"
#GRES="gpu:v100:1"
ACCOUNT="lappi"
NTASKS=1
CPUS_PER_TASK=16
TIME="2-12:00:00"
MEMORY="40G"
NEVAL=1e9

EXEC="sf_batched_nlo_sdaw_trip_T.py"
DIPOLE="median_bk.dat"
PARAM_FILE="betadep_kinematic_points_batched.txt"

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
    
    QS_STR="${Q_BATCH[*]}"
    BETAS_STR="${BETA_BATCH[*]}"
    XPOMS_STR="${XPOM_BATCH[*]}"
    
    # Define job-specific paths
    # Note: We use a placeholder since we don't have the JOB_ID yet, 
    # or we use SLURM's internal variables.
    JOB_NAME="${EXPERIMENT_NAME}_c${CURRENT_CHUNK_ID}"
    
    # This is the actual command string
    # We use --wrap to keep it simple, or a heredoc for complex multi-line logic
    if [ "$DRYRUN" -eq 1 ]; then
        echo "------------------------------------------------"
        echo "CHUNK ID: $CURRENT_CHUNK_ID (Size: ${#BETA_BATCH[@]})"
        echo "QS:    $QS_STR"
        echo "BETAS: $BETAS_STR"
        echo "XPOMS: $XPOMS_STR"
        echo "WOULD RUN: mkdir, module load, and python $EXEC"
        echo ""
    else
        # We submit a multi-line script directly to sbatch
        jid=$(sbatch --parsable <<EOF
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

# Execution
python $EXEC --Q $QS_STR --beta $BETAS_STR --x $XPOMS_STR \
             --xmax 40.0 --neval $NEVAL --dipole_path $DIPOLE \
             --save_dir "\${save_dir}" --json "result.json"
EOF
)
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