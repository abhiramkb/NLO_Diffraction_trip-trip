#!/bin/bash
#set -euo pipefail

# Load db_info.sh so EXPERIMENT_NAME etc. are available
source db_info.sh

JOBID_FILE="$RESULT_DATABASE_PATH/$EXPERIMENT_NAME/submitted_jobs.txt"

if [ ! -f "$JOBID_FILE" ]; then
    echo "ERROR: $JOBID_FILE not found. Did you run the scheduler script?"
    exit 1
fi

while true; do
    still_running=0
    for jid in $(cat "$JOBID_FILE"); do
        if squeue -j "$jid" 2>/dev/null | grep -q "$jid"; then
            still_running=1
            break
        fi
    done

    if [ $still_running -eq 0 ]; then
        echo "All jobs finished. Running collector..."
        julia collect_results.jl
        break
    else
        echo "Jobs still running... checking again in 2 minutes."
        sleep 120
    fi
done

