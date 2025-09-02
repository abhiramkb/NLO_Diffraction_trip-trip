# This file contains the location of the directory that stores all experiments and the experiment name.
# As of the current commit, each experiment has its own database. No global database across experiments has been implemented yet.
# Choose a shared folder for the database (and experiment results) that all members of the group can access.
# This info is read by the SLURM submission script and used to set the output file locations.
# It may also be read by any other helper scripts that are used to add entries to the database.
export RESULT_DATABASE_PATH="/projappl/lappi/abhiram/trip_database"
export EXPERIMENT_NAME="struct_fn"
# SLURM outputs from a given experiment are all kept in a common folder. One can use the job id to identify which result they correspond to.
export SLRM_OUTPUT_DIR="$RESULT_DATABASE_PATH/$EXPERIMENT_NAME/SLURM_OUTPUT"

if [ ! -d "${SLRM_OUTPUT_DIR}" ]; then
    mkdir -p "${SLRM_OUTPUT_DIR}"
fi
