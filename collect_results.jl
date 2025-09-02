using SQLite, JSON, Glob

# --- Sanity checks for environment variables ---
if !haskey(ENV, "RESULT_DATABASE_PATH")
    error("Environment variable RESULT_DATABASE_PATH is not set.")
end
if !haskey(ENV, "EXPERIMENT_NAME")
    error("Environment variable EXPERIMENT_NAME is not set.")
end

# Pick up paths from environment
base_path = ENV["RESULT_DATABASE_PATH"]
experiment = ENV["EXPERIMENT_NAME"]
database_path = joinpath(base_path, experiment, "results.db")

println("Accessing database at ",database_path)


db = SQLite.DB(database_path)

# Create the table if it does not exist
SQLite.execute(db, """
    CREATE TABLE IF NOT EXISTS runs (
        job_id           TEXT PRIMARY KEY,
        Q                REAL,
        beta             REAL,
        xmax             REAL,
        neval            INTEGER,
        a                REAL,
        result           REAL,
        error            REAL,
        chi2_dof         REAL,
        script_file      TEXT,
        git_commit       TEXT,
        git_is_dirty     BOOLEAN,
        script_is_dirty  BOOLEAN,
        save_dir         TEXT,
        json_file        TEXT,
        duration         REAL,
	nthreads	 INTEGER,
	integrator_time	 REAL,
	partition	 TEXT
    )
""")


pattern = "*/result.json"
experiment_dir = joinpath(base_path, experiment)
slurm_output_dir = joinpath(base_path, experiment, "SLURM_OUTPUT")

if !isdir(slurm_output_dir)
    @warn "SLURM output directory does not exist: $slurm_output_dir"
end

for filepath in Glob.glob(pattern, experiment_dir)
    println("Opening ",filepath)
    job_id = basename(dirname(filepath))  # folder name = SLURM_JOB_ID

    data = open(filepath) do io
        JSON.parse(io)
    end

    params = data["parameters"]
    metrics = data["metrics"]
    provenance = data["provenance"]
    meta = data["meta"]

    # Default: duration not available
    duration = nothing

    # Default: nthreads not available
    nthreads = nothing

    # Default: integrator_time not available
    integrator_time = nothing

    # Default: partition not available
    partition = nothing

    # Construct SLURM output file path
    slurm_out_file = joinpath(slurm_output_dir, string(experiment, "_", job_id, ".out"))

    if isfile(slurm_out_file)
        for line in eachline(slurm_out_file)
            if occursin("Job duration:", line)
                m = match(r"Job duration:\s*([0-9.]+)\s*seconds", line)
                if m !== nothing
                    duration = parse(Float64, m.captures[1])
                end
                break
            end
        end
    end

    if isfile(slurm_out_file)
        for line in eachline(	slurm_out_file)
            if occursin("Number of threads:", line)
                m = match(r"Number of threads:\s*([0-9]+)", line)
                if m !== nothing
                    nthreads = parse(Int64, m.captures[1])
                end
                break
            end
        end
    end

    if isfile(slurm_out_file)
        for line in eachline(slurm_out_file)
            if occursin("Total iterations * blocks", line)
		m = match(r"Time:\s*(\d+):(\d{2}):(\d{2})", line)
		if m !== nothing
		    hours = m.captures[1] === nothing ? 0 : parse(Int, m.captures[1])
                    minutes = parse(Int, m.captures[2])
                    seconds = parse(Int, m.captures[3])
                    integrator_time = hours*3600 + minutes*60 + seconds
                end
                break
            end
        end
    end

    if isfile(slurm_out_file)
        for line in eachline(slurm_out_file)
            if occursin("Partition:", line)
                m = match(r"Partition:\s*(\S+)", line)
                if m !== nothing
                	partition = m.captures[1]
            	end
		break
            end
        end
    end



    SQLite.execute(db, """
        INSERT OR IGNORE INTO runs
        (job_id, Q, beta, xmax, neval, a, result, error, chi2_dof,
         script_file, git_commit, git_is_dirty, script_is_dirty, save_dir, json_file, duration, nthreads, integrator_time, partition)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        job_id,
        params["Q"], params["beta"], params["xmax"], params["neval"], params["a"],
        metrics["result"], metrics["error"], metrics["chi2/dof"],
        provenance["script_file"], provenance["git_commit"], provenance["git_is_dirty"], provenance["script_is_dirty"],
        meta["save_dir"], meta["json"], duration, nthreads, integrator_time, partition
    ))
end

SQLite.close(db)

