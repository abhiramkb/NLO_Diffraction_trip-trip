# NLO Diffraction trip-trip 
Code for the NLO trip contribution to the diffractive structure function derived in [Phys. Rev. D 106, 094014](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.106.094014) ([arXiv:2206.13161](https://arxiv.org/abs/2206.13161)).

## Background  
This repository contains the implementation of the “trip-trip” contribution to the diffractive structure function. This corresponds to all partons in the tripole interacting with the shockwave in both the ampitude and the conjugate.  

## Codes  
- Implementation of the photon longitudinal (L) and transverse (T) contributions: `photon_L.jl`, `photon_T.jl`.
  - You can execute the individual Julia programs from the command line as follows:
  ```bash
  julia photon_L.jl
  julia photon_T.jl
- Large-Nc versions of the above codes: `photon_L_large_Nc.jl`, `photon_T_large_Nc.jl`.  In the corresponding publication (link not yet available) the large-Nc versions were used to generate the numbers.
- Munier-Shoshi limit ($M_X\to\infty/\beta\to0$ ): `photon_T_MS_limit_WL_structure.jl`. This was used to verify that the trip kinematics approaches the MS limit when the appropriately modified Wilson line is used. To compare with MS limit, run the code at as small a value of $\beta$ for which the code is numerically stable and doesn't take too long to run. 
- Helper codes for scheduling in a HPC system (modify as appropriate to your system): `db_info.sh`, `submit_job.sh`, `schedule_job_submission_script.sh`, `wait_and_collect.sh`, `collect_results.jl`.
  - The helper codes involve a rudimentary experiment tracking system that accumulates the results in a SQLite database file as you schedule the codes with various input parameters. I will try to write down a detailed explanation of this when I have the time.
- Environment setup: `load_modules.sh`.

### Prerequisites
- **Julia** (code runs well using version 1.11. MCIntegration package seems to have problems with newer versions of Julia)
- **MCIntegration** Julia package
- **JSON** Julia package
- **SpecialFunctions** Julia package
- **ArgParse** Julia package
- **SLURM** scheduler in case you want to use the helper codes (scheduling/rudimentary experiment tracking system)
