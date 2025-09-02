using JSON
using MCIntegration
using SpecialFunctions
using ArgParse
using Base.Filesystem: basename

# We set z2 = 1 - z0 - z1 below
function GNLOT(Q, beta, z0, z1, x20, th20, x20b, th20b, x21, th21, x21b, th21b)
    Mx = sqrt(1.0/beta - 1.0)*Q;
    X012 = sqrt(z0*z1*(x21^2+x20^2 - 2*x21*x20*cos(th21 - th20)) + z0*(1.0 - z0 - z1)*x20^2 + z1*(1.0 - z0 - z1)*x21^2)
    
    X012b = sqrt(z0*z1*(x21b^2+x20b^2 - 2*x21b*x20b*cos(th21b - th20b)) + z0*(1.0 - z0 - z1)*x20b^2 + z1*(1.0 - z0 - z1)*x21b^2)
    
    Y012 = sqrt((z0*z1*(x21b^2 + x20b^2 - 2.0*x21b*x20b*cos(th21b - th20b) + x21^2 + x20^2 - 2.0*x21*x20*cos(th21 - th20) - 2.0*x21b*x21*cos(th21b - th21) + 2.0*x21b*x20*cos(th21b - th20) 
        + 2.0*x20b*x21*cos(th20b - th21) - 2.0*x20b*x20*cos(th20b - th20))
        + z0*(1.0 - z0 - z1)*(x20b^2 + x20^2 - 2.0*x20b*x20*cos(th20b - th20))
        + z1*(1.0 - z0 - z1)*(x21b^2 + x21^2 - 2.0*x21b*x21*cos(th21b - th21))))

    # Basic dot products:
    dot_x20_x20b = x20*x20b*cos(th20 - th20b);
    dot_x21_x21b = x21*x21b*cos(th21 - th21b);
    dot_x20_x21b = x20*x21b*cos(th20 - th21b);
    dot_x21_x20b = x21*x20b*cos(th21 - th20b);
    dot_x20_x21 = x20*x21*cos(th20 - th21);
    dot_x20b_x21b = x20b*x21b*cos(th20b - th21b);

    # Composite dot products first appearing in (29)
    dot_x20_x0p2c1 = dot_x20_x21 - (z0/(1.0-z1))*x20^2; # x0p2c1 = x21 - x20*z0/(1-z1)
    dot_x20b_x0p2c1b = dot_x20b_x21b - (z0/(1.0-z1))*x20b^2; # x0p2c1b = x21b - x20b*z0/(1-z1)
    dot_x20b_x0p2c1 = dot_x21_x20b - (z0/(1.0-z1))*dot_x20_x20b;
    dot_x20_x0p2c1b = dot_x20_x21b - (z0/(1.0-z1))*dot_x20_x20b;
    dot_x0p2c1_x0p2c1b = dot_x21_x21b - (z0/(1.0-z1))*(dot_x21_x20b + dot_x20_x21b) + (z0^2/(1.0 - z1)^2)*dot_x20_x20b;

    # Composite dot products first appearing in (30)
    dot_x21_x0c1p2 = (z1/(1.0 - z0))*x21^2 - dot_x20_x21;
    dot_x21_x0c1p2b = (z1/(1.0 - z0))*dot_x21_x21b - dot_x21_x20b;
    dot_x21b_x0c1p2 = (z1/(1.0 - z0))*dot_x21_x21b - dot_x20_x21b;
    dot_x21b_x0c1p2b = (z1/(1.0 - z0))*x21b^2 - dot_x20b_x21b;
    dot_x0c1p2_x0c1p2b = dot_x20_x20b - (z1/(1.0 - z0))*(dot_x21_x20b + dot_x20_x21b) + (z1/(1.0 - z0))^2*dot_x21_x21b;

    # Composite dot products first appearing in (33)
    dot_x0p2c1_x0c1p2b = (z1/(1.0 - z0))*dot_x21_x21b - dot_x21_x20b - (z0*z1/((1.0 - z0)*(1.0 - z1)))*dot_x20_x21b + (z0/(1.0 - z1))*dot_x20_x20b;
    dot_x0c1p2_x0p2c1b = (z1/(1.0 - z0))*dot_x21_x21b - dot_x20_x21b - (z0*z1/((1.0 - z0)*(1.0 - z1)))*dot_x21_x20b + (z0/(1.0 - z1))*dot_x20_x20b;
    dot_x20_x0c1p2b = (z1/(1.0 - z0))*dot_x20_x21b - dot_x20_x20b;
    dot_x21b_x0p2c1 = dot_x21_x21b - (z0/(1.0 - z1))*dot_x20_x21b;
    dot_x20b_x0c1p2 = (z1/(1.0 - z0))*dot_x21_x20b - dot_x20_x20b;
    dot_x21_x0p2c1b = dot_x21_x21b - (z0/(1.0 - z1))*dot_x21_x20b;
    
    # Coefficient functions from Eq. (29) - (33)
    term1b = (z0^2 + (1.0 - z1)^2)*(1.0 - 2*z1*(1.0 - z1))*dot_x0p2c1_x0p2c1b*dot_x20_x20b;
    term2b = -1.0*((1.0-z1)^2-z0^2)*(2.0*z1-1.0)*(dot_x20_x0p2c1*dot_x20b_x0p2c1b - dot_x20_x0p2c1b*dot_x20b_x0p2c1);
    Y_b_reg = (z1^2/(x20^2 * x20b^2))*(term1b + term2b);

    term1c = (z1^2 + (1.0 - z0)^2)*(1.0 - 2*z0*(1.0 - z0))*dot_x0c1p2_x0c1p2b*dot_x21_x21b;
    term2c = -1.0*((1.0-z0)^2-z1^2)*(2.0*z0-1.0)*(dot_x21_x0c1p2*dot_x21b_x0c1p2b - dot_x21_x0c1p2b*dot_x21b_x0c1p2);
    Y_c_reg = (z0^2 / (x21^2 * x21b^2)) * (term1c + term2c);

    term1d = (z0^2*z1^2*(1.0 - z0 - z1)^2)/(1.0 - z1)^2;
    term2d = -1.0*((z0^2*z1^3*(1.0 - z0 - z1))/(1.0 - z1)) * (dot_x20_x0p2c1/x20^2 + dot_x20b_x0p2c1b/x20b^2);
    term3d = ((z0^2*z1*(1.0 - z0 - z1)*(1.0 - z0)^2)/(1.0 - z1)) * (dot_x21_x0c1p2/x21^2 + dot_x21b_x0c1p2b/x21b^2);
    Y_d_inst = term1d + term2d + term3d;

    term1e = (z0^2*z1^2*(1.0 - z0 - z1)^2)/(1.0 - z0)^2;
    term2e = ((z0^3*z1^2*(1.0 - z0 - z1))/(1.0 - z0)) * (dot_x21_x0c1p2/x21^2 + dot_x21b_x0c1p2b/x21b^2);
    term3e = -1.0*((z0*z1^2*(1.0 - z0 - z1)*(1.0 - z1)^2)/(1.0 - z0)) * (dot_x20_x0p2c1/x20^2 + dot_x20b_x0p2c1b/x20b^2);
    Y_e_inst = term1e + term2e + term3e;

    

    term1bc_pref = -1.0*z0*z1*(z0*(1.0 - z1) + z1*(1.0 - z0))*(z0*(1.0 - z0) + z1*(1.0 - z1));
    term1bc_prods = dot_x0c1p2_x0p2c1b*dot_x21_x20b/(x21^2*x20b^2) + dot_x0p2c1_x0c1p2b*dot_x20_x21b/(x20^2*x21b^2);
    term2bc_pref = z0*z1*(1.0 - z0 - z1)*(z0 - z1)^2;
    term2bc_prod1 = (dot_x20_x0p2c1*dot_x21b_x0c1p2b - dot_x20_x0c1p2b*dot_x21b_x0p2c1)/(x20^2*x21b^2);
    term2bc_prod2 = (dot_x21_x0c1p2*dot_x20b_x0p2c1b - dot_x21_x0p2c1b*dot_x20b_x0c1p2)/(x21^2*x20b^2);
    Y_bc_interf = term1bc_pref*term1bc_prods + term2bc_pref*(term2bc_prod1 + term2bc_prod2);

    sum_Y_terms = Y_b_reg + Y_c_reg + Y_d_inst + Y_e_inst + Y_bc_interf;
    
    return z0*z1*besselk(1,Q*X012)*besselk(1,Q*X012b)*(1.0/(X012*X012b))*(1.0/Y012)*besselj(1,Mx*Y012)*sum_Y_terms;
end

function S(a,r)
    return exp(-a*r^2)
end
                    
function S012(Nc, CF, a, x20, th20, x21, th21)
    x10 = sqrt(x20^2 + x21^2 - 2.0*x20*x21*cos(th20 - th21))
    return S(a,x20)*S(a,x21)
end
                    
# We set th20 = 0 by using the global rotational symmetry

function integrand(Nc, CF, a, Q, beta, z0, z1, x20, x20b, th20b, x21, th21, x21b, th21b)
    th20 = 0.0
    measure = x20*x20b*x21*x21b
    return measure*GNLOT(Q, beta, z0, z1, x20, th20, x20b, th20b, x21, th21, x21b, th21b)*(1.0 - S012(Nc, CF, a, x20, th20, x21, th21))*(1.0 - S012(Nc, CF, a, x20b, th20b, x21b, th21b));
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
         # Positional arguments
        "Q"
            help = "Q - Photon virtuality"
            arg_type = Float64
            required = true
        "beta"
            help = "beta - DIS variable"
            arg_type = Float64
            required = true
        "xmax"
            help = "xmax (upper integration bound for |x_ij|)"
            arg_type = Float64
            required = true
        
        # Optional arguments
        "--neval"
            help = "Maximum number of points to be used for integration"
            arg_type = Float64
            default = 1e8
        "--a"
            help = "Dipole parameter (default = 0.1 - from GBW fit)"
            arg_type = Float64
            default = 0.1
        "--save_dir"
            help = "Saves result to specified folder"
            default = ""
	"--json"
	   help="Provide JSON filename to store input and output to JSON (located in save_dir)"
	   default = ""
    end

    args = parse_args(s)

    # --- provenance info ---
    args["script_file"] = basename(@__FILE__)

    try
        args["git_commit"] = readchomp(`git rev-parse HEAD`)
    catch
        args["git_commit"] = "N/A"
    end

    try
    	# Does the repo have any uncommitted changes? Ignore untracked files checking this.
	repo_dirty = !isempty(readchomp(`git status --porcelain`))
    	
    	# Does the file have any uncommitted changes?
    	file_dirty = !isempty(readchomp(`git status --porcelain -- $(abspath(@__FILE__))`))
    	args["git_is_dirty"] = repo_dirty
    	args["script_is_dirty"] = file_dirty
    catch
    	args["git_is_dirty"] = "N/A"
    	args["script_is_dirty"] = "N/A"
    end

    return args
end

function main()

    Nc = 3.0
    CF = (Nc^2 - 1.0)/(2.0*Nc)
    a = 0.1 # Parameter for dipole

    Q = sqrt(3.0)
    beta = 0.5

    xmax = 10.0

    parsed_args = parse_commandline()
    
    # Input parameters
    param_keys = ["Q", "beta", "xmax", "neval", "a"]
    params = Dict(k => parsed_args[k] for k in param_keys)

    # Metadata (where output files are stored etc)
    meta_keys = ["save_dir", "json"]
    meta = Dict(k => parsed_args[k] for k in meta_keys)

    # Provenance info
    provenance_keys = ["script_file", "git_commit", "git_is_dirty", "script_is_dirty"]
    provenance = Dict(k => parsed_args[k] for k in provenance_keys)


    Q = parsed_args["Q"]
    beta = parsed_args["beta"]
    xmax = parsed_args["xmax"]
    a = parsed_args["a"]
    n_points = Int64(parsed_args["neval"]) # parse as Int
    save_dir = parsed_args["save_dir"]
    json = parsed_args["json"]

    println("Parsed args: Q = ",Q)
    println("Parsed args: beta = ",beta)
    println("Parsed args: xmax = ",xmax)
    println("Parsed args: n_points = ",n_points)
    println("Parsed args: a = ",a)
    println("Description: Monte Carlo integral for trip T contribution (large Nc)");
    
    
    variables = Continuous([(0,1),(0,1),(0,xmax),(0,xmax),(0,2.0*pi),(0,xmax),(0,2.0*pi),(0,xmax),(0,2.0*pi)])

    function f((z0, t, x20, x20b, th20b, x21, th21, x21b, th21b),c)
        # Note that z1 goes from 0 to z0 so we do a change of variables to get an integration over [0,1].
        zmin = 0.0
        zmax = (1.0 - z0[1])
        z1 = zmin + (zmax - zmin)*t[1]
        jac = (zmax - zmin)
        return jac*integrand(Nc, CF, a, Q, beta, z0[1], z1, x20[1], x20b[1], th20b[1], x21[1], th21[1], x21b[1], th21b[1])
    end

    res = integrate(f; var = variables, neval=n_points, parallel = :thread)

    result_filename = "result_mcint_neval_"*string(parsed_args["neval"])*"_xmax_"*string(xmax)*"_a_"*string(a)*"_Q_"*string(Q)*"_beta_"*string(beta)*".txt";
    result_path = save_dir*"/"*result_filename;
    json_file_path = save_dir*"/"*json;

    println("Result: ", res)
    println("Report: ")
    println(report(res))

    if save_dir != ""
        mkpath(save_dir)
        open(result_path, "w") do file
            write(file, "($(res[1][1]), $(res[1][2]), $(res[1][3]))")
        end
    end


   if json != ""
	payload = Dict(
			"parameters" => params,
			"metrics" => Dict("result"=>res[1][1], "error"=>res[1][2], "chi2/dof"=>res[1][3]),
			"provenance" => provenance,
			"meta" => meta
		)
	path = joinpath(save_dir == "" ? "." : save_dir, json)
    	open(path, "w") do io
        	JSON.print(io, payload) 
    	end
    	println("Saved JSON results to $path")
   end
        
end

main()