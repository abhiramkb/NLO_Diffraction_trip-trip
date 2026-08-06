# Trip contribution to the diffractive structure function at NLO. Code includes all factors except transverse profile integral.
#
# Coupling fixed at smallest dipole width.
#
# alpha_s fixed to smallest dipole width in amplitude only.
#
# The "sdaw" in the filename stands for the alpha_s prescription: Smallest Dipole, Width from Amplitude.
import time
import os
import subprocess
import json
import math
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from vegasflow import VegasFlow
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.experimental import numpy as tnp #Use tnp instead of numpy
import argparse

@tf.function(jit_compile=True)
def alphas(r, Csq):
    LambdaQCD = 0.241
    Nc = 3.0
    Nf = 3.0
    beta = (11.0*Nc - 2.0*Nf)/3.0
    LambdaQCDsq = LambdaQCD**2

    res = 12*tnp.pi/((33.0 - 2.0*Nf)*tnp.log(4*Csq/(LambdaQCDsq*r*r)))
    alphas_cutoff = 1.0
    
    return tf.minimum(res,alphas_cutoff)
    
def ReadBKDipole(path_to_file):
    with open(path_to_file) as f:
        content = f.read().split("###")

    content = content[1:]
    content = [i.split() for i in content]

    NrY_data = []
    pars = []

    for i in content:
        x = list(map(float, i))
        if len(x) == 1:
            pars.append(x)
        else:
            NrY_data.append(x)

    NrY_data = np.array(NrY_data)

    Y_values = NrY_data[:, 0]
    N_values = NrY_data[:, 1:]

    pars = np.array(pars).flatten()
    minr, mult, n = pars[0], pars[1], int(pars[2])

    r_values = np.array([minr * mult**i for i in range(n)])
    #logr_values = np.array([np.log(r) for r in r_values])

    #print(r_values)

    # N_values should have shape (len(Y), len(r))
    interpolator = RegularGridInterpolator(
        (Y_values, r_values),
        N_values
    )

    return interpolator

def GetGridParameters(path_to_file):
    with open(path_to_file) as f:
        content = f.read().split("###")

    content = content[1:]
    content = [i.split() for i in content]

    NrY_data = []
    pars = []

    for i in content:
        x = list(map(float, i))
        if len(x) == 1:
            pars.append(x)
        else:
            NrY_data.append(x)

    NrY_data = np.array(NrY_data)

    Y_values = NrY_data[:, 0]

    ymin, ymax, yinc = min(Y_values), max(Y_values), Y_values[2]-Y_values[1]
    
    pars = np.array(pars).flatten()

    minr, mult, n = pars[0], pars[1], int(pars[2])
    return minr, mult, n, ymin, ymax, yinc


def GetYRgrid(path_to_file):
    # Returns a 2D numpy grid of dipole values with increasing r(Y) on the  x(y) axis.
    with open(path_to_file) as f:
        content = f.read().split("###")

    content = content[1:]
    content = [i.split() for i in content]

    NrY_data = []
    pars = []

    for i in content:
        x = list(map(float, i))
        if len(x) == 1:
            pars.append(x)
        else:
            NrY_data.append(x)

    NrY_data = np.array(NrY_data)
    return NrY_data[:,1:]

@tf.function(jit_compile=True)
def calculate_GNLOT_terms(z0, z1, z2, x20, th20, x20b, th20b, x21, th21, x21b, th21b):
    """
    Computes sum_Y_terms, X012, X012b and Y012 with XLA JIT compilation enabled.
    Inputs are expected to have shape (N, 1).
    """
    cos_th20_m_th20b = tf.cos(th20 - th20b)
    cos_th21_m_th21b = tf.cos(th21 - th21b)
    cos_th20_m_th21b = tf.cos(th20 - th21b)
    cos_th21_m_th20b = tf.cos(th21 - th20b)
    cos_th20_m_th21 = tf.cos(th20 - th21)
    cos_th20b_m_th21b = tf.cos(th20b - th21b)
    
    # Basic dot products
    dot_x20_x20b = x20 * x20b * cos_th20_m_th20b
    dot_x21_x21b = x21 * x21b * cos_th21_m_th21b
    dot_x20_x21b = x20 * x21b * cos_th20_m_th21b
    dot_x21_x20b = x21 * x20b * cos_th21_m_th20b
    dot_x20_x21 = x20 * x21 * cos_th20_m_th21
    dot_x20b_x21b = x20b * x21b * cos_th20b_m_th21b

    inv_1m_z1 = 1.0 / (1.0 - z1)
    inv_1m_z0 = 1.0 / (1.0 - z0)
    z0_over_1m_z1 = z0 * inv_1m_z1
    z1_over_1m_z0 = z1 * inv_1m_z0

    one_m_z1_sq = (1.0 - z1)**2
    one_m_z0_sq = (1.0 - z0)**2

    z0_sq = z0**2
    z1_sq = z1**2

    z0_x_z1 = z0 * z1 
    z0_x_z2 = z0 * z2
    z1_x_z2 = z1 * z2

    x20_sq = x20**2
    x21_sq = x21**2
    x20b_sq = x20b**2
    x21b_sq = x21b**2

    X012 = tf.sqrt(z0_x_z1 * (x21_sq + x20_sq - 2*dot_x20_x21) + z0_x_z2 * x20_sq + z1_x_z2 * x21_sq)
    X012b = tf.sqrt(z0_x_z1 * (x21b_sq + x20b_sq - 2*dot_x20b_x21b) + z0_x_z2 * x20b_sq + z1_x_z2 * x21b_sq)

    Y012_part1 = (z0_x_z1 * (x21b_sq + x20b_sq - 2*dot_x20b_x21b + x21_sq + x20_sq - 2*dot_x20_x21 - 2*dot_x21_x21b + 2*dot_x20_x21b + 2*dot_x21_x20b - 2*dot_x20_x20b))
    Y012_part2 = (z0_x_z2 * (x20b_sq + x20_sq - 2*dot_x20_x20b))
    Y012_part3 = (z1_x_z2 * (x21b_sq + x21_sq - 2*dot_x21_x21b))
    Y012 = tf.sqrt(Y012_part1 + Y012_part2 + Y012_part3) # Shape: (N, 1)

    
    
    # Composite dot products from (29)
    dot_x20_x0p2c1 = dot_x20_x21 - z0_over_1m_z1 * x20_sq
    dot_x20b_x0p2c1b = dot_x20b_x21b - z0_over_1m_z1 * x20b_sq
    dot_x20b_x0p2c1 = dot_x21_x20b - z0_over_1m_z1 * dot_x20_x20b
    dot_x20_x0p2c1b = dot_x20_x21b - z0_over_1m_z1 * dot_x20_x20b
    dot_x0p2c1_x0p2c1b = (
        dot_x21_x21b -
        z0_over_1m_z1 * (dot_x21_x20b + dot_x20_x21b) +
        (z0_sq/one_m_z1_sq) * dot_x20_x20b
    )

    # Composite dot products from (30)
    dot_x21_x0c1p2 = z1_over_1m_z0 * x21_sq - dot_x20_x21
    dot_x21_x0c1p2b = z1_over_1m_z0 * dot_x21_x21b - dot_x21_x20b
    dot_x21b_x0c1p2 = z1_over_1m_z0 * dot_x21_x21b - dot_x20_x21b
    dot_x21b_x0c1p2b = z1_over_1m_z0 * x21b_sq - dot_x20b_x21b
    dot_x0c1p2_x0c1p2b = (
        dot_x20_x20b -
        z1_over_1m_z0 * (dot_x21_x20b + dot_x20_x21b) +
        z1_over_1m_z0**2 * dot_x21_x21b
    )

    # Composite dot products from (33)
    dot_x0p2c1_x0c1p2b = (
        z1_over_1m_z0 * dot_x21_x21b - dot_x21_x20b -
        (z0_x_z1/((1.0 - z0)*(1.0 - z1))) * dot_x20_x21b +
        z0_over_1m_z1 * dot_x20_x20b
    )
    dot_x0c1p2_x0p2c1b = (
        z1_over_1m_z0 * dot_x21_x21b - dot_x20_x21b -
        (z0_x_z1/((1.0 - z0)*(1.0 - z1))) * dot_x21_x20b +
        z0_over_1m_z1 * dot_x20_x20b
    )
    dot_x20_x0c1p2b = z1_over_1m_z0 * dot_x20_x21b - dot_x20_x20b
    dot_x21b_x0p2c1 = dot_x21_x21b - z0_over_1m_z1 * dot_x20_x21b
    dot_x20b_x0c1p2 = z1_over_1m_z0 * dot_x21_x20b - dot_x20_x20b
    dot_x21_x0p2c1b = dot_x21_x21b - z0_over_1m_z1 * dot_x21_x20b

    # Coefficient functions from Eqs. (29)–(33)
    term1b = (
        (z0_sq + one_m_z1_sq) * (1.0 - 2*z1*(1.0 - z1)) *
        dot_x0p2c1_x0p2c1b * dot_x20_x20b
    )
    term2b = -(
        (one_m_z1_sq - z0_sq) * (2.0*z1 - 1.0) *
        (dot_x20_x0p2c1 * dot_x20b_x0p2c1b - dot_x20_x0p2c1b * dot_x20b_x0p2c1)
    )
    Y_b_reg = (z1_sq / (x20_sq * x20b_sq)) * (term1b + term2b)

    term1c = (
        (z1_sq + one_m_z0_sq) * (1.0 - 2*z0*(1.0 - z0)) *
        dot_x0c1p2_x0c1p2b * dot_x21_x21b
    )
    term2c = -(
        (one_m_z0_sq - z1_sq) * (2.0*z0 - 1.0) *
        (dot_x21_x0c1p2 * dot_x21b_x0c1p2b - dot_x21_x0c1p2b * dot_x21b_x0c1p2)
    )
    Y_c_reg = (z0_sq / (x21_sq * x21b_sq)) * (term1c + term2c)

    term1d = (z0_sq * z1_sq * z2**2) / one_m_z1_sq
    term2d = -(
        (z0_sq * z1**3 * z2) / (1.0 - z1)
    ) * (dot_x20_x0p2c1 / x20_sq + dot_x20b_x0p2c1b / x20b_sq)
    term3d = (
        (z0_sq * z1_x_z2 * one_m_z0_sq) / (1.0 - z1)
    ) * (dot_x21_x0c1p2 / x21_sq + dot_x21b_x0c1p2b / x21b_sq)
    Y_d_inst = term1d + term2d + term3d

    term1e = (z0_sq * z1_sq * z2**2) / one_m_z0_sq
    term2e = (
        (z0**3 * z1_sq * z2) / (1.0 - z0)
    ) * (dot_x21_x0c1p2 / x21_sq + dot_x21b_x0c1p2b / x21b_sq)
    term3e = -(
        (z0 * z1_sq * z2 * one_m_z1_sq) / (1.0 - z0)
    ) * (dot_x20_x0p2c1 / x20_sq + dot_x20b_x0p2c1b / x20b_sq)
    Y_e_inst = term1e + term2e + term3e

    term1bc_pref = -z0_x_z1 * (z0*(1.0 - z1) + z1*(1.0 - z0)) * (z0*(1.0 - z0) + z1*(1.0 - z1))
    term1bc_prods = (
        dot_x0c1p2_x0p2c1b * dot_x21_x20b / (x21_sq * x20b_sq) +
        dot_x0p2c1_x0c1p2b * dot_x20_x21b / (x20_sq * x21b_sq)
    )
    term2bc_pref = z0_x_z1 * z2 * (z0 - z1)**2
    term2bc_prod1 = (
        (dot_x20_x0p2c1 * dot_x21b_x0c1p2b - dot_x20_x0c1p2b * dot_x21b_x0p2c1) /
        (x20_sq * x21b_sq)
    )
    term2bc_prod2 = (
        (dot_x21_x0c1p2 * dot_x20b_x0p2c1b - dot_x21_x0p2c1b * dot_x20b_x0c1p2) /
        (x21_sq * x20b_sq)
    )
    Y_bc_interf = term1bc_pref * term1bc_prods + term2bc_pref * (term2bc_prod1 + term2bc_prod2)

    sum_Y_terms = Y_b_reg + Y_c_reg + Y_d_inst + Y_e_inst + Y_bc_interf
    kin_factor = z0 * z1 * sum_Y_terms

    return X012, X012b, Y012, kin_factor

# --- VECTORIZED GNLOT ---
def GNLOT(Q, Mx, z0, z1, z2, x20, th20, x20b, th20b, x21, th21, x21b, th21b):

    # All non-Bessel coordinate and kinematic calculations in ONE fused XLA block
    X012, X012b, Y012, kin_factor = calculate_GNLOT_terms(z0, z1, z2, x20, th20, x20b, th20b, x21, th21, x21b, th21b)

    # Bessel calculation: Mx (1, M) * Y012 (N, 1) -> (N, M)
    # bessel_k1(Q * X012) -> (N, 1)
    res_bessel = (
        tf.math.special.bessel_k1(Q * X012) * tf.math.special.bessel_k1(Q * X012b) * (1.0 / (X012 * X012b)) * 
    (1.0 / Y012) * tf.math.special.bessel_j1(Mx * Y012)
    )
    
    return res_bessel * kin_factor # Shape: (N, M)


@tf.function(jit_compile=True)
def S012(tfgrid, x_ref_min, x_ref_max, Y, x10, x20, th20, x21, th21):
    # Y is (N, M), coordinates are (N, 1)
    Nc = 3.0
    CF = 4.0/3.0
    #x10 = tf.sqrt(x20**2 + x21**2 - 2.0 * x20 * x21 * tf.cos(th20 - th21))

    shape_N_M = tf.shape(Y)

  # Broadcast coordinates to (N, M)
    log_x20 = tf.broadcast_to(tf.math.log(x20), shape_N_M)
    log_x21 = tf.broadcast_to(tf.math.log(x21), shape_N_M)
    log_x10 = tf.broadcast_to(tf.math.log(x10), shape_N_M)

    # Stack into (3, N, M, 2)
    s0 = tf.stack([Y, log_x20], axis=-1)
    s1 = tf.stack([Y, log_x21], axis=-1)
    s2 = tf.stack([Y, log_x10], axis=-1)
    coords = tf.stack([s0, s1, s2], axis=0) # Shape: (3, N, M, 2)
    # --- THE FIX: FLATTEN ---
    # Collapse (3, N, M) into a single batch dimension
    flat_coords = tf.reshape(coords, [-1, 2]) # Shape: (TotalPoints, 2)

    # Interpolate using the flattened coordinates
    # Because flat_coords is rank-2, tfp won't try to broadcast the grid
    flat_Nvals = tfp.math.batch_interp_regular_nd_grid(
        flat_coords, x_ref_min, x_ref_max, tfgrid,  axis=-2,
        fill_value='constant_extension'
    )

    # Reshape back to (3, N, M)
    Nvals = tf.reshape(flat_Nvals, [3, shape_N_M[0], shape_N_M[1]])

    Svals = 1.0 - Nvals
    return (Nc / (2.0 * CF)) * (Svals[0] * Svals[1] - (1.0 / Nc**2) * Svals[2])

@tf.function(jit_compile=True)
def both_S012s(tfgrid, x_ref_min, x_ref_max, Y, x10, x20, th20, x21, th21, x10b, x20b, th20b, x21b, th21b):
    # Y is (N, M), coordinates are (N, 1)
    Nc = tf.constant(3.0, dtype = tf.float64)
    CF = tf.constant(4.0/3.0, dtype = tf.float64)
    #x10 = tf.sqrt(x20**2 + x21**2 - 2.0 * x20 * x21 * tf.cos(th20 - th21))

    shape_N_M = tf.shape(Y)

  # Broadcast coordinates to (N, M)
    log_x20 = tf.broadcast_to(tf.math.log(x20), shape_N_M)
    log_x21 = tf.broadcast_to(tf.math.log(x21), shape_N_M)
    log_x10 = tf.broadcast_to(tf.math.log(x10), shape_N_M)
    log_x20b = tf.broadcast_to(tf.math.log(x20b), shape_N_M)
    log_x21b = tf.broadcast_to(tf.math.log(x21b), shape_N_M)
    log_x10b = tf.broadcast_to(tf.math.log(x10b), shape_N_M)

    # Stack into (3, N, M, 2)
    s0 = tf.stack([Y, log_x20], axis=-1)
    s1 = tf.stack([Y, log_x21], axis=-1)
    s2 = tf.stack([Y, log_x10], axis=-1)
    s0b = tf.stack([Y, log_x20b], axis=-1)
    s1b = tf.stack([Y, log_x21b], axis=-1)
    s2b = tf.stack([Y, log_x10b], axis=-1)
    coords = tf.stack([s0, s1, s2, s0b, s1b, s2b], axis=0) # Shape: (6, N, M, 2)
    # --- THE FIX: FLATTEN ---
    # Collapse (6, N, M) into a single batch dimension
    flat_coords = tf.reshape(coords, [-1, 2]) # Shape: (TotalPoints, 2)

    # Interpolate using the flattened coordinates
    # Because flat_coords is rank-2, tfp won't try to broadcast the grid
    flat_Nvals = tfp.math.batch_interp_regular_nd_grid(
        flat_coords, x_ref_min, x_ref_max, tfgrid,  axis=-2,
        fill_value='constant_extension'
    )

    # Reshape back to (6, N, M)
    Nvals = tf.reshape(flat_Nvals, [6, shape_N_M[0], shape_N_M[1]])

    Svals = 1.0 - Nvals
    S012 = (Nc / (2.0 * CF)) * (Svals[0] * Svals[1] - (1.0 / Nc**2) * Svals[2])
    S012b = (Nc / (2.0 * CF)) * (Svals[3] * Svals[4] - (1.0 / Nc**2) * Svals[5])

    return S012, S012b

@tf.function(jit_compile=True)
def compute_integrand_preamble(xx, Csq, beta_vec, Q, xpom, Q0sq):
    """
    Computes coordinate transformations, kinematics, Yqqg, and term1 in XLA.
    Outputs:
        term1: (N, 1)
        z1, z2: (N, 1)
        Yqqg: (N, M)
    """
    # Unstack coordinates (N, 1)
    unstacked = tf.unstack(xx, axis=-1)
    z0, t, x20, x20b, th20b, x21, th21, x21b, th21b = [v[:, tf.newaxis] for v in unstacked]

    # Transverse distance calculations
    x01 = tf.sqrt(x20**2 + x21**2 - 2.0 * x20 * x21 * tf.cos(th21))
    x01b = tf.sqrt(x20b**2 + x21b**2 - 2.0 * x20b * x21b * tf.cos(th20b - th21b))
    measure = x20 * x20b * x21 * x21b
    
    # Kinematic substitutions
    zmin = 0.0
    zmax = (1.0 - z0)
    z1 = zmin + (zmax - zmin) * t
    jac = (zmax - zmin)
    z2 = 1.0 - z0 - z1

    Qsq = Q**2
    
    # Wsq: (1, M), Yqqg: (N, M) via broadcasting with z2 (N, 1)
    Wsq = Qsq * (1.0 / (beta_vec * xpom) - 1.0)
    Yqqg = tf.math.log(z2 * (Wsq + Qsq) / Q0sq)

    term1 = jac * measure * alphas(tf.math.minimum(x01,tf.math.minimum(x20,x21)), Csq)

    return term1, z0, z1, z2, x01, x01b, x20, x20b, th20b, x21, th21, x21b, th21b, Yqqg

@tf.function
def integrand(xx, tfgrid, Csq, x_ref_min, x_ref_max, Mx, Q=2.0, beta=0.1, xpom=0.01):
    beta_vec = tf.reshape(beta, (1, -1)) # Shape: (1, M)
    Q0sq = 1.0
    th20 = 0.0

    # 1. Accelerated Preamble (XLA JIT)
    (term1, z0, z1, z2, x01, x01b, x20, x20b, 
     th20b, x21, th21, x21b, th21b, Yqqg) = compute_integrand_preamble(
        xx, Csq, beta_vec, Q, xpom, Q0sq
    )

    (s012, s012b) = both_S012s(tfgrid, x_ref_min, x_ref_max, Yqqg, x01, x20, th20, x21, th21, x01b, x20b, th20b, x21b, th21b)

    #tf.print(s012b_ref - s012b_new)

    term2 = GNLOT(Q, Mx, z0, z1, z2, x20, th20, x20b, th20b, x21, th21, x21b, th21b)
    term3 = (1.0 - s012)
    term4 = (1.0 - s012b)

    return term1 * (term2 * (term3 * term4)) # Result: (N, M)

# --- VERIFICATION BLOCK ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trip-trip (T) contribution from dipole grid.")
    # Replace the Q and beta arguments in your current block:
    parser.add_argument("--Q", type=float, default=3.1622, help="Q - Photon virtuality")
    parser.add_argument("--beta", type=float, nargs='+', default=[0.5], help="beta - DIS variable - supply one or more values")
    parser.add_argument("--x", type=float, default=0.01, help="xpom - Pomeron-x")
    parser.add_argument("--xmax", type=float, default=40.0, help="xmax (upper integration bound for |x_ij|)")
    parser.add_argument("--dipole_path", type=str, required=True, help="Path to the BK table")
    parser.add_argument("--Csq", type=float, required=True, help="Csq - alpha_s parameter associated with dipole grid")
    parser.add_argument("--neval", type=float, default=1e6, help="Number of integration points")
    parser.add_argument("--input_grid_path", type=str, default="", help="Path to the pre-trained VEGAS grid (if available)")
    parser.add_argument("--save_dir", type=str, default="", help="Saves result to specified folder")
    args = vars(parser.parse_args())

    # --- Provenance info ---
    args["script_file"] = os.path.basename(__file__)

    # Helper function to run shell commands safely
    def run_cmd(cmd):
        try:
            return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode().strip()
        except subprocess.CalledProcessError:
            return None

    # Get Git Commit
    commit = run_cmd("git rev-parse HEAD")
    args["git_commit"] = commit if commit else "N/A"

    if commit:
        # Does the repo have any uncommitted changes?
        repo_status = run_cmd("git status --porcelain")
        # Does the specific file have any uncommitted changes?
        file_path = os.path.abspath(__file__)
        file_status = run_cmd(f"git status --porcelain -- {file_path}")

        args["git_is_dirty"] = bool(repo_status)
        args["script_is_dirty"] = bool(file_status)
    else:
        args["git_is_dirty"] = "N/A"
        args["script_is_dirty"] = "N/A"

    Qval = args["Q"]
    betavals = np.array(args["beta"]) #Array of beta values for simultaneous processing
    xpomval = args["x"]
    xmaxval = args["xmax"]
    Csqval = args["Csq"]

    print(f"Processing beta value(s): {betavals}")
    
    Q=tf.constant(args["Q"], dtype=tf.float64)
    beta_list=tf.constant(args["beta"], dtype=tf.float64)
    xpom=tf.constant(args["x"], dtype=tf.float64)
    xmax=tf.constant(args["xmax"], dtype=tf.float64)
    Csq=tf.constant(args["Csq"], dtype=tf.float64)
    n_events = int(args["neval"])

    beta_vec = tf.reshape(beta_list, (1, -1))       # (1, M)
    Mx_const = tf.sqrt(1.0 / beta_vec - 1.0) * Q     # precomputed once

    raw_dipole_path = args["dipole_path"]
    dipole_path = os.path.abspath(raw_dipole_path) if raw_dipole_path !="" else ""
    args["dipole_path"] = dipole_path #Updating dict with absolute path
    raw_save_dir = args["save_dir"]
    save_dir = os.path.abspath(raw_save_dir) if raw_save_dir !="" else ""
    args["save_dir"] = save_dir #Updating dict with absolute path
    raw_input_grid_path = args["input_grid_path"]
    input_grid_path = os.path.abspath(raw_input_grid_path) if raw_input_grid_path !="" else ""
    args["input_grid_path"] = input_grid_path

    # Assemble prefactor
    Nc = 3.0
    CF = 4.0/3.0
    sum_ef_squared = 2.0/3.0 # 4/9 + 1/9 + 1/9 = 2/3
    # Note that prefactor does not contain transverse profile (squared) integral
    prefactorsT = Nc*CF*Qval**7 * np.sqrt(1.0/betavals - 1.0)/((2*np.pi)**5 * betavals*2*np.pi**2) * sum_ef_squared
    
    
    
    th20=tf.constant(0.0, dtype=tf.float64)

    interp = ReadBKDipole(dipole_path)
    #Getting grid parameters:
    rmin,mult,n,ymin,ymax,yinc=GetGridParameters(dipole_path)
    rmax = rmin*mult**(n-1)
    logrmin = np.log(rmin)
    logrmax = np.log(rmax)
    x_ref_min = tf.constant(np.array([ymin, logrmin]))
    x_ref_max = tf.constant(np.array([ymax, logrmax]))

    numpy_grid = GetYRgrid(dipole_path)
    tfgrid = tf.constant(numpy_grid, dtype = tf.float64)
    #print(f"Type of tfgrid: {type(tfgrid)}")
    
    n_dim = 9
    
    n_iter = 10

    main_dimension = 0 #main dimension not provided as argument. Supply beta value for main dimension first

    vegas_instance = VegasFlow(n_dim, n_events, xmin=[0, 0, 0, 0, 0, 0, 0, 0, 0], xmax=[1, 1, xmax, xmax, 2.0*np.pi, xmax, 2.0*np.pi, xmax, 2.0*np.pi],main_dimension = main_dimension)

    integrand_vegasflow = lambda xx: integrand(xx,tfgrid,Csq,x_ref_min,x_ref_max,Mx_const,Q=Q,beta=beta_list,xpom=xpom)
    
    vegas_instance.compile(integrand_vegasflow)

    # Load pre-trained grid if available
    if input_grid_path != "":
        vegas_instance.load_grid(input_grid_path)

    print(f"VEGAS MC, npoints={n_events}:")
    start = time.time()
    result = vegas_instance.run_integration(n_iter)
    result_final = list((list([prefactorsT[n]*elem for n, elem in enumerate(result[i])]) for i in [0,1]))
    end = time.time()
    print(f"Result of VEGAS: {result_final}")
    print(f"Vegas took: time (s): {end-start}")

    vegas_instance.freeze_grid()
    
    for n, beta in enumerate(betavals):
        # --- Organize data into dictionaries ---
        # Input parameters
        param_keys = ["Q", "x", "xmax", "neval", "dipole_path", "Csq"]
        params = {k: args[k] for k in param_keys}
        params["beta"] = beta

        params["batched"] = False
        params["grid_adaptation_beta"] = betavals[main_dimension]
        
        if len(betavals) > 1:
            params["batched"] = True
        
        # Metadata
        meta_keys = ["save_dir", "input_grid_path"]
        meta = {k: args[k] for k in meta_keys}
        
        
        # Provenance info
        provenance_keys = ["script_file", "git_commit", "git_is_dirty", "script_is_dirty"]
        provenance = {k: args[k] for k in provenance_keys}

        
        result_filename = (f"result_mcint_neval_{n_events}_xmax_{xmaxval}_x_{xpomval}_Q_{Qval}_beta_{beta}.txt")
        json_filename = (f"result_neval_{n_events}_xmax_{xmaxval}_x_{xpomval}_Q_{Qval}_beta_{beta}.json")
    
        result_path = os.path.join(save_dir, result_filename)
        json_file_path = os.path.join(save_dir, json_filename)
    
        chisqdof=-1.0
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)
            with open(result_path, "w") as f:
                # VegasFlow does not return chisq/dof. Setting it to -1.0.
                f.write(f"({result_final[0]}, {result_final[1]}, {chisqdof})")
            
            trained_grid_filename = (f"grid_niter_{n_iter}_neval_{n_events}_x_{xpomval}_Q_{Qval}_beta_{betavals[main_dimension]}.json")
            meta["trained_grid"] = trained_grid_filename
            trained_grid_path = os.path.join(save_dir, trained_grid_filename)
            if n == 0:
                vegas_instance.save_grid(trained_grid_path)
        
        # --- Save JSON Payload ---
        
        if json_filename != "":
            payload = {
                "parameters": params,
                "metrics": {
                    "result": result_final[0][n],
                    "error": result_final[1][n],
                    "chi2/dof": chisqdof
                },
                "provenance": provenance,
                "meta": meta
            }
        
        # Logic for current directory if save_dir is empty
        if save_dir != "":
            target_dir = save_dir 
            path = os.path.join(target_dir, json_filename)
            
            with open(path, "w") as io:
                json.dump(payload, io, indent=4)
                
            print(f"Saved JSON results to {path}")



