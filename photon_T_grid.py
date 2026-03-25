import time
import math
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from vegasflow import VegasFlow
import tensorflow as tf
import tensorflow_probability as tfp
import argparse

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


def GetGridAtY(path_to_file,Y):
    interp = ReadBKDipole(path_to_file)
    minr, mult, n, ymin, ymax, yinc = GetGridParameters(path_to_file)
    grid = [float(interp((Y,minr*mult**i))) for i in range(200)]
    return np.array(grid)

def GNLOT(Q, beta, z0, z1, x20, th20, x20b, th20b, x21, th21, x21b, th21b):
    """
    TensorFlow version of GNLOT.
    All inputs are tensors (scalar or batched). Assumes same shape or broadcastable.
    """
    # Precompute some frequently used quantities
    Mx = tf.sqrt(1.0/beta - 1.0) * Q

    # Cosine differences
    cos_th21_m_th20 = tf.cos(th21 - th20)
    cos_th21b_m_th20b = tf.cos(th21b - th20b)

    # X012
    X012 = tf.sqrt(
        z0 * z1 * (x21**2 + x20**2 - 2*x21*x20*cos_th21_m_th20) +
        z0 * (1.0 - z0 - z1) * x20**2 +
        z1 * (1.0 - z0 - z1) * x21**2
    )

    # X012b
    X012b = tf.sqrt(
        z0 * z1 * (x21b**2 + x20b**2 - 2*x21b*x20b*cos_th21b_m_th20b) +
        z0 * (1.0 - z0 - z1) * x20b**2 +
        z1 * (1.0 - z0 - z1) * x21b**2
    )

    # Cosine differences needed for Y012
    cos_th21b_m_th21 = tf.cos(th21b - th21)
    cos_th21b_m_th20 = tf.cos(th21b - th20)
    cos_th20b_m_th21 = tf.cos(th20b - th21)
    cos_th20b_m_th20 = tf.cos(th20b - th20)

    # Y012 (split into two parts for readability)
    Y012_part1 = (
        z0 * z1 * (
            x21b**2 + x20b**2 - 2*x21b*x20b*cos_th21b_m_th20b +
            x21**2 + x20**2 - 2*x21*x20*cos_th21_m_th20 -
            2*x21b*x21*cos_th21b_m_th21 +
            2*x21b*x20*cos_th21b_m_th20 +
            2*x20b*x21*cos_th20b_m_th21 -
            2*x20b*x20*cos_th20b_m_th20
        )
    )
    Y012_part2 = (
        z0 * (1.0 - z0 - z1) * (
            x20b**2 + x20**2 - 2*x20b*x20*cos_th20b_m_th20
        )
    )
    Y012_part3 = (
        z1 * (1.0 - z0 - z1) * (
            x21b**2 + x21**2 - 2*x21b*x21*cos_th21b_m_th21
        )
    )
    Y012 = tf.sqrt(Y012_part1 + Y012_part2 + Y012_part3)

    # Basic dot products
    dot_x20_x20b = x20 * x20b * tf.cos(th20 - th20b)
    dot_x21_x21b = x21 * x21b * tf.cos(th21 - th21b)
    dot_x20_x21b = x20 * x21b * tf.cos(th20 - th21b)
    dot_x21_x20b = x21 * x20b * tf.cos(th21 - th20b)
    dot_x20_x21 = x20 * x21 * tf.cos(th20 - th21)
    dot_x20b_x21b = x20b * x21b * tf.cos(th20b - th21b)

    # Composite dot products from (29)
    dot_x20_x0p2c1 = dot_x20_x21 - (z0/(1.0-z1)) * x20**2
    dot_x20b_x0p2c1b = dot_x20b_x21b - (z0/(1.0-z1)) * x20b**2
    dot_x20b_x0p2c1 = dot_x21_x20b - (z0/(1.0-z1)) * dot_x20_x20b
    dot_x20_x0p2c1b = dot_x20_x21b - (z0/(1.0-z1)) * dot_x20_x20b
    dot_x0p2c1_x0p2c1b = (
        dot_x21_x21b -
        (z0/(1.0-z1)) * (dot_x21_x20b + dot_x20_x21b) +
        (z0**2/(1.0 - z1)**2) * dot_x20_x20b
    )

    # Composite dot products from (30)
    dot_x21_x0c1p2 = (z1/(1.0 - z0)) * x21**2 - dot_x20_x21
    dot_x21_x0c1p2b = (z1/(1.0 - z0)) * dot_x21_x21b - dot_x21_x20b
    dot_x21b_x0c1p2 = (z1/(1.0 - z0)) * dot_x21_x21b - dot_x20_x21b
    dot_x21b_x0c1p2b = (z1/(1.0 - z0)) * x21b**2 - dot_x20b_x21b
    dot_x0c1p2_x0c1p2b = (
        dot_x20_x20b -
        (z1/(1.0 - z0)) * (dot_x21_x20b + dot_x20_x21b) +
        (z1/(1.0 - z0))**2 * dot_x21_x21b
    )

    # Composite dot products from (33)
    dot_x0p2c1_x0c1p2b = (
        (z1/(1.0 - z0)) * dot_x21_x21b - dot_x21_x20b -
        (z0*z1/((1.0 - z0)*(1.0 - z1))) * dot_x20_x21b +
        (z0/(1.0 - z1)) * dot_x20_x20b
    )
    dot_x0c1p2_x0p2c1b = (
        (z1/(1.0 - z0)) * dot_x21_x21b - dot_x20_x21b -
        (z0*z1/((1.0 - z0)*(1.0 - z1))) * dot_x21_x20b +
        (z0/(1.0 - z1)) * dot_x20_x20b
    )
    dot_x20_x0c1p2b = (z1/(1.0 - z0)) * dot_x20_x21b - dot_x20_x20b
    dot_x21b_x0p2c1 = dot_x21_x21b - (z0/(1.0 - z1)) * dot_x20_x21b
    dot_x20b_x0c1p2 = (z1/(1.0 - z0)) * dot_x21_x20b - dot_x20_x20b
    dot_x21_x0p2c1b = dot_x21_x21b - (z0/(1.0 - z1)) * dot_x21_x20b

    # Coefficient functions from Eqs. (29)–(33)
    term1b = (
        (z0**2 + (1.0 - z1)**2) * (1.0 - 2*z1*(1.0 - z1)) *
        dot_x0p2c1_x0p2c1b * dot_x20_x20b
    )
    term2b = -(
        ((1.0 - z1)**2 - z0**2) * (2.0*z1 - 1.0) *
        (dot_x20_x0p2c1 * dot_x20b_x0p2c1b - dot_x20_x0p2c1b * dot_x20b_x0p2c1)
    )
    Y_b_reg = (z1**2 / (x20**2 * x20b**2)) * (term1b + term2b)

    term1c = (
        (z1**2 + (1.0 - z0)**2) * (1.0 - 2*z0*(1.0 - z0)) *
        dot_x0c1p2_x0c1p2b * dot_x21_x21b
    )
    term2c = -(
        ((1.0 - z0)**2 - z1**2) * (2.0*z0 - 1.0) *
        (dot_x21_x0c1p2 * dot_x21b_x0c1p2b - dot_x21_x0c1p2b * dot_x21b_x0c1p2)
    )
    Y_c_reg = (z0**2 / (x21**2 * x21b**2)) * (term1c + term2c)

    term1d = (z0**2 * z1**2 * (1.0 - z0 - z1)**2) / (1.0 - z1)**2
    term2d = -(
        (z0**2 * z1**3 * (1.0 - z0 - z1)) / (1.0 - z1)
    ) * (dot_x20_x0p2c1 / x20**2 + dot_x20b_x0p2c1b / x20b**2)
    term3d = (
        (z0**2 * z1 * (1.0 - z0 - z1) * (1.0 - z0)**2) / (1.0 - z1)
    ) * (dot_x21_x0c1p2 / x21**2 + dot_x21b_x0c1p2b / x21b**2)
    Y_d_inst = term1d + term2d + term3d

    term1e = (z0**2 * z1**2 * (1.0 - z0 - z1)**2) / (1.0 - z0)**2
    term2e = (
        (z0**3 * z1**2 * (1.0 - z0 - z1)) / (1.0 - z0)
    ) * (dot_x21_x0c1p2 / x21**2 + dot_x21b_x0c1p2b / x21b**2)
    term3e = -(
        (z0 * z1**2 * (1.0 - z0 - z1) * (1.0 - z1)**2) / (1.0 - z0)
    ) * (dot_x20_x0p2c1 / x20**2 + dot_x20b_x0p2c1b / x20b**2)
    Y_e_inst = term1e + term2e + term3e

    term1bc_pref = -z0 * z1 * (z0*(1.0 - z1) + z1*(1.0 - z0)) * (z0*(1.0 - z0) + z1*(1.0 - z1))
    term1bc_prods = (
        dot_x0c1p2_x0p2c1b * dot_x21_x20b / (x21**2 * x20b**2) +
        dot_x0p2c1_x0c1p2b * dot_x20_x21b / (x20**2 * x21b**2)
    )
    term2bc_pref = z0 * z1 * (1.0 - z0 - z1) * (z0 - z1)**2
    term2bc_prod1 = (
        (dot_x20_x0p2c1 * dot_x21b_x0c1p2b - dot_x20_x0c1p2b * dot_x21b_x0p2c1) /
        (x20**2 * x21b**2)
    )
    term2bc_prod2 = (
        (dot_x21_x0c1p2 * dot_x20b_x0p2c1b - dot_x21_x0p2c1b * dot_x20b_x0c1p2) /
        (x21**2 * x20b**2)
    )
    Y_bc_interf = term1bc_pref * term1bc_prods + term2bc_pref * (term2bc_prod1 + term2bc_prod2)

    sum_Y_terms = Y_b_reg + Y_c_reg + Y_d_inst + Y_e_inst + Y_bc_interf

    # Final result
    result = (
        z0 * z1 *
        tf.math.special.bessel_k1(Q * X012) * tf.math.special.bessel_k1(Q * X012b) *
        (1.0 / (X012 * X012b)) *
        (1.0 / Y012) *
        tf.math.special.bessel_j1(Mx * Y012) *
        sum_Y_terms
    )
    return result

def S(tfgrid,rmin,rmax, r):
    value = tfp.math.interp_regular_1d_grid(
    x=[r], x_ref_min=rmin, x_ref_max=rmax, y_ref=tfgrid, grid_regularizing_transform=tf.math.log)
    
    return tf.squeeze(value)

def S012(tfgrid,rmin,rmax, x20, th20, x21, th21):
    Nc = 3.0
    CF = 4.0/3.0
    x10 = tf.sqrt(x20**2 + x21**2 - 2.0 * x20 * x21 * tf.cos(th20 - th21))
    return (Nc / (2.0 * CF)) * (S(tfgrid,rmin,rmax, x20) * S(tfgrid,rmin,rmax, x21) - (1.0 / Nc**2) * S(tfgrid,rmin,rmax, x10))

@tf.function
def integrand(xx, tfgrid, rmin, rmax, Q=2.0, beta=0.5):
    # Unpack the tensor
    z0, t, x20, x20b, th20b, x21, th21, x21b, th21b = tf.unstack(xx, axis=-1)

    measure = x20 * x20b * x21 * x21b

    zmin = 0.0
    zmax = (1.0 - z0)
    z1 = zmin + (zmax - zmin)*t
    jac = (zmax - zmin)

    th20 = 0
    return jac*measure * GNLOT(Q, beta, z0, z1, x20, th20, x20b, th20b, x21, th21, x21b, th21b) * (1.0 - S012(tfgrid,rmin,rmax, x20, th20, x21, th21)) * (1.0 - S012(tfgrid,rmin,rmax, x20b, th20b, x21b, th21b))

# --- VERIFICATION BLOCK ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trip-trip (T) contribution from dipole grid.")
    parser.add_argument("-Q", type=float, default=3.1622, help="Photon virtuality Q")
    parser.add_argument("--beta", type=float, default=0.5, help="Diffraction variable beta")
    parser.add_argument("--dipole_path", type=str, required=True, help="Path to the BK table")
    parser.add_argument("--Y", type=float, default=0.0, help="Rapidity Y")
    parser.add_argument("--events", type=int, default=1000000, help="Number of integration points")
    args = parser.parse_args()
    
    Q=tf.constant(args.Q, dtype=tf.float64)
    beta=tf.constant(args.beta, dtype=tf.float64)
    Y=args.Y
    
    th20=tf.constant(0.0, dtype=tf.float64)

    interp = ReadBKDipole(args.dipole_path)
    #Getting grid parameters:
    rmin,mult,n,ymin,ymax,yinc=GetGridParameters(args.dipole_path)
    rmax = rmin*mult**(n-1)
    # Getting grid at fixed rapidity:
    grid = GetGridAtY(args.dipole_path,Y)
    tfgrid = tf.constant(grid)
    
    n_dim = 9
    n_events = args.events
    n_iter = 10

    xmax = 40.0

    vegas_instance = VegasFlow(n_dim, n_events, xmin=[0, 0, 0, 0, 0, 0, 0, 0, 0], xmax=[1, 1, xmax, xmax, 2.0*np.pi, xmax, 2.0*np.pi, xmax, 2.0*np.pi])

    integrand_vegasflow = lambda xx: integrand(xx,tfgrid,rmin,rmax,Q=Q,beta=beta)
    
    vegas_instance.compile(integrand_vegasflow)

    print(f"VEGAS MC, npoints={n_events}:")
    start = time.time()
    result = vegas_instance.run_integration(n_iter)
    end = time.time()
    print(f"Result of VEGAS: {result}")
    print(f"Vegas took: time (s): {end-start}")


