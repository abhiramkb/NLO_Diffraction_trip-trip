import time
import math
import numpy as np
from vegasflow import VegasFlow
import tensorflow as tf
import argparse

@tf.function(jit_compile=True)
def Sum_Y_terms(Q, beta, xx):
    z0  = xx[..., 0]
    t   = xx[..., 1]
    x20 = xx[..., 2]
    x20b = xx[..., 3]
    th20b = xx[..., 4]
    x21 = xx[..., 5]
    th21 = xx[..., 6]
    x21b = xx[..., 7]
    th21b = xx[..., 8]

    z1 = (1.0 - z0)*t
    z2 = 1.0 - z0 - z1

    x20sq = x20**2
    x20bsq = x20b**2
    x21sq = x21**2
    x21bsq = x21b**2

    # Basic dot products
    dot_x20_x20b = x20 * x20b * tf.cos(th20b)
    dot_x21_x21b = x21 * x21b * tf.cos(th21 - th21b)
    dot_x20_x21b = x20 * x21b * tf.cos(th21b)
    dot_x21_x20b = x21 * x20b * tf.cos(th21 - th20b)
    dot_x20_x21 = x20 * x21 * tf.cos(th21)
    dot_x20b_x21b = x20b * x21b * tf.cos(th20b - th21b)

    # Composite dot products from (29)
    z0by1mz1 = z0/(1.0-z1)
    dot_x20_x0p2c1 = dot_x20_x21 - z0by1mz1 * x20sq
    dot_x20b_x0p2c1b = dot_x20b_x21b - z0by1mz1 * x20bsq
    dot_x20b_x0p2c1 = dot_x21_x20b - z0by1mz1 * dot_x20_x20b
    dot_x20_x0p2c1b = dot_x20_x21b - z0by1mz1 * dot_x20_x20b
    dot_x0p2c1_x0p2c1b = (
        dot_x21_x21b -
        z0by1mz1 * (dot_x21_x20b + dot_x20_x21b) +
        (z0**2/(1.0 - z1)**2) * dot_x20_x20b
    )

    # Composite dot products from (30)
    z1by1mz0 = z1/(1.0 - z0)
    dot_x21_x0c1p2 = z1by1mz0 * x21sq - dot_x20_x21
    dot_x21_x0c1p2b = z1by1mz0 * dot_x21_x21b - dot_x21_x20b
    dot_x21b_x0c1p2 = z1by1mz0 * dot_x21_x21b - dot_x20_x21b
    dot_x21b_x0c1p2b = z1by1mz0 * x21bsq - dot_x20b_x21b
    dot_x0c1p2_x0c1p2b = (
        dot_x20_x20b -
        z1by1mz0 * (dot_x21_x20b + dot_x20_x21b) +
        z1by1mz0**2 * dot_x21_x21b
    )

    # Composite dot products from (33)
    dot_x0p2c1_x0c1p2b = (
        z1by1mz0 * dot_x21_x21b - dot_x21_x20b -
        (z0*z1/((1.0 - z0)*(1.0 - z1))) * dot_x20_x21b +
        (z0/(1.0 - z1)) * dot_x20_x20b
    )
    dot_x0c1p2_x0p2c1b = (
        z1by1mz0 * dot_x21_x21b - dot_x20_x21b -
        (z0*z1/((1.0 - z0)*(1.0 - z1))) * dot_x21_x20b +
        (z0/(1.0 - z1)) * dot_x20_x20b
    )
    dot_x20_x0c1p2b = z1by1mz0 * dot_x20_x21b - dot_x20_x20b
    dot_x21b_x0p2c1 = dot_x21_x21b - (z0/(1.0 - z1)) * dot_x20_x21b
    dot_x20b_x0c1p2 = z1by1mz0 * dot_x21_x20b - dot_x20_x20b
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
    Y_b_reg = (z1**2 / (x20sq * x20bsq)) * (term1b + term2b)

    term1c = (
        (z1**2 + (1.0 - z0)**2) * (1.0 - 2*z0*(1.0 - z0)) *
        dot_x0c1p2_x0c1p2b * dot_x21_x21b
    )
    term2c = -(
        ((1.0 - z0)**2 - z1**2) * (2.0*z0 - 1.0) *
        (dot_x21_x0c1p2 * dot_x21b_x0c1p2b - dot_x21_x0c1p2b * dot_x21b_x0c1p2)
    )
    Y_c_reg = (z0**2 / (x21sq * x21bsq)) * (term1c + term2c)

    term1d = (z0**2 * z1**2 * z2**2) / (1.0 - z1)**2
    term2d = -(
        (z0**2 * z1**3 * z2) / (1.0 - z1)
    ) * (dot_x20_x0p2c1 / x20sq + dot_x20b_x0p2c1b / x20bsq)
    term3d = (
        (z0**2 * z1 * z2 * (1.0 - z0)**2) / (1.0 - z1)
    ) * (dot_x21_x0c1p2 / x21sq + dot_x21b_x0c1p2b / x21bsq)
    Y_d_inst = term1d + term2d + term3d

    term1e = (z0**2 * z1**2 * z2**2) / (1.0 - z0)**2
    term2e = (
        (z0**3 * z1**2 * z2) / (1.0 - z0)
    ) * (dot_x21_x0c1p2 / x21sq + dot_x21b_x0c1p2b / x21bsq)
    term3e = -(
        (z0 * z1**2 * z2 * (1.0 - z1)**2) / (1.0 - z0)
    ) * (dot_x20_x0p2c1 / x20sq + dot_x20b_x0p2c1b / x20bsq)
    Y_e_inst = term1e + term2e + term3e

    term1bc_pref = -z0 * z1 * (z0*(1.0 - z1) + z1*(1.0 - z0)) * (z0*(1.0 - z0) + z1*(1.0 - z1))
    term1bc_prods = (
        dot_x0c1p2_x0p2c1b * dot_x21_x20b / (x21sq * x20bsq) +
        dot_x0p2c1_x0c1p2b * dot_x20_x21b / (x20sq * x21bsq)
    )
    term2bc_pref = z0 * z1 * z2 * (z0 - z1)**2
    term2bc_prod1 = (
        (dot_x20_x0p2c1 * dot_x21b_x0c1p2b - dot_x20_x0c1p2b * dot_x21b_x0p2c1) /
        (x20sq * x21bsq)
    )
    term2bc_prod2 = (
        (dot_x21_x0c1p2 * dot_x20b_x0p2c1b - dot_x21_x0p2c1b * dot_x20b_x0c1p2) /
        (x21sq * x20bsq)
    )
    Y_bc_interf = term1bc_pref * term1bc_prods + term2bc_pref * (term2bc_prod1 + term2bc_prod2)

    return Y_b_reg + Y_c_reg + Y_d_inst + Y_e_inst + Y_bc_interf 

@tf.function(jit_compile=True)
def funcY012(Q, beta, xx):
    z0  = xx[..., 0]
    t   = xx[..., 1]
    x20 = xx[..., 2]
    x20b = xx[..., 3]
    th20b = xx[..., 4]
    x21 = xx[..., 5]
    th21 = xx[..., 6]
    x21b = xx[..., 7]
    th21b = xx[..., 8]

    z1 = (1.0 - z0)*t
    
    z2 = 1.0 - z0 - z1
    # Cosine differences
    cos_th21_m_th20 = tf.cos(th21)
    cos_th21b_m_th20b = tf.cos(th21b - th20b)
    # Cosine differences needed for Y012
    cos_th21b_m_th21 = tf.cos(th21b - th21)
    cos_th21b_m_th20 = tf.cos(th21b)
    cos_th20b_m_th21 = tf.cos(th20b - th21)
    cos_th20b_m_th20 = tf.cos(th20b)

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
        z0 * z2 * (
            x20b**2 + x20**2 - 2*x20b*x20*cos_th20b_m_th20
        )
    )
    Y012_part3 = (
        z1 * z2 * (
            x21b**2 + x21**2 - 2*x21b*x21*cos_th21b_m_th21
        )
    )
    return tf.sqrt(Y012_part1 + Y012_part2 + Y012_part3)

@tf.function(jit_compile=False)
def GNLOT(Q, beta, xx):
    """
    TensorFlow version of GNLOT.
    All inputs are tensors (scalar or batched). Assumes same shape or broadcastable.
    """

    z0  = xx[..., 0]
    t   = xx[..., 1]
    x20 = xx[..., 2]
    x20b = xx[..., 3]
    th20b = xx[..., 4]
    x21 = xx[..., 5]
    th21 = xx[..., 6]
    x21b = xx[..., 7]
    th21b = xx[..., 8]

    z1 = (1.0 - z0)*t
    
    # X012
    X012 = tf.sqrt(
        z0 * z1 * (x21**2 + x20**2 - 2*x21*x20*tf.cos(th21)) +
        z0 * (1.0 - z0 - z1) * x20**2 +
        z1 * (1.0 - z0 - z1) * x21**2
    )

    # X012b
    X012b = tf.sqrt(
        z0 * z1 * (x21b**2 + x20b**2 - 2*x21b*x20b*tf.cos(th21b - th20b)) +
        z0 * (1.0 - z0 - z1) * x20b**2 +
        z1 * (1.0 - z0 - z1) * x21b**2
    )

    Y012 = funcY012(Q, beta, xx)
    
    sum_Y_terms = Sum_Y_terms(Q, beta, xx)

    # Final result
    result = (
        z0 * z1 *
        tf.math.special.bessel_k1(Q * X012) * tf.math.special.bessel_k1(Q * X012b) *
        (1.0 / (X012 * X012b)) *
        (1.0 / Y012) *
        tf.math.special.bessel_j1(tf.sqrt(1.0/beta - 1.0) * Q * Y012) *
        sum_Y_terms
    )
    return result

@tf.function(jit_compile=True)
def S(a, r):
    return tf.exp(-a * r**2)

@tf.function(jit_compile=True)
def S012(a, x20, th20, x21, th21):
        Nc = 3.0
        CF = 4.0/3.0
        x10 = tf.sqrt(x20**2 + x21**2 - 2.0 * x20 * x21 * tf.cos(th21-th20))
        return (Nc / (2.0 * CF)) * (S(a, x20) * S(a, x21) - (1.0 / Nc**2) * S(a, x10))

@tf.function(jit_compile=False)
def integrand(xx, a=0.1, Q=2.0, beta=0.5):
    # Unpack the tensor
    #z0, t, x20, x20b, th20b, x21, th21, x21b, th21b = tf.unstack(xx, axis=-1)

    z0  = xx[..., 0]
    t   = xx[..., 1]
    x20 = xx[..., 2]
    x20b = xx[..., 3]
    th20b = xx[..., 4]
    x21 = xx[..., 5]
    th21 = xx[..., 6]
    x21b = xx[..., 7]
    th21b = xx[..., 8]

    #zmin = 0.0
    #zmax = (1.0 - z0)
    #z1 = zmin + (zmax - zmin)*t
    #jac = (zmax - zmin)
    #z1 = (1.0 - z0)*t

    return (1.0 - z0) * x20 * x20b * x21 * x21b * GNLOT(Q, beta, xx) * (1.0 - S012(a, x20, 0.0, x21, th21)) * (1.0 - S012(a, x20b, th20b, x21b, th21b))

# --- VERIFICATION BLOCK ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trip-trip (T) contribution from dipole grid.")
    parser.add_argument("-Q", type=float, default=3.1622, help="Photon virtuality Q")
    parser.add_argument("--beta", type=float, default=0.5, help="Diffraction variable beta")
    parser.add_argument("--a", type=float, default=0.1, help="Dipole parameter a")
    parser.add_argument("--events", type=int, default=1000000, help="Number of integration points")
    args = parser.parse_args()

    Q=tf.constant(args.Q, dtype=tf.float64)
    beta=tf.constant(args.beta, dtype=tf.float64)
    a = tf.constant(args.a, dtype=tf.float64)

    n_dim = 9
    n_events = args.events
    n_iter = 10

    xmax = 40.0

    vegas_instance = VegasFlow(n_dim, n_events, xmin=[0, 0, 0, 0, 0, 0, 0, 0, 0], xmax=[1, 1, xmax, xmax, 2.0*np.pi, xmax, 2.0*np.pi, xmax, 2.0*np.pi])

    integrand_vegasflow = lambda xx: integrand(xx,a=a,Q=Q,beta=beta)
    
    vegas_instance.compile(integrand_vegasflow)

    print(f"VEGAS MC, npoints={n_events}:")
    start = time.time()
    result = vegas_instance.run_integration(n_iter)
    end = time.time()
    print(f"Result of VEGAS: {result}")
    print(f"Vegas took: time (s): {end-start}")


