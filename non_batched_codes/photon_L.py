import time
import math
import numpy as np
from vegasflow import VegasFlow
import tensorflow as tf
import argparse

def GNLOL(Q, beta, z0, z1, x20, th20, x20b, th20b, x21, th21, x21b, th21b):
    # Precompute some frequently used quantities
    Mx = tf.sqrt(1.0/beta - 1.0) * Q
    z2 = 1.0 - z0 - z1

    # Cosine differences
    cos_th21_m_th20 = tf.cos(th21 - th20)
    cos_th21b_m_th20b = tf.cos(th21b - th20b)

    # X012
    X012 = tf.sqrt(
        z0 * z1 * (x21**2 + x20**2 - 2*x21*x20*cos_th21_m_th20) +
        z0 * z2 * x20**2 +
        z1 * z2 * x21**2
    )

    # X012b
    X012b = tf.sqrt(
        z0 * z1 * (x21b**2 + x20b**2 - 2*x21b*x20b*cos_th21b_m_th20b) +
        z0 * z2 * x20b**2 +
        z1 * z2 * x21b**2
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
        z0 * z2 * (
            x20b**2 + x20**2 - 2*x20b*x20*cos_th20b_m_th20
        )
    )
    Y012_part3 = (
        z1 * z2 * (
            x21b**2 + x21**2 - 2*x21b*x21*cos_th21b_m_th21
        )
    )
    Y012 = tf.sqrt(Y012_part1 + Y012_part2 + Y012_part3)

    # Basic dot products
    dot_x20_x20b = x20 * x20b * tf.cos(th20 - th20b)
    dot_x21_x21b = x21 * x21b * tf.cos(th21 - th21b)
    dot_x20_x21b = x20 * x21b * tf.cos(th20 - th21b)
    dot_x21_x20b = x21 * x20b * tf.cos(th21 - th20b)

    epsilon = 1e-10

    # Final result
    result = (
        z0 * z1 *
        tf.math.special.bessel_k0(Q * X012) * tf.math.special.bessel_k0(Q * X012b) *(1.0 / Y012) *tf.math.special.bessel_j1(Mx * Y012) *
            (
                z1**2*(2.0*z0*(1.0 - z1) + z2**2)*dot_x20_x20b/((x20**2 + epsilon) * (x20b**2 + epsilon))
              + z0**2*(2.0*z1*(1.0 - z0) + z2**2)*dot_x21_x21b/((x21**2 + epsilon) * (x21b**2 + epsilon))
              - z0*z1*(z0*(1.0 - z0) + z1*(1.0 - z1))*
                (
                    dot_x20_x21b/((x20**2 + epsilon) * (x21b**2 + epsilon))
                  + dot_x21_x20b/((x21**2 + epsilon) * (x20b**2 + epsilon))
                )
            )
    )
    return result

def S(a, r):
    return tf.exp(-a * r**2)

def S012(a, x20, th20, x21, th21):
    Nc = 3.0
    CF = 4.0/3.0
    x10 = tf.sqrt(x20**2 + x21**2 - 2.0 * x20 * x21 * tf.cos(th20 - th21))
    return (Nc / (2.0 * CF)) * (S(a, x20) * S(a, x21) - (1.0 / Nc**2) * S(a, x10))

@tf.function
def integrand(xx, a=0.1, Q=2.0, beta=0.5):
    # Unpack the tensor
    z0, t, x20, x20b, th20b, x21, th21, x21b, th21b = tf.unstack(xx, axis=-1)

    measure = x20 * x20b * x21 * x21b

    zmin = 0.0
    zmax = (1.0 - z0)
    z1 = zmin + (zmax - zmin)*t
    jac = (zmax - zmin)

    th20 = 0
    return jac*measure * GNLOL(Q, beta, z0, z1, x20, th20, x20b, th20b, x21, th21, x21b, th21b) * (1.0 - S012(a, x20, th20, x21, th21)) * (1.0 - S012(a, x20b, th20b, x21b, th21b))

# --- VERIFICATION BLOCK ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trip-trip (L) contribution from dipole grid.")
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


