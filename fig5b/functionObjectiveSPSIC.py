"""Evaluate the transformed local-SIC power-control objective for numerical diagnostics."""

import numpy as np


def functionObjectiveSPSIC(P, K, AV, Noise_power, l1, l2, z1, z2, zt1, zt2, J, d1, d2, d3):
    """Return the minimization form of the transformed local-SIC objective."""
    # Split the stacked optimization vector into the two stream-power vectors.
    p1, p2 = P[:K], P[K:]
    # Reconstruct the desired and interference terms of the transformed problem.
    M1, M2 = d1 * p1, d2 * p2
    N1 = AV.zeta1_V @ p1 - M1 + AV.zeta1_H @ p2 + AV.xi1_V @ p1 + AV.xi1_H @ p2 + Noise_power
    N2 = AV.zeta2_V @ p1 - d3 * p1 + AV.zeta2_H @ p2 - M2 + AV.xi2_V @ p1 + AV.xi2_H @ p2 + Noise_power
    # Evaluate the quadratic-transform utility and dispersion penalty.
    Q1 = np.sum(2 * z1 * np.sqrt((1 + l1) * np.maximum(0, M1)) + 2 * z2 * np.sqrt((1 + l2) * np.maximum(0, M2)))
    Q2 = np.sum(z1**2 * (M1 + N1) + z2**2 * (M2 + N2))
    Q3 = np.sum(J / np.sqrt(np.maximum(np.finfo(float).eps, 1 + 2 * zt1 * np.sqrt(np.maximum(np.finfo(float).eps, N1)) - zt1**2 * M1)) +
                J / np.sqrt(np.maximum(np.finfo(float).eps, 1 + 2 * zt2 * np.sqrt(np.maximum(np.finfo(float).eps, N2)) - zt2**2 * M2)))
    return Q2 + Q3 - Q1, None
