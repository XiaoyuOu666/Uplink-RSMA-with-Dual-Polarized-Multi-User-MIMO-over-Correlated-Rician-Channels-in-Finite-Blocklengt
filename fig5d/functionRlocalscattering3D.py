"""Construct a three-dimensional local-scattering spatial correlation matrix."""

import numpy as np
from scipy.integrate import dblquad


def functionRlocalscattering3D(M_H, M_V, d_H, d_V, varphi, stdphi, theta, stdtheta, dist):
    """Integrate the angular power distribution to form a Hermitian correlation matrix."""
    # Select the normalized azimuth and elevation power distributions.
    if dist == "Laplace":
        f1 = lambda x: np.exp(-np.sqrt(2) * abs(x - varphi) / stdphi) / (np.sqrt(2) * stdphi)
        f2 = lambda x: np.exp(-np.sqrt(2) * abs(x - theta) / stdtheta) / (np.sqrt(2) * stdtheta)
    elif dist == "Gaussian":
        f1 = lambda x: np.exp(-(x - varphi)**2 / (2 * stdphi**2)) / (np.sqrt(2 * np.pi) * stdphi)
        f2 = lambda x: np.exp(-(x - theta)**2 / (2 * stdtheta**2)) / (np.sqrt(2 * np.pi) * stdtheta)
    elif dist == "Uniform":
        f1, f2 = lambda x: 1 / (2 * stdphi), lambda x: 1 / (2 * stdtheta)
    else:
        raise ValueError("Please provide a valid angle distribution")

    M = M_H * M_V
    R = np.zeros((M, M), complex)
    # Cache correlations by antenna displacement to avoid repeated integration.
    lookup = np.zeros((2 * M_H - 1, M_V), complex)
    i, j = lambda m: m % M_H, lambda m: m // M_H
    p0, p1 = (varphi - stdphi, varphi + stdphi) if dist == "Uniform" else (varphi - 20 * stdphi, varphi + 20 * stdphi)
    t0, t1 = (theta - stdtheta, theta + stdtheta) if dist == "Uniform" else (theta - 20 * stdtheta, theta + 20 * stdtheta)
    # Integrate unique displacements and fill the Hermitian matrix by symmetry.
    for m in range(M):
        for l in range(m + 1):
            a, b = i(l) - i(m) + M_H - 1, j(l) - j(m) + M_V - 1
            if lookup[a, b] == 0:
                fun = lambda p, t: (np.exp(1j * 2 * np.pi * d_V * (j(l) - j(m)) * np.sin(t)) *
                                    np.exp(1j * 2 * np.pi * d_H * (i(l) - i(m)) * np.cos(t) * np.sin(p)) * f1(p) * f2(t))
                real = dblquad(lambda t, p: fun(p, t).real, p0, p1, lambda p: t0, lambda p: t1)[0]
                imag = dblquad(lambda t, p: fun(p, t).imag, p0, p1, lambda p: t0, lambda p: t1)[0]
                lookup[a, b] = real + 1j * imag
            R[m, l], R[l, m] = lookup[a, b], lookup[a, b].conjugate()
    return R
