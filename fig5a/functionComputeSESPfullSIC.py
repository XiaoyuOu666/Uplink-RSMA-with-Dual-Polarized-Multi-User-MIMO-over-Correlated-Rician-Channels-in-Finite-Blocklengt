"""Evaluate the finite-blocklength sum spectral efficiency of the SP full-SIC benchmark."""

from types import SimpleNamespace
import numpy as np
from scipy.special import ndtri


def functionComputeSESPfullSIC(AV, Pk1, Pk2, Noise_power, Nth, Eth):
    """Return the full-SIC SINRs and finite-blocklength sum SE for the current powers."""
    upper = np.triu(np.ones((len(Pk1), len(Pk1))), 1)
    # Separate desired-signal power from interference, estimation error, and noise.
    M1 = np.diag(AV.zeta1_V) * Pk1
    M2 = np.diag(AV.zeta2_H) * Pk2
    N1 = (upper * AV.zeta1_V) @ Pk1 + (upper * AV.zeta1_H) @ Pk2 + np.diag(AV.zeta1_H) * Pk2 + AV.xi1_V @ Pk1 + AV.xi1_H @ Pk2 + Noise_power
    N2 = (upper * AV.zeta2_V) @ Pk1 + (upper * AV.zeta2_H) @ Pk2 + AV.xi2_V @ Pk1 + AV.xi2_H @ Pk2 + Noise_power
    # Compute the instantaneous SINR of every decoded stream.
    gamma1, gamma2 = M1 / N1, M2 / N2
    # Apply the finite-blocklength normal approximation to each SINR.
    f = lambda x: np.log2(1 + x) - np.sqrt(2 * x / ((1 + x) * Nth)) * ndtri(1 - Eth) / np.log(2)
    return SimpleNamespace(name="SE related parameters", gamma1=gamma1, gamma2=gamma2,
                           N1_mathcal=N1, N2_mathcal=N2, M1_mathcal=M1, M2_mathcal=M2,
                           SE=np.sum(f(gamma1)) + np.sum(f(gamma2)))
