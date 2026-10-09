"""Evaluate the infinite-blocklength sum spectral efficiency of the SP local-SIC receiver."""

from types import SimpleNamespace
import numpy as np


def functionComputeSESPSIC_IFB(AV, Pk1, Pk2, Noise_power):
    """Return the local-SIC SINRs and Shannon sum SE for the current powers."""
    # Separate desired-signal power from interference, estimation error, and noise.
    M1 = np.diag(AV.zeta1_V) * Pk1
    M2 = np.diag(AV.zeta2_H) * Pk2
    N1 = AV.zeta1_V @ Pk1 - M1 + AV.zeta1_H @ Pk2 + AV.xi1_V @ Pk1 + AV.xi1_H @ Pk2 + Noise_power
    N2 = AV.zeta2_V @ Pk1 - np.diag(AV.zeta2_V) * Pk1 + AV.zeta2_H @ Pk2 - M2 + AV.xi2_V @ Pk1 + AV.xi2_H @ Pk2 + Noise_power
    # Compute the instantaneous SINR of every decoded stream.
    gamma1, gamma2 = M1 / N1, M2 / N2
    return SimpleNamespace(name="SE related parameters", gamma1=gamma1, gamma2=gamma2,
                           N1_mathcal=N1, N2_mathcal=N2, M1_mathcal=M1, M2_mathcal=M2,
                           SE=np.sum(np.log2(1 + gamma1)) + np.sum(np.log2(1 + gamma2)))
