"""Generate correlated Rician dual-polarized channels and their LMMSE estimates."""

import numpy as np
from scipy.linalg import sqrtm


def functionChannelEstimationDP(K, M, rho, Rcorr, K_overline, h_LoS_V, h_LoS_H, Noise_power):
    """Generate a DP channel realization and return its LMMSE estimate and error covariance."""
    tau_p = K
    # Generate pilot noise and reusable standard complex Gaussian samples.
    z = np.sqrt(Noise_power / 2) * (np.random.randn(4 * M, K) + 1j * np.random.randn(4 * M, K))
    iid = np.sqrt(.5) * (np.random.randn(4 * M, K) + 1j * np.random.randn(4 * M, K))
    # Form correlated Rician channel realizations from their mean and covariance.
    h = np.empty((4 * M, K), complex)
    for k in range(K):
        nlos = sqrtm(Rcorr[:, :, k]) @ iid[:, k]
        h[:, k] = np.sqrt(K_overline[k] / (1 + K_overline[k])) * np.r_[h_LoS_V[:, k], h_LoS_H[:, k]] + nlos
    # Apply pilot transmission before computing the LMMSE estimates.
    y = np.sqrt(2 * rho * tau_p) * h + z
    h_hat = np.empty_like(h); B = np.empty((4 * M, 4 * M, K), complex)
    for k in range(K):
        mean = np.sqrt(K_overline[k] / (1 + K_overline[k])) * np.r_[h_LoS_V[:, k], h_LoS_H[:, k]]
        # Solve once for the innovation and covariance right-hand sides.
        solution = np.linalg.solve(2 * rho * tau_p * Rcorr[:, :, k] + Noise_power * np.eye(4 * M), np.column_stack((y[:, k] - np.sqrt(2 * rho * tau_p) * mean, Rcorr[:, :, k])))
        h_hat[:, k] = mean + np.sqrt(2 * rho * tau_p) * Rcorr[:, :, k] @ solution[:, 0]
        B[:, :, k] = Rcorr[:, :, k] - 2 * rho * tau_p * Rcorr[:, :, k] @ solution[:, 1:]
    return h_hat[:2 * M], h_hat[2 * M:], B[:2 * M, :2 * M], B[2 * M:, 2 * M:]
