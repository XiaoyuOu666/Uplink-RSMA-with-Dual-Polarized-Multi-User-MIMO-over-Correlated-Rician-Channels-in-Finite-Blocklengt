"""Generate correlated Rician dual-polarized channels and their LMMSE estimates."""

import numpy as np
from scipy.linalg import sqrtm


def functionChannelEstimationDP(K, M, rho, Rcorr_V, Rcorr_H, K_overline, h_LoS_V, h_LoS_H, Noise_power):
    """Generate a DP channel realization and return its LMMSE estimate and error covariance."""
    tau_p = K
    # Generate pilot noise and reusable standard complex Gaussian samples.
    z = np.sqrt(Noise_power / 2) * (np.random.randn(4 * M, K) + 1j * np.random.randn(4 * M, K))
    iid_V = np.sqrt(.5) * (np.random.randn(2 * M, K) + 1j * np.random.randn(2 * M, K))
    iid_H = np.sqrt(.5) * (np.random.randn(2 * M, K) + 1j * np.random.randn(2 * M, K))
    # Form correlated Rician channel realizations from their mean and covariance.
    h = np.empty((4 * M, K), complex)
    for k in range(K):
        nlos_V = sqrtm(Rcorr_V[:, :, k]) @ iid_V[:, k]
        nlos_H = sqrtm(Rcorr_H[:, :, k]) @ iid_H[:, k]
        h[:, k] = np.sqrt(K_overline[k] / (1 + K_overline[k])) * np.r_[h_LoS_V[:, k], h_LoS_H[:, k]] + np.r_[nlos_V, nlos_H]
    # Apply pilot transmission before computing the LMMSE estimates.
    y = np.sqrt(2 * rho * tau_p) * h + z
    hV, hH = np.empty((2 * M, K), complex), np.empty((2 * M, K), complex)
    BV, BH = np.empty((2 * M, 2 * M, K), complex), np.empty((2 * M, 2 * M, K), complex)
    for k in range(K):
        mean = np.sqrt(K_overline[k] / (1 + K_overline[k]))
        # Solve once for the innovation and covariance right-hand sides.
        solutionV = np.linalg.solve(2 * rho * tau_p * Rcorr_V[:, :, k] + Noise_power * np.eye(2 * M), np.column_stack((y[:2 * M, k] - np.sqrt(2 * rho * tau_p) * mean * h_LoS_V[:, k], Rcorr_V[:, :, k])))
        solutionH = np.linalg.solve(2 * rho * tau_p * Rcorr_H[:, :, k] + Noise_power * np.eye(2 * M), np.column_stack((y[2 * M:, k] - np.sqrt(2 * rho * tau_p) * mean * h_LoS_H[:, k], Rcorr_H[:, :, k])))
        hV[:, k] = mean * h_LoS_V[:, k] + np.sqrt(2 * rho * tau_p) * Rcorr_V[:, :, k] @ solutionV[:, 0]
        hH[:, k] = mean * h_LoS_H[:, k] + np.sqrt(2 * rho * tau_p) * Rcorr_H[:, :, k] @ solutionH[:, 0]
        BV[:, :, k] = Rcorr_V[:, :, k] - 2 * rho * tau_p * Rcorr_V[:, :, k] @ solutionV[:, 1:]
        BH[:, :, k] = Rcorr_H[:, :, k] - 2 * rho * tau_p * Rcorr_H[:, :, k] @ solutionH[:, 1:]
    return hV, hH, BV, BH
