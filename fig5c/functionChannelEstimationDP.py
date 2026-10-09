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
    hV = np.zeros((2 * M, K, 3), complex); hH = np.zeros((2 * M, K, 3), complex)
    BV = np.zeros((2 * M, 2 * M, K, 3), complex); BH = np.zeros_like(BV)
    for i in range(3):
        # Form correlated Rician channel realizations from their mean and covariance.
        h = np.empty((4 * M, K), complex)
        for k in range(K):
            nV = sqrtm(Rcorr_V[:, :, k, i]) @ iid_V[:, k]
            nH = sqrtm(Rcorr_H[:, :, k, i]) @ iid_H[:, k]
            mean = np.sqrt(K_overline[k] / (1 + K_overline[k])) * np.r_[h_LoS_V[:, k, i], h_LoS_H[:, k, i]]
            h[:, k] = mean + np.r_[nV, nH]
        # Apply pilot transmission before computing the LMMSE estimates.
        y = np.sqrt(2 * rho * tau_p) * h + z
        for k in range(K):
            mean = np.sqrt(K_overline[k] / (1 + K_overline[k]))
            RV, RH = Rcorr_V[:, :, k, i], Rcorr_H[:, :, k, i]
            # Solve once for the innovation and covariance right-hand sides.
            solutionV = np.linalg.solve(2 * rho * tau_p * RV + Noise_power * np.eye(2 * M), np.column_stack((y[:2 * M, k] - np.sqrt(2 * rho * tau_p) * mean * h_LoS_V[:, k, i], RV)))
            solutionH = np.linalg.solve(2 * rho * tau_p * RH + Noise_power * np.eye(2 * M), np.column_stack((y[2 * M:, k] - np.sqrt(2 * rho * tau_p) * mean * h_LoS_H[:, k, i], RH)))
            hV[:, k, i] = mean * h_LoS_V[:, k, i] + np.sqrt(2 * rho * tau_p) * RV @ solutionV[:, 0]
            hH[:, k, i] = mean * h_LoS_H[:, k, i] + np.sqrt(2 * rho * tau_p) * RH @ solutionH[:, 0]
            BV[:, :, k, i] = RV - 2 * rho * tau_p * RV @ solutionV[:, 1:]
            BH[:, :, k, i] = RH - 2 * rho * tau_p * RH @ solutionH[:, 1:]
    return hV, hH, BV, BH
