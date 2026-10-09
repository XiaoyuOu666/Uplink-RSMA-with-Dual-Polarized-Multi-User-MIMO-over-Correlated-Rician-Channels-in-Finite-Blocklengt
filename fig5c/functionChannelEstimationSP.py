"""Generate correlated Rician single-polarized channels and their LMMSE estimates."""

import numpy as np
from scipy.linalg import sqrtm


def functionChannelEstimationSP(K, M, rho, Rcorr_BSSP, Rcorr_UESP, K_overline, h_LoS_1, h_LoS_2, Noise_power):
    """Generate an SP channel realization and return its LMMSE estimate and error covariance."""
    tau_p = K
    # Generate pilot noise and reusable standard complex Gaussian samples.
    z = np.sqrt(Noise_power / 2) * (np.random.randn(4 * M, K) + 1j * np.random.randn(4 * M, K))
    iid = np.sqrt(.5) * (np.random.randn(2 * M, 2, K) + 1j * np.random.randn(2 * M, 2, K))
    # Form correlated Rician channel realizations from their mean and covariance.
    h, R = np.empty((4 * M, K), complex), np.empty((4 * M, 4 * M, K), complex)
    for k in range(K):
        nlos = sqrtm(Rcorr_BSSP[:, :, k]) @ iid[:, :, k] @ sqrtm(Rcorr_UESP[:, :, k])
        mean = np.sqrt(K_overline[k] / (1 + K_overline[k])) * np.r_[h_LoS_1[:, k], h_LoS_2[:, k]]
        h[:, k] = mean + np.sqrt(1 / (1 + K_overline[k])) * nlos.ravel(order="F")
        R[:, :, k] = np.kron(Rcorr_UESP[:, :, k].T, Rcorr_BSSP[:, :, k]) / (1 + K_overline[k])
    # Apply pilot transmission before computing the LMMSE estimates.
    y = np.sqrt(2 * rho * tau_p) * h + z
    h_hat = np.empty_like(h); B = np.empty_like(R)
    for k in range(K):
        mean = np.sqrt(K_overline[k] / (1 + K_overline[k])) * np.r_[h_LoS_1[:, k], h_LoS_2[:, k]]
        # Solve once for the innovation and covariance right-hand sides.
        solution = np.linalg.solve(2 * rho * tau_p * R[:, :, k] + Noise_power * np.eye(4 * M), np.column_stack((y[:, k] - np.sqrt(2 * rho * tau_p) * mean, R[:, :, k])))
        h_hat[:, k] = mean + np.sqrt(2 * rho * tau_p) * R[:, :, k] @ solution[:, 0]
        B[:, :, k] = R[:, :, k] - 2 * rho * tau_p * R[:, :, k] @ solution[:, 1:]
    return h_hat[:2 * M], h_hat[2 * M:], B[:2 * M, :2 * M], B[2 * M:, 2 * M:]
