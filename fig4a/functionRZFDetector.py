"""Construct normalized regularized zero-forcing receive combiners."""

import numpy as np


def functionRZFDetector(h_est_V, h_est_H, Noise_power, K, M):
    """Solve the regularized Gram system and normalize the RZF combiners."""
    # Regularize the estimated-channel Gram matrix by the receiver noise power.
    H = np.hstack((h_est_V, h_est_H))
    Q = np.linalg.solve(H.conj().T @ H + Noise_power * np.eye(2 * K), H.conj().T).conj().T
    q1, q2 = Q[:, :K], Q[:, K:]
    return q1 / np.linalg.norm(q1, axis=0), q2 / np.linalg.norm(q2, axis=0)
