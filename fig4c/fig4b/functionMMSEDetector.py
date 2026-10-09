"""Construct normalized MMSE combiners for the selected receiver architecture."""

import numpy as np


def functionMMSEDetector(h_est_V, h_est_H, Rerror_V, Rerror_H, M, K, Noise_power, which_scheme):
    """Solve the receiver covariance systems and normalize the resulting combiners."""
    # Assemble the signal, error, and noise covariance seen by the receiver.
    J = Noise_power * np.eye(2 * M, dtype=complex)
    for k in range(K):
        J += np.outer(h_est_V[:, k], h_est_V[:, k].conj()) + np.outer(h_est_H[:, k], h_est_H[:, k].conj()) + Rerror_V[:, :, k] + Rerror_H[:, :, k]
    q1 = np.linalg.solve(J, h_est_V)
    # Use the covariance appropriate to no-SIC, local-SIC, or full-SIC decoding.
    if which_scheme in ("DP-noSIC", "SP-noSIC"):
        q2 = np.linalg.solve(J, h_est_H)
    else:
        q2 = np.empty((2 * M, K), complex)
        for k in range(K):
            J2 = J - np.outer(h_est_V[:, k], h_est_V[:, k].conj()) - np.outer(h_est_H[:, k], h_est_H[:, k].conj())
            q2[:, k] = np.linalg.solve(J2, h_est_H[:, k])
    # Normalize every combiner so that subsequent noise powers remain consistent.
    q1 /= np.linalg.norm(q1, axis=0); q2 /= np.linalg.norm(q2, axis=0)
    return q1, q2
