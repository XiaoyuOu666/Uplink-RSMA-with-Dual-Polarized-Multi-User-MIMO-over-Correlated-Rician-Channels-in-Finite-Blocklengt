"""Construct normalized MMSE combiners for the selected receiver architecture."""

import numpy as np


def functionMMSEDetector(hV, hH, BV, BH, M, K, Noise_power, scheme):
    """Solve the receiver covariance systems and normalize the resulting combiners."""
    # Assemble the signal, error, and noise covariance seen by the receiver.
    I = Noise_power * np.eye(2 * M, dtype=complex)
    errors = np.sum(BV + BH, axis=2)
    J = I + errors
    for k in range(K):
        J += np.outer(hV[:, k], hV[:, k].conj()) + np.outer(hH[:, k], hH[:, k].conj())
    # Use the covariance appropriate to no-SIC, local-SIC, or full-SIC decoding.
    if scheme in ("DP-noSIC", "SP-noSIC"):
        q1, q2 = np.linalg.solve(J, hV), np.linalg.solve(J, hH)
    elif scheme == "SP-fullSIC":
        q1 = np.empty((2 * M, K), complex); q2 = np.empty_like(q1); future = np.zeros_like(I)
        for k in range(K - 1, -1, -1):
            q1[:, k] = np.linalg.solve(I + errors + np.outer(hH[:, k], hH[:, k].conj()) + future, hV[:, k])
            q2[:, k] = np.linalg.solve(I + errors + future, hH[:, k])
            future += np.outer(hV[:, k], hV[:, k].conj()) + np.outer(hH[:, k], hH[:, k].conj())
    else:
        q1 = np.linalg.solve(J, hV); q2 = np.empty((2 * M, K), complex)
        for k in range(K):
            J2 = J - np.outer(hV[:, k], hV[:, k].conj()) - np.outer(hH[:, k], hH[:, k].conj())
            q2[:, k] = np.linalg.solve(J2, hH[:, k])
    # Normalize every combiner so that subsequent noise powers remain consistent.
    q1 /= np.linalg.norm(q1, axis=0); q2 /= np.linalg.norm(q2, axis=0)
    return q1, q2
