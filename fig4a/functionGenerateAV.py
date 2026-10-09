"""Build the effective channel-gain and estimation-error terms used by power control."""

from types import SimpleNamespace
import numpy as np


def functionGenerateAV(q1, q2, h_est_1, h_est_2, Rerror_1, Rerror_2, K):
    """Project channel and error covariances onto the receive combiners."""
    AV = SimpleNamespace(name="auxiliary variables for power control")
    # Store squared effective channel gains for all stream and combiner pairs.
    AV.zeta1_V = 1e10 * abs(q1.conj().T @ h_est_1)**2
    AV.zeta1_H = 1e10 * abs(q1.conj().T @ h_est_2)**2
    AV.zeta2_V = 1e10 * abs(q2.conj().T @ h_est_1)**2
    AV.zeta2_H = 1e10 * abs(q2.conj().T @ h_est_2)**2
    AV.xi1_V = np.empty((K, K)); AV.xi1_H = np.empty((K, K))
    AV.xi2_V = np.empty((K, K)); AV.xi2_H = np.empty((K, K))
    # Project each estimation-error covariance onto each receive combiner.
    for k in range(K):
        for kp in range(K):
            AV.xi1_V[k, kp] = 1e10 * abs(q1[:, k].conj() @ Rerror_1[:, :, kp] @ q1[:, k])
            AV.xi1_H[k, kp] = 1e10 * abs(q1[:, k].conj() @ Rerror_2[:, :, kp] @ q1[:, k])
            AV.xi2_V[k, kp] = 1e10 * abs(q2[:, k].conj() @ Rerror_1[:, :, kp] @ q2[:, k])
            AV.xi2_H[k, kp] = 1e10 * abs(q2[:, k].conj() @ Rerror_2[:, :, kp] @ q2[:, k])
    return AV
