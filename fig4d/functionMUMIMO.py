"""Evaluate the conventional single-stream MU-MIMO benchmark."""

import numpy as np
from scipy.special import ndtri


def functionMUMIMO(h1, h2, Rerror, Noise_power, Pk, Eth, Nth, combinerIndex):
    """Combine each UE antenna pair into one stream and evaluate its achievable sum SE."""
    Nr, K = h1.shape
    # Map the two SP antenna signals of each UE to one normalized data stream.
    v = np.ones((2, K)) / np.sqrt(2)
    h = np.empty((Nr, K), complex); B = np.empty((Nr, Nr, K), complex)
    for k in range(K):
        h[:, k] = np.column_stack((h1[:, k], h2[:, k])) @ v[:, k]
        R = Rerror[:, :, k]
        B11, B12, B21, B22 = R[:Nr, :Nr], R[:Nr, Nr:], R[Nr:, :Nr], R[Nr:, Nr:]
        v1, v2 = v[:, k]
        Bk = v1 * v1.conjugate() * B11 + v1 * v2.conjugate() * B12 + v2 * v1.conjugate() * B21 + v2 * v2.conjugate() * B22
        B[:, :, k] = (Bk + Bk.conj().T) / 2
    # Select MMSE, RZF, or MR combining under the same channel realization.
    if combinerIndex == 1:
        J = Noise_power * np.eye(Nr, dtype=complex)
        for k in range(K): J += np.outer(h[:, k], h[:, k].conj()) + B[:, :, k]
        J = (J + J.conj().T) / 2
        q = np.linalg.solve(J, h)
    elif combinerIndex == 2:
        q = h @ np.linalg.pinv(h.conj().T @ h + Noise_power * np.eye(K))
    else:
        q = h.copy()
    q /= np.linalg.norm(q, axis=0)
    B_total = Pk * np.sum(B, axis=2)
    # Evaluate the desired power, inter-user interference, and estimation error.
    gamma = np.empty(K)
    for k in range(K):
        gain = abs(q[:, k].conj() @ h)**2
        desired = Pk * gain[k]
        interference = Pk * (np.sum(gain) - gain[k])
        gamma[k] = desired / (interference + np.real(q[:, k].conj() @ B_total @ q[:, k]) + Noise_power * np.real(q[:, k].conj() @ q[:, k]))
    f = lambda x: np.log2(1 + x) - np.sqrt(2 * x / ((1 + x) * Nth)) * ndtri(1 - Eth) / np.log(2)
    return np.sum(f(gamma)), gamma, q, v, h, B
