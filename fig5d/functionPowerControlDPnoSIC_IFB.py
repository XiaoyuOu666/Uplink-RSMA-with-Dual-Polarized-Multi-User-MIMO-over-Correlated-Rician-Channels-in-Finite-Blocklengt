"""Optimize the two stream powers for the no-SIC infinite-blocklength model."""

import cvxpy as cp
import numpy as np
from functionComputeSEDPnoSIC_IFB import functionComputeSEDPnoSIC_IFB
from functionNormalizePowerControl import functionNormalizePowerControl


def functionPowerControlDPnoSIC_IFB(AV, K, Pk, Noise_power):
    """Optimize no-SIC powers using the infinite-blocklength transformed objective."""
    AV_original = AV
    AV, noise1, noise2, scale1, scale2 = functionNormalizePowerControl(AV, Noise_power)
    SE_iter = np.zeros(50); p1 = np.ones(K) * Pk / 2; p2 = np.ones(K) * Pk / 2
    # Alternate closed-form auxiliary updates with convex power optimization.
    for i in range(50):
        result = functionComputeSEDPnoSIC_IFB(AV_original, p1, p2, Noise_power)
        SE_iter[i] = result.SE
        l1, l2 = result.gamma1, result.gamma2
        M1, M2, N1, N2 = result.M1_mathcal, result.M2_mathcal, result.N1_mathcal, result.N2_mathcal
        M1, N1 = M1 / scale1, N1 / scale1
        M2, N2 = M2 / scale2, N2 / scale2
        z1 = np.sqrt((1 + l1) * M1) / (M1 + N1); z2 = np.sqrt((1 + l2) * M2) / (M2 + N2)
        x1, x2 = cp.Variable(K, nonneg=True), cp.Variable(K, nonneg=True)
        m1 = cp.multiply(np.diag(AV.zeta1_V), x1); m2 = cp.multiply(np.diag(AV.zeta2_H), x2)
        n1 = AV.zeta1_V @ x1 - m1 + AV.zeta1_H @ x2 + AV.xi1_V @ x1 + AV.xi1_H @ x2 + noise1
        n2 = AV.zeta2_V @ x1 + AV.zeta2_H @ x2 - m2 + AV.xi2_V @ x1 + AV.xi2_H @ x2 + noise2
        q1 = cp.sum(cp.multiply(2 * z1, cp.sqrt(cp.multiply(1 + l1, m1))) + cp.multiply(2 * z2, cp.sqrt(cp.multiply(1 + l2, m2))))
        q2 = cp.sum(cp.multiply(z1**2, m1 + n1) + cp.multiply(z2**2, m2 + n2))
        # Solve the concave surrogate over the per-UE total-power constraints.
        problem = cp.Problem(cp.Maximize(q1 - q2), [x1 + x2 <= Pk])
        problem.solve(solver=cp.MOSEK)
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE): break
        p1, p2 = x1.value, x2.value
        # Stop when the original sum-SE improvement meets the convergence tolerance.
        if i >= 1 and abs(SE_iter[i] - SE_iter[i - 1]) <= 1e-3: break
    return np.max(SE_iter), SE_iter, p1, p2
