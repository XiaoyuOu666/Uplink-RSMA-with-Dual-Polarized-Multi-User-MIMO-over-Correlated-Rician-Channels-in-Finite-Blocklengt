from types import SimpleNamespace
import numpy as np


def functionNormalizePowerControl(AV, Noise_power):
    gain1 = np.diag(AV.zeta1_V).copy()
    gain2 = np.diag(AV.zeta2_H).copy()
    scale1 = np.where(gain1 > 0, gain1, 1.0)
    scale2 = np.where(gain2 > 0, gain2, 1.0)
    values = vars(AV).copy()
    for name in ("zeta1_V", "zeta1_H", "xi1_V", "xi1_H"):
        values[name] = values[name] / scale1[:, None]
    for name in ("zeta2_V", "zeta2_H", "xi2_V", "xi2_H"):
        values[name] = values[name] / scale2[:, None]
    return SimpleNamespace(**values), Noise_power / scale1, Noise_power / scale2, scale1, scale2
