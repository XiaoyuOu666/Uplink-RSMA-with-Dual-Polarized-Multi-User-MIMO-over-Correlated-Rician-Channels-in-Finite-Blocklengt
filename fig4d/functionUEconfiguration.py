"""Generate UE positions, antenna offsets, distances, and angular parameters."""

from types import SimpleNamespace
import numpy as np


def functionUEconfiguration(K, InFConfig, wavelength, r_scatter, BS_height):
    """Sample UE geometry and derive the distances and angular spreads used by the channel model."""
    UE = SimpleNamespace(name="UE configuration")
    # Sample UE centers and place the two SP antennas symmetrically around them.
    UE.positions = np.column_stack(((np.random.rand(K) - .5) * InFConfig.InF_width,
                                    (np.random.rand(K) - .5) * InFConfig.InF_length,
                                    np.random.rand(K) * 3))
    angle = 2 * np.pi * np.random.rand(K)
    offset = np.column_stack((wavelength / 4 * np.cos(angle),
                              wavelength / 4 * np.sin(angle), np.zeros(K)))
    UE.antenna1_positions = UE.positions + offset
    UE.antenna2_positions = UE.positions - offset
    # Derive distances, arrival angles, and angular spreads for every antenna position.
    for suffix, positions in (("", UE.positions), ("_antenna1", UE.antenna1_positions),
                              ("_antenna2", UE.antenna2_positions)):
        d2D = np.hypot(positions[:, 0], positions[:, 1])
        setattr(UE, "d2D" + suffix, d2D)
        setattr(UE, "d3D" + suffix, np.sqrt(d2D**2 + (positions[:, 2] - BS_height)**2))
        phi = np.arctan2(positions[:, 1], positions[:, 0])
        setattr(UE, "phi" + suffix, (phi + np.pi) % (2 * np.pi) - np.pi if not suffix else phi % (2 * np.pi))
        setattr(UE, "dphi" + suffix, np.arctan(r_scatter / d2D))
        theta_max = np.arctan((BS_height - positions[:, 2]) / np.maximum(d2D - r_scatter, 0))
        theta_min = np.arctan((BS_height - positions[:, 2]) / (d2D + r_scatter))
        setattr(UE, "theta" + suffix, (theta_min + theta_max) / 2)
        setattr(UE, "dtheta" + suffix, (theta_max - theta_min) / 2)
    return UE
