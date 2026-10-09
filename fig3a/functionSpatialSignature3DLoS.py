"""Compute the line-of-sight array steering vector for a three-dimensional geometry."""

import numpy as np


def functionSpatialSignature3DLoS(U, varphi, theta, wavelength):
    """Return the phase response across the supplied antenna coordinates."""
    # Project the wave vector onto each antenna position to obtain its relative phase.
    k = -2 * np.pi / wavelength * np.array([
        np.cos(varphi) * np.cos(theta),
        np.sin(varphi) * np.cos(theta),
        np.sin(theta),
    ])
    return np.exp(1j * k @ U)
