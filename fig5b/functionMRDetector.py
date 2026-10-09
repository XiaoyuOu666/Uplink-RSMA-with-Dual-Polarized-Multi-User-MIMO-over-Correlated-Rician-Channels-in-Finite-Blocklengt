"""Construct normalized matched-filter receive combiners."""

import numpy as np


def functionMRDetector(h_est_V, h_est_H, K):
    """Normalize each estimated channel to obtain an MR combiner."""
    # MR uses each channel estimate directly and only enforces unit norm.
    return (h_est_V / np.linalg.norm(h_est_V, axis=0),
            h_est_H / np.linalg.norm(h_est_H, axis=0))
