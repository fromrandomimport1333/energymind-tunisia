"""C3 aggregation rules (Goal 4 Eq. 5-10): clip + noise, FedAvg, coordinate-wise median."""
import numpy as np


def protect(delta, C, sigma, K, rng):
    d = delta * min(1.0, C / (np.linalg.norm(delta) + 1e-12))
    return d + rng.normal(0, sigma * C / np.sqrt(K), size=d.shape)


def fedavg(updates):
    return np.mean(updates, axis=0)


def coord_median(updates):
    return np.median(updates, axis=0)
