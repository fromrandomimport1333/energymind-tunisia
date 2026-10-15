"""Layer 2 - robust z-score on profile residuals and per-meter CUSUM (the main bias detector)."""
import numpy as np
import pandas as pd


def robust_z(x, med=None, mad=None):
    med = np.nanmedian(x) if med is None else med
    mad = np.nanmedian(np.abs(x - med)) if mad is None else mad
    return (x - med) / (1.4826 * mad + 1e-9)


def cusum(x, k=0.5, h=4.0):
    """Two-sided CUSUM on a standardized series. Returns (S_plus, S_minus, alarm)."""
    sp = np.zeros(len(x)); sm = np.zeros(len(x))
    for t in range(1, len(x)):
        v = 0.0 if np.isnan(x[t]) else x[t]
        sp[t] = max(0.0, sp[t - 1] + v - k)
        sm[t] = max(0.0, sm[t - 1] - v - k)
    return sp, sm, (sp > h) | (sm > h)


def neighbour_ratio(df):
    """Daily energy of each meter divided by the median of its feeder neighbours.

    A sensor bias or drift shows as a level shift in this ratio; run cusum() on its log.
    """
    daily = (df.set_index("timestamp").groupby(["feeder_id", "meter_id"]).energy_kwh
             .resample("1D").sum().reset_index())
    med = daily.groupby(["feeder_id", "timestamp"]).energy_kwh.transform("median")
    daily["ratio"] = daily.energy_kwh / med.replace(0, np.nan)
    return daily
