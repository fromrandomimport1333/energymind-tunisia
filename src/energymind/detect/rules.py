"""Layer 1 - physics and domain rules. Each function returns a boolean Series (True = flag).

These are the rules the Goal 1 report and the compatibility review agree on. They are
deliberately simple and auditable; thresholds live in configs/thresholds.yaml.
"""
import numpy as np
import pandas as pd

DT_H = 0.5  # 30-min interval


def impossible_value(df, e_max=50.0):
    """Negative energy, saturation (99.999) or unit error (x1000) - data corruption."""
    e = df.energy_kwh
    return (e < 0) | (e >= e_max) | np.isclose(e, 99.999)


def physics_ratio(df):
    """E / (V*I*PF*dt). ~1 for honest readings. NaN when undefined (no supply, exporting, tiny load).

    Catches data-only falsification and corruption. Blind to sensor bias and physical bypass,
    which scale energy and current together.
    """
    expected = df.voltage_v * df.current_a * df.power_factor * DT_H / 1000
    ok = (df.voltage_v > 1) & (df.pv_export_kwh.fillna(0) == 0) & (expected > 0.02)
    return (df.energy_kwh / expected).where(ok)


def physics_inconsistent(df, tol=0.05):
    r = physics_ratio(df)
    return (r - 1).abs() > tol


def voltage_above_feeder(df, margin_v=1.5, window=48):
    """Meter voltage persistently above the feeder head -> voltage sensor offset.

    Uses the 24-h rolling median of (V - feeder V) on non-exporting rows: single readings
    and PV export (which raises local voltage) are too noisy. On the sample week this flags
    68/91 voltage_offset_drift rows vs 24 of ~17,500 normal rows.
    """
    d = df.sort_values(["meter_id", "timestamp"])
    dv = (d.voltage_v - d.feeder_voltage_v).where(d.pv_export_kwh.fillna(0) == 0)
    med = dv.groupby(d.meter_id).transform(
        lambda s: s.rolling(window, min_periods=window // 2).median())
    return (med > margin_v).reindex(df.index)


def supply_lost(df):
    """V = 0 or a power_fail log: a genuine supply loss, not a stuck sensor."""
    return (df.voltage_v.fillna(1) <= 1) | (df.meter_event_code == "power_fail")


def flat_line(df, min_steps=6):
    """Same energy for >= min_steps intervals while supply is on and the meter is not exporting."""
    live = (df.voltage_v > 1) & (df.pv_export_kwh.fillna(0) == 0)
    d = df.sort_values(["meter_id", "timestamp"])
    same = d.groupby("meter_id").energy_kwh.diff().eq(0)   # first row of each meter is False
    run = same.groupby((~same).cumsum()).cumsum()
    return (run >= min_steps - 1).reindex(df.index) & live


def balance_gap_high(df, gap_pct=12.0):
    """Feeder loses much more than the ~5% normal: large under-reporting on the feeder."""
    return df.energy_balance_gap_pct > gap_pct
