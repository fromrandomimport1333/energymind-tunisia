"""C2 calibration insight: median log E/(V*I*PF*dt) per meter model, released only if >= k meters."""
import numpy as np

from energymind.detect.rules import physics_ratio


def calibration_summary(df, k_min=3):
    d = df.assign(r=np.log(physics_ratio(df))).dropna(subset=["r"])
    per_meter = d.groupby(["meter_model", "meter_id"]).r.median()
    out = per_meter.groupby("meter_model").agg(["median", "size"])
    return out[out["size"] >= k_min]["median"]   # suppressed when fewer than k_min meters
