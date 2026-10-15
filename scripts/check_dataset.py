"""Print every dataset fact quoted in the proposal and the goal reports.

The detection results (Goal 1) come from docs/goal1_detection/n1_experiment.py; everything else
quoted about the data comes from this script.   Run: python scripts/check_dataset.py [--sample]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from energymind.data.load import NODE, load  # noqa: E402
from energymind.detect.rules import physics_ratio  # noqa: E402
from energymind.shared.calibration import calibration_summary  # noqa: E402

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)
FAMILY = {
    "partial_reduction_bypass": "physical bypass", "selective_bypass": "physical bypass",
    "meter_zeroing": "physical bypass", "fdi_scale": "data-only FDI", "fdi_evasion": "data-only FDI",
    "fdi_inflate": "data-only FDI", "fdi_replay": "data-only FDI", "coordinated_fdi": "data-only FDI",
    "bias_step": "sensor drift", "gain_drift": "sensor drift", "voltage_offset_drift": "sensor drift",
    "clock_shift": "sensor drift", "stuck_value": "sensor drift", "spike_gain": "corruption",
    "negative_value": "corruption", "saturation": "corruption", "unit_error_x1000": "corruption",
}
CLASS = {"grid_event": "GENUINE_EVENT", "equipment_failure": "GENUINE_EVENT", "sensor_fault": "METER_FAULT",
         "data_corruption": "METER_FAULT", "falsification": "SUSPICIOUS_DATA", "comm_loss": "NO_DATA"}


def title(t):
    print(f"\n=== {t}")


df = load(sample="--sample" in sys.argv, node=None)
events = df[df.gt_event_id != "none"].groupby("gt_event_id").agg(
    node=("node_id", "first"), label=("gt_label", "first"), sub=("gt_subtype", "first"),
    start=("timestamp", "min"), end=("timestamp", "max"), meters=("meter_id", "nunique"))
events["cls"] = events.label.map(CLASS)
events["family"] = events["sub"].map(FAMILY)

title("Dataset")
print(f"{len(df):,} rows x {df.shape[1]} columns | step "
      f"{df.timestamp.drop_duplicates().sort_values().diff().mode()[0]} | {df.timestamp.min()} .. {df.timestamp.max()}")
print(df.groupby("node_id").agg(meters=("meter_id", "nunique"), feeders=("feeder_id", "nunique"), rows=("meter_id", "size")))
print("label shares (%):", (df.gt_label.value_counts(normalize=True) * 100).round(2).to_dict())

n1 = df[df.node_id == NODE]
ev1 = events[events.node == NODE]
title(f"{NODE}: meters, feeders, PV, customer classes, meter models")
m1 = n1.drop_duplicates("meter_id")
print("meters per feeder:", m1.groupby("feeder_id").size().to_dict(), "| PV meters:", m1[m1.has_pv == 1].meter_id.tolist())
print(pd.crosstab(m1.feeder_id, m1.customer_class))
print("meters per model and node:\n", pd.crosstab(df.drop_duplicates("meter_id").meter_model, df.drop_duplicates("meter_id").node_id))

title(f"{NODE}: events per diagnosis class and per calendar split")
ev1 = ev1.assign(split=pd.cut(ev1.start, pd.to_datetime(["2024-12-31", "2025-09-01", "2025-11-01", "2026-01-01"]),
                              labels=["Jan-Aug", "Sep-Oct", "Nov-Dec"]))
print("events:", len(ev1), ev1.cls.value_counts().to_dict())
print(pd.crosstab([ev1.label, ev1["sub"]], ev1.split))

title(f"{NODE}: supply loss is recorded as zeros, communication loss as blanks")
for st in ["outage", "load_shedding", "cable_fault", "transformer_trip", "comm_dropout_burst"]:
    s = n1[n1.gt_subtype == st]
    print(f"{st:20s} rows {len(s):4d}  blank {s.energy_kwh.isna().mean():6.1%}  V=0 {(s.voltage_v == 0).mean():6.1%}")
print("supply-loss readings (V = 0, genuine events):",
      int(((n1.voltage_v == 0) & n1.gt_label.isin(["grid_event", "equipment_failure"])).sum()))
normal = n1[n1.gt_label == "normal"]
print(f"clean readings with E = 0: {(normal.energy_kwh == 0).mean():.2%}")

title(f"{NODE}: physics rule E/(V I PF dt) on clean readings, with and without the PV-export gate")
raw = normal.energy_kwh / (normal.voltage_v * normal.current_a * normal.power_factor * 0.5 / 1000)
off = (raw - 1).abs() > 0.05
print(f"ungated |ratio-1|>5%: non-PV homes {off[normal.has_pv == 0].mean():.2%}, PV homes {off[normal.has_pv == 1].mean():.2%}")
gated = physics_ratio(normal)
print(f"gated: {((gated - 1).abs() > 0.05).mean():.2%} of all clean readings")
print("median ratio per subtype:", n1.assign(r=physics_ratio(n1)).groupby("gt_subtype").r.median().round(2).dropna().to_dict())

title("Falsification events per node, attack families per node")
print(pd.crosstab(events[events.label == "falsification"]["sub"], events[events.label == "falsification"].node))
print(pd.crosstab(events.family, events.node))
cfdi = events[events["sub"] == "coordinated_fdi"]
print("coordinated_fdi:", cfdi[["node", "start", "end", "meters"]].to_string())

title("C2 calibration summary on clean readings (negative control)")
for node, g in df[df.gt_label == "normal"].groupby("node_id"):
    print(node, calibration_summary(g).round(4).to_dict())

title(f"{NODE}: reference responses (gt_expected_response) by label, readings")
print(pd.crosstab(n1.gt_label, n1.gt_expected_response))

title("Realism checks quoted in the proposal")
res = df[df.customer_class.str.startswith("res")]
annual = res.groupby(["node_id", "meter_id"]).gt_true_energy_kwh.sum().groupby("node_id").mean()
print("residential annual kWh per home (true energy):", annual.round(0).to_dict())
ac = res[res.has_ac == 1]
month = ac.timestamp.dt.month
ratio = ac[month.isin([7, 8])].gt_true_energy_kwh.mean() / ac[month.isin([1, 2])].gt_true_energy_kwh.mean()
print(f"AC homes, Jul-Aug / Jan-Feb consumption: {ratio:.2f}x")
live = df[df.gt_data_trustworthy.eq(1) & (df.voltage_v > 1) & (df.pv_export_kwh == 0) & (df.energy_kwh > 0.05)]
err = (live.energy_kwh / (live.voltage_v * live.current_a * live.power_factor * 0.5 / 1000) - 1).abs()
print(f"energy vs V*I*PF*dt on trustworthy readings: median error {err.median():.2%}, p95 {err.quantile(0.95):.2%}")
