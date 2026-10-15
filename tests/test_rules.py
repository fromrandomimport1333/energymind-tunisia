import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from energymind.data.load import feature_columns, to_class  # noqa: E402
from energymind.detect import rules  # noqa: E402
from energymind.explain.cascade import decide  # noqa: E402

TH = {"s": 0.3, "g": 0.5, "phi": 0.05, "rho": 0.05, "drop": 0.3, "gap": 12.0}
BASE = dict(v_zero=False, s=0.0, n_aff=0, n_tot=9, g=0.0, phi=1.0, impossible=False,
            rho=1.0, cusum_alarm=False, drop=0.0, balance_alarm=False)


def row(**kw):
    d = dict(energy_kwh=0.5, voltage_v=230.0, current_a=4.6, power_factor=0.95,
             pv_export_kwh=0.0, feeder_voltage_v=231.0, meter_event_code="none")
    d.update(kw)
    return pd.DataFrame([d])


def test_no_labels_in_features():
    df = pd.DataFrame(columns=["energy_kwh", "gt_label", "gt_true_energy_kwh", "fault_weight", "meter_id"])
    assert feature_columns(df) == ["energy_kwh", "meter_id"]


def test_class_map():
    assert to_class("falsification") == "SUSPICIOUS_DATA"
    assert to_class("equipment_failure") == "GENUINE_EVENT"


def test_physics_ratio_honest_and_fdi():
    assert abs(rules.physics_ratio(row()).iloc[0] - 1) < 0.01
    assert rules.physics_inconsistent(row(energy_kwh=0.3)).iloc[0]          # fdi_scale-like


def test_physics_skipped_while_exporting():
    assert rules.physics_ratio(row(energy_kwh=0.0, pv_export_kwh=0.4)).isna().iloc[0]


def test_voltage_above_feeder():
    ts = pd.date_range("2025-01-01", periods=48, freq="30min")
    day = pd.concat([row(voltage_v=234.5)] * 48, ignore_index=True).assign(timestamp=ts, meter_id="M1")
    assert rules.voltage_above_feeder(day).iloc[-1]
    assert not rules.voltage_above_feeder(day.assign(voltage_v=229.0)).iloc[-1]


def test_outage_is_not_stuck_sensor():
    v = decide({**BASE, "v_zero": True, "s": 1.0, "n_aff": 9, "phi": 0.0}, TH)
    assert v.verdict == "GENUINE_EVENT" and v.rule == "R0"


def test_bypass_is_suspicious_not_appliance():
    v = decide({**BASE, "drop": 0.45, "balance_alarm": True}, TH)
    assert v.verdict == "SUSPICIOUS_DATA" and v.rule == "R5b"


