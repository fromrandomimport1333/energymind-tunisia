"""Load the EnergyMind dataset and keep labels out of the model inputs.

The column groups follow docs/data_card.md. Every goal imports from here, so the
rules about what a model may see live in one place.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT / "data" / "energymind_full_joined.csv.gz"
SAMPLE_PATH = ROOT / "data" / "sample" / "energymind_sample.csv.gz"

NODE = "N1"   # Phase 1 prototype node: coastal suburb, 30 meters on 3 feeders

IDS = ["timestamp", "node_id", "feeder_id", "meter_id"]
SIMULATOR_COLS = ["annual_kwh_target", "fault_weight"]           # simulator settings, never inputs
ORACLE_COLS = ["gt_true_energy_kwh", "gt_true_voltage_v", "gt_metering_discrepancy_kwh",
               "gt_tech_loss_kwh", "gt_streetlight_kwh"]          # error analysis only
LABEL_COLS = ["gt_label", "gt_subtype", "gt_event_id", "gt_data_trustworthy",
              "gt_expected_response", "gt_feeder_label", "gt_feeder_subtype",
              "gt_feeder_event_id", "gt_n_meters_falsified", "gt_n_meters_sensor_fault"]

# dataset label -> the 4 diagnosis classes shared by Goals 2 and 3
CLASS_MAP = {
    "normal": "NORMAL",
    "grid_event": "GENUINE_EVENT",
    "equipment_failure": "GENUINE_EVENT",
    "sensor_fault": "METER_FAULT",
    "data_corruption": "METER_FAULT",
    "falsification": "SUSPICIOUS_DATA",
    "comm_loss": "NO_DATA",
}


def load(path=None, sample=False, usecols=None, node=NODE):
    """Read the joined table, restricted to one node (node=None keeps all three).

    sample=True reads the small committed slice.
    """
    path = Path(path) if path else (SAMPLE_PATH if sample else DATA_PATH)
    if usecols is not None:
        needed = ["timestamp"] + (["node_id"] if node is not None else [])
        usecols = list(dict.fromkeys(list(usecols) + needed))
    df = pd.read_csv(path, parse_dates=["timestamp"], usecols=usecols, low_memory=False)
    return df[df.node_id == node].reset_index(drop=True) if node is not None else df


def feature_columns(df):
    """Columns a model may use: everything except labels, oracle values and simulator settings."""
    banned = set(LABEL_COLS + ORACLE_COLS + SIMULATOR_COLS)
    return [c for c in df.columns if c not in banned and not c.startswith("gt_")]


def to_class(gt_label):
    """Map gt_label (scalar or Series) to the 4-class diagnosis (+ NO_DATA)."""
    if isinstance(gt_label, pd.Series):
        return gt_label.map(CLASS_MAP)
    return CLASS_MAP[gt_label]


def event_folds(df, k=5, seed=2025):
    """Fold id per row, grouped by gt_event_id so no event is split across folds.

    Normal rows ("none") are assigned in contiguous weekly blocks per meter to limit
    leakage between neighbouring readings. Use this instead of the Jan-Aug / Sep-Oct /
    Nov-Dec time split for calibration: most rare subtypes have no event after August.
    """
    rng = np.random.default_rng(seed)
    fold = pd.Series(-1, index=df.index, dtype=int)

    events = rng.permutation(np.asarray(df.loc[df.gt_event_id != "none", "gt_event_id"].unique(), dtype=object))
    # stratify by subtype so each fold gets its share of rare types
    first_sub = (df[df.gt_event_id != "none"].groupby("gt_event_id").gt_subtype.first()
                 .reindex(events))
    ev_fold = {}
    for _, ids in first_sub.groupby(first_sub, sort=False):
        for i, eid in enumerate(ids.index):
            ev_fold[eid] = (i + rng.integers(k)) % k
    is_ev = df.gt_event_id != "none"
    fold[is_ev] = df.loc[is_ev, "gt_event_id"].map(ev_fold)

    week = df.loc[~is_ev, "timestamp"].dt.isocalendar().week.astype(int)
    key = (week.astype(str) + "|" + df.loc[~is_ev, "meter_id"]).to_numpy(dtype=object)
    fold[~is_ev] = (pd.util.hash_array(key) % k).astype(int)   # deterministic, unlike hash()
    return fold
