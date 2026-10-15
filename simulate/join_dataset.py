"""STAGE 3: join the outputs of inject_anomalies.py into the single analysis table.

One row per meter per half-hour: readings + meter details + weather/calendar + feeder context + labels.
Run:  python join_dataset.py --data ./energymind_tn_dataset/data --out energymind_full_joined.csv.gz
"""
import argparse
import os

import pandas as pd

METER_DETAILS = ["meter_id", "phase", "customer_class", "dwelling_type", "floor_area_m2", "occupants",
                 "distance_to_transformer_m", "lat", "lon", "has_ac", "ac_units", "ac_unit_kw", "has_pv", "pv_kwp",
                 "has_electric_heater", "has_electric_water_heater", "meter_install_year", "comm_tech", "meter_model",
                 "meter_accuracy_class", "annual_kwh_target", "fault_weight"]
WEATHER = ["timestamp", "node_id", "temp_c", "humidity_pct", "ghi_wm2", "is_weekend", "is_public_holiday", "is_ramadan"]
FEEDER = ["timestamp", "node_id", "feeder_id", "n_meters_total", "n_meters_reporting", "feeder_kwh_net", "feeder_voltage_v",
          "feeder_current_a", "feeder_thd_pct", "frequency_hz", "sum_meters_import_kwh", "sum_meters_export_kwh",
          "energy_balance_gap_kwh", "energy_balance_gap_pct", "gt_label", "gt_subtype", "gt_event_id",
          "gt_tech_loss_kwh", "gt_streetlight_kwh", "gt_metering_discrepancy_kwh", "gt_n_meters_falsified",
          "gt_n_meters_sensor_fault"]
ORDER = (["timestamp", "node_id", "feeder_id", "meter_id", "energy_kwh", "pv_export_kwh", "reactive_kvarh", "voltage_v",
          "voltage_min_v", "voltage_max_v", "current_a", "power_factor", "thd_v_pct", "meter_event_code"]
         + METER_DETAILS[1:] + WEATHER[2:] + FEEDER[3:14]
         + ["gt_label", "gt_subtype", "gt_event_id", "gt_data_trustworthy", "gt_expected_response",
            "gt_true_energy_kwh", "gt_true_voltage_v", "gt_feeder_label", "gt_feeder_subtype", "gt_feeder_event_id",
            "gt_tech_loss_kwh", "gt_streetlight_kwh", "gt_metering_discrepancy_kwh", "gt_n_meters_falsified",
            "gt_n_meters_sensor_fault"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join("energymind_tn_dataset", "data"))
    ap.add_argument("--out", default="energymind_full_joined.csv.gz")
    a = ap.parse_args()
    meters = pd.read_csv(os.path.join(a.data, "meters_metadata.csv"))[METER_DETAILS]
    weather = pd.read_csv(os.path.join(a.data, "weather_calendar.csv.gz"), parse_dates=["timestamp"])[WEATHER]
    feeder = pd.read_csv(os.path.join(a.data, "feeder_readings.csv.gz"), parse_dates=["timestamp"])[FEEDER]
    feeder = feeder.rename(columns={"gt_label": "gt_feeder_label", "gt_subtype": "gt_feeder_subtype",
                                    "gt_event_id": "gt_feeder_event_id"})
    parts = []
    for node in ["N1", "N2", "N3"]:
        r = pd.read_csv(os.path.join(a.data, f"{node}_meter_readings.csv.gz"), parse_dates=["timestamp"], low_memory=False)
        r = (r.merge(meters, on="meter_id", how="left")
              .merge(weather, on=["timestamp", "node_id"], how="left")
              .merge(feeder, on=["timestamp", "node_id", "feeder_id"], how="left"))
        parts.append(r[ORDER])
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(a.out, index=False)
    print(f"{len(df):,} rows x {df.shape[1]} columns -> {a.out}")


if __name__ == "__main__":
    main()
