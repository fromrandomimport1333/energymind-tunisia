#!/usr/bin/env python3
"""
inject_anomalies.py  -  STAGE 2: events, faults, falsification and labels
========================================================================
EnergyMind Tunisia / IEEE TSYP14.  Needs `generate_baseline.py` in the same folder.

This file takes the healthy network from STAGE 1 and, at the right physical moment, injects:

  PASS A  (before voltages are computed)  genuine events that change SUPPLY or LOAD:
          outage, load shedding, cable fault, demand surge, transformer trip (+ cold-load pick-up)
  PASS B  (after voltages are computed)   genuine events that change VOLTAGE / FREQUENCY:
          transformer degradation, voltage sag, voltage swell, frequency excursion
  PASS C  (after the meters have measured) DATA PROBLEMS on the reported channels:
          sensor faults   gain_drift, voltage_offset_drift, bias_step, clock_shift, stuck_value
          corruption      spike_gain / negative_value / saturation, unit_error_x1000
          falsification   partial_reduction_bypass, selective_bypass, meter_zeroing (physical),
                          fdi_scale, fdi_replay, fdi_evasion, fdi_inflate, coordinated_fdi (data-only)
          comm loss       dropout bursts + minor gaps

Every injection (1) edits the data, (2) writes an event record (what, where, when, parameters, recipe),
(3) labels the affected rows (`gt_*` columns).  Rows are labelled only where the anomaly is realised.

Run:
    python inject_anomalies.py --out ./energymind_tn_dataset --seed 2025 --contamination 1.0

`--contamination` multiplies the number of data-quality / falsification events (genuine events are unaffected).
"""
import argparse, json, os
import numpy as np
import pandas as pd

import generate_baseline as gb
from generate_baseline import (T, SPD, NDAY, ts, hod, doy, dow, is_wkend, is_hol, is_ram, ar1, NODES, make_heat_episodes)

SEED, CONT = 2025, 1.0          # overwritten from the command line in __main__

# ----------------------------------------------------------------------------- how many events of each type per node
EVENT_COUNTS = {
    "N1": dict(
        grid=dict(outage_f=5, outage_n=2, shed=8, sag=9, swell=4, surge=6, cable=2, xfmr=1, freq=3),
        dq=dict(partial=1, selective=1, zeroing=1, fdi_scale=1, replay=1, evasion=1, inflate=1, coord=1, drift_gain=2, drift_volt=1, bias=1, stuck=3, spike=25, unit=3, clock=1, comm=20)),
    "N2": dict(
        grid=dict(outage_f=9, outage_n=2, shed=10, sag=8, swell=5, surge=5, cable=3, xfmr=1, freq=3),
        dq=dict(partial=1, selective=0, zeroing=1, fdi_scale=1, replay=0, evasion=1, inflate=0, coord=0, drift_gain=1, drift_volt=1, bias=1, stuck=2, spike=12, unit=2, clock=1, comm=14)),
    "N3": dict(
        grid=dict(outage_f=7, outage_n=2, shed=9, sag=8, swell=4, surge=6, cable=2, xfmr=1, freq=3),
        dq=dict(partial=1, selective=1, zeroing=0, fdi_scale=0, replay=1, evasion=0, inflate=1, coord=1, drift_gain=1, drift_volt=0, bias=1, stuck=2, spike=12, unit=2, clock=0, comm=14)),
}


# ----------------------------------------------------------------------------- label vocab
LABELS = ["normal", "sensor_fault", "data_corruption", "falsification", "comm_loss", "grid_event", "equipment_failure"]
LCODE = {l: i for i, l in enumerate(LABELS)}          # code order == overwrite priority
RESP = ["none", "auto_log", "auto_correct", "notify_operator", "human_decision"]
RCODE = {r: i for i, r in enumerate(RESP)}
SUBS = ["none"]
EVCODES = ["none", "power_fail", "power_restore", "voltage_sag", "voltage_swell", "cover_open", "magnetic_tamper"]
ECODE = {c: i for i, c in enumerate(EVCODES)}


def subcode(s):
    if s not in SUBS:
        SUBS.append(s)
    return SUBS.index(s)




# ----------------------------------------------------------------------------- injection for one node
def inject_node(nd, ni, heat):
    S = gb.build_baseline(nd, ni, heat, SEED)                   # STAGE 1: weather, meters, load, PV (healthy)
    rng, P, wx, md, M = S["rng"], S["P"], S["wx"], S["md"], S["M"]
    load, ac, pv = S["load"], S["ac"], S["pv"]
    feeders, fcols, dist, mid = S["feeders"], S["fcols"], S["dist"], S["mid"]

    evs = []
    lab_cls = np.zeros((T, M), np.uint8); lab_sub = np.zeros((T, M), np.uint16)
    lab_evt = np.zeros((T, M), np.int32); lab_rsp = np.zeros((T, M), np.uint8)
    supply = np.ones((T, M)); lmult = np.ones((T, M))
    mevent = np.zeros((T, M), np.uint8)
    freq_lab = np.zeros(T, np.int32)             # node-level frequency events -> event index
    feeder_evt = {f: np.zeros(T, np.int32) for f in feeders}   # feeder-level physical event index (priority by overwrite)

    def new_event(label, subtype, scope, feeder, cols, s, e, params, method, resp):
        evs.append(dict(event_id=f"{nd}-EV{len(evs) + 1:03d}", node_id=nd, label=label, subtype=subtype, scope=scope,
                        feeder_id=feeder, cols=list(map(int, cols)), s=int(s), e=int(e), params=params, method=method,
                        expected_response=resp))
        return len(evs)          # 1-based index

    def mark(evi, s, e, cols, mask=None, sub=None, resp=None):
        ev = evs[evi - 1]
        code = LCODE[ev["label"]]
        cols = np.asarray(cols, int)
        ix = np.ix_(np.arange(s, e), cols)
        m = np.ones((e - s, len(cols)), bool) if mask is None else np.broadcast_to(mask, (e - s, len(cols)))
        cur = lab_cls[ix]
        sel = m & (code >= cur)
        sc = subcode(sub or ev["subtype"]); rc = RCODE[resp or ev["expected_response"]]
        for arr, val in ((lab_cls, code), (lab_sub, sc), (lab_evt, evi), (lab_rsp, rc)):
            a = arr[ix]; a[sel] = val; arr[ix] = a

    def rstart(length, lo_day=0, hi_day=None, hours=None):
        hi = (NDAY if hi_day is None else min(hi_day, NDAY)) * SPD - length
        lo = lo_day * SPD
        if hi <= lo:
            return max(0, T - length - 1)
        s = int(rng.integers(lo, hi))
        if hours is not None:                    # snap to a random half-hour inside [h0,h1)
            d = s // SPD
            s = d * SPD + int(rng.integers(int(hours[0] * 2), int(hours[1] * 2)))
            s = min(s, T - length - 1)
        return s

    G_ = EVENT_COUNTS[nd]["grid"]
    # ===== PASS A: events that change supply / load (genuine physical phenomena) =====
    def do_outage(label, subtype, scope, feeder, cols, s, L, method, resp, params=None):
        e = s + L
        evi = new_event(label, subtype, scope, feeder, cols, s, e, params or {}, method, resp)
        supply[s:e, cols] = 0
        pk = 2 if L >= 2 else 1                  # cold-load pick-up after restoration
        for k in range(pk):
            if e + k < T:
                lmult[e + k, cols] *= (1.6 if k == 0 else 1.25) if L >= 2 else 1.25
        mark(evi, s, e, cols)
        if e < T:
            mark(evi, e, min(e + pk, T), cols, sub="cold_load_pickup", resp="auto_log")
            mevent[e, cols] = np.where(rng.random(len(cols)) < 0.95, ECODE["power_restore"], mevent[e, cols])
        mevent[s, cols] = np.where(rng.random(len(cols)) < 0.9, ECODE["power_fail"], mevent[s, cols])
        return evi

    for _ in range(G_["outage_f"]):
        f = feeders[int(rng.integers(len(feeders)))]; L = int(rng.integers(1, 7)); s = rstart(L)
        do_outage("grid_event", "outage", "feeder", f, fcols[f], s, L, "Feeder-wide supply interruption: supply=0 for all meters on feeder, then cold-load pick-up (x1.6 then x1.25).", "notify_operator")
    for _ in range(G_["outage_n"]):
        L = int(rng.integers(1, 5)); s = rstart(L)
        do_outage("grid_event", "outage", "node", "", np.arange(M), s, L, "Node-wide supply interruption (MV side).", "notify_operator")
    for _ in range(G_["shed"]):                  # STEG load management in the Jul-Aug peak
        f = feeders[int(rng.integers(len(feeders)))]; L = int(rng.integers(3, 5))
        hrs = (13, 16) if rng.random() < 0.5 else (20, 22)
        s = rstart(L, 182, 243, hrs)
        do_outage("grid_event", "load_shedding", "feeder", f, fcols[f], s, L, "Scheduled rolling cut on one feeder (1.5-2 h) during July-August peak demand.", "auto_log")
    for _ in range(G_["cable"]):                 # natural equipment failure: part of a feeder is lost
        f = feeders[int(rng.integers(len(feeders)))]
        cols = fcols[f]; dcut = np.quantile(dist[cols], rng.uniform(0.25, 0.7))
        down = cols[dist[cols] > dcut]; L = int(rng.integers(2, 11)); s = rstart(L)
        do_outage("equipment_failure", "cable_fault", "feeder_partial", f, down, s, L,
                  "Cable fault: meters downstream of a cut point lose supply; upstream meters stay energised.", "human_decision",
                  {"cut_point_m": round(float(dcut))})
    for _ in range(G_["surge"]):                 # genuine demand surge (wedding season/heat), evening
        f = feeders[int(rng.integers(len(feeders)))]; L = int(rng.integers(3, 9)); s = rstart(L, 150, 270, (18, 22))
        m_ = float(rng.uniform(1.35, 1.8)); cols = fcols[f]
        evi = new_event("grid_event", "demand_surge", "feeder", f, cols, s, s + L, {"load_multiplier": round(m_, 2)},
                        "Genuine local demand surge: all meters on feeder load x(1.35-1.8) for 1.5-4 h.", "auto_log")
        lmult[s:s + L, cols] *= m_
        mark(evi, s, s + L, cols)
    # transformer degradation ends with a trip (natural equipment failure)
    xf = []
    for _ in range(G_["xfmr"]):
        Ld = int(rng.integers(3, 6)) * SPD; s = rstart(Ld + 6, 20)
        xf.append((s, s + Ld))
        Lt = int(rng.integers(1, 4))
        do_outage("equipment_failure", "transformer_trip", "node", "", np.arange(M), s + Ld, Lt,
                  "Distribution transformer trip at the end of a degradation episode.", "human_decision")


    # ===== STAGE 1 again: loads actually drawn after the supply / load events of PASS A =====
    EL = gb.apply_supply(S, supply, lmult)
    sup_b, load_e, ac_e, pv_e = EL["sup_b"], EL["load_e"], EL["ac_e"], EL["pv_e"]
    net, imp_kw, exp_kw = EL["net"], EL["imp_kw"], EL["exp_kw"]
    vnl_node, loss_mult, thd_extra = gb.base_voltage_profile(S)

    # ----- PASS B (part 1): transformer degradation changes the busbar voltage, THD and losses -----
    for s, e in xf:                                             # transformer degradation signature
        ramp = np.linspace(0, 1, e - s) ** 1.5
        dv = rng.uniform(3, 8)
        vnl_node[s:e] -= dv * ramp; thd_extra[s:e] += 1.6 * ramp; loss_mult[s:e] = 1 + 0.8 * ramp
        evi = new_event("equipment_failure", "transformer_degradation", "node", "", np.arange(M), s, e,
                        {"voltage_drop_v": round(float(dv), 1)}, "Slow degradation: busbar voltage ramps down, THD up, technical losses x1.8, precedes a trip.", "human_decision")
        mark(evi, s, e, np.arange(M), mask=(ramp > 0.3)[:, None])
        for f in feeders:
            feeder_evt[f][s:e] = evi


    # ===== STAGE 1 again: voltage at each meter =====
    V, Pf, Pref, vf_bus, vmin_x, vmax_x, thd_x = gb.feeder_voltages(S, net, vnl_node)
    freq = gb.grid_frequency(S)

    # ===== PASS B (part 2): voltage / frequency events =====
    for _ in range(G_["sag"]):
        scope_node = rng.random() < 0.3
        f = "" if scope_node else feeders[int(rng.integers(len(feeders)))]
        cols = np.arange(M) if scope_node else fcols[f]; L = int(rng.integers(1, 3)); s = rstart(L)
        dv = float(rng.uniform(15, 45)); ex = float(rng.uniform(40, 110))
        evi = new_event("grid_event", "voltage_sag", "node" if scope_node else "feeder", f, cols, s, s + L,
                        {"mean_voltage_drop_v": round(dv, 1), "min_voltage_extra_dip_v": round(ex, 1)},
                        "Upstream fault ride-through: 30-min mean voltage -15..-45 V and Vmin dips a further 40-110 V on all affected meters.", "auto_log")
        V[s:s + L, cols] -= dv; vmin_x[s:s + L, cols] += ex; thd_x[s:s + L, cols] += 0.6
        mark(evi, s, s + L, cols, mask=(supply[s:s + L][:, cols] > 0))
        mevent[s, cols] = np.where((rng.random(len(cols)) < 0.85) & (mevent[s, cols] == 0), ECODE["voltage_sag"], mevent[s, cols])
        for ff in (feeders if scope_node else [f]):
            feeder_evt[ff][s:s + L] = evi
    for _ in range(G_["swell"]):
        scope_node = rng.random() < 0.3
        f = "" if scope_node else feeders[int(rng.integers(len(feeders)))]
        cols = np.arange(M) if scope_node else fcols[f]; L = int(rng.integers(1, 4)); s = rstart(L)
        dv = float(rng.uniform(8, 22)); ex = float(rng.uniform(10, 30))
        evi = new_event("grid_event", "voltage_swell", "node" if scope_node else "feeder", f, cols, s, s + L,
                        {"mean_voltage_rise_v": round(dv, 1), "max_voltage_extra_v": round(ex, 1)},
                        "Capacitor switching / load rejection: 30-min mean voltage +8..+22 V, Vmax +10..30 V further.", "notify_operator")
        V[s:s + L, cols] += dv; vmax_x[s:s + L, cols] += ex; thd_x[s:s + L, cols] += 0.4
        mark(evi, s, s + L, cols, mask=(supply[s:s + L][:, cols] > 0))
        mevent[s, cols] = np.where((rng.random(len(cols)) < 0.85) & (mevent[s, cols] == 0), ECODE["voltage_swell"], mevent[s, cols])
        for ff in (feeders if scope_node else [f]):
            feeder_evt[ff][s:s + L] = evi
    for _ in range(G_["freq"]):
        L = int(rng.integers(1, 4)); s = rstart(L); dfq = float(rng.uniform(0.25, 0.55))
        evi = new_event("grid_event", "frequency_excursion", "node", "", [], s, s + L, {"frequency_dip_hz": round(dfq, 2)},
                        "Generation-trip frequency dip (49.45-49.75 Hz average); visible only in feeder/node table.", "auto_log")
        freq[s:s + L] -= dfq; freq_lab[s:s + L] = evi

    # ===== STAGE 1 again: true electrical quantities and what the meters report =====
    D = gb.derive_electrical(S, EL, V, thd_x, thd_extra)
    V, E_imp, E_exp = D["V"], D["E_imp"], D["E_exp"]
    R = gb.measure_reported(S, D, vmin_x, vmax_x)
    ALL = list(R.keys())

    # ===== PASS C: data-quality problems & falsification (meter level) =====
    DQ = {k: int(round(v * CONT)) if v > 0 else 0 for k, v in EVENT_COUNTS[nd]["dq"].items()}
    for k, v in EVENT_COUNTS[nd]["dq"].items():                 # keep at least one of each when requested at CONT>=0.5
        if v > 0 and CONT >= 0.5 and DQ[k] == 0:
            DQ[k] = 1
    busy = set()
    wfault = md.fault_weight.values / md.fault_weight.values.sum()
    wcons = md.annual_kwh_target.values / md.annual_kwh_target.sum()

    def pick(weights, k=1, same_feeder=False):
        free = [i for i in range(M) if i not in busy]
        if len(free) < k:
            return None
        w = weights[free] / weights[free].sum()
        if same_feeder:
            for _ in range(30):
                j0 = int(rng.choice(free, p=w)); fe = md.feeder_id.values[j0]
                cand = [i for i in free if md.feeder_id.values[i] == fe]
                if len(cand) >= k:
                    return list(rng.choice(cand, k, replace=False))
            return None
        return list(rng.choice(free, k, replace=False, p=w))

    def scale_ch(j, s, e, fac, chans=("E", "X", "Q", "I")):
        fac = np.asarray(fac).reshape(-1) if np.ndim(fac) else fac
        for c in chans:
            R[c][s:e, j] = R[c][s:e, j] * fac

    tamper_at = lambda j, s, code: mevent.__setitem__((s, j), ECODE[code]) if rng.random() < 0.4 else None
    SF = "sensor_fault"; DC = "data_corruption"; FZ = "falsification"

    for _ in range(DQ["drift_gain"]):
        js = pick(wfault)
        if not js: continue
        j = js[0]; busy.add(j); L = int(rng.integers(70, 160)) * SPD; s = rstart(L, 5)
        e = min(s + L, T); rate = float(rng.uniform(0.00012, 0.0006)); sgn = -1 if rng.random() < 0.75 else 1
        cap = float(rng.uniform(0.04, 0.12))
        dev = np.clip(sgn * rate * (np.arange(e - s) / SPD), -cap, cap)
        evi = new_event(SF, "gain_drift", "meter", md.feeder_id[j], [j], s, e, {"rate_per_day": round(sgn * rate, 5), "cap": round(cap, 3), "label_threshold": 0.01},
                        "Slow CT/ADC gain drift (older meters more likely): E, I, Q scaled by 1+dev(t); rows labelled once |dev|>=1%.", "human_decision")
        scale_ch(j, s, e, 1 + dev); mark(evi, s, e, [j], mask=(np.abs(dev) >= 0.01)[:, None] & sup_b[s:e, [j]])
    for _ in range(DQ["drift_volt"]):
        js = pick(wfault)
        if not js: continue
        j = js[0]; busy.add(j); L = int(rng.integers(50, 110)) * SPD; s = rstart(L, 5); e = min(s + L, T)
        off = float(rng.choice([-1, 1]) * rng.uniform(4, 11)); ramp = np.linspace(0, off, e - s)
        evi = new_event(SF, "voltage_offset_drift", "meter", md.feeder_id[j], [j], s, e, {"final_offset_v": round(off, 1), "label_threshold_v": 2.0},
                        "Voltage-sensor offset ramps to +/-4..11 V (energy channels untouched); rows labelled once |offset|>=2 V.", "notify_operator")
        live = sup_b[s:e, j]
        for c in ("V", "Vmin", "Vmax"):
            R[c][s:e, j] = np.where(live, R[c][s:e, j] + ramp, R[c][s:e, j])
        mark(evi, s, e, [j], mask=((np.abs(ramp) >= 2.0) & live)[:, None])
    for _ in range(DQ["bias"]):
        js = pick(wfault)
        if not js: continue
        j = js[0]; busy.add(j); L = int(rng.integers(20, 80)) * SPD; s = rstart(L, 5); e = min(s + L, T)
        f_ = float(rng.choice([rng.uniform(0.86, 0.93), rng.uniform(1.07, 1.14)]))
        evi = new_event(SF, "bias_step", "meter", md.feeder_id[j], [j], s, e, {"gain": round(f_, 3)},
                        "Constant gain step (mis-calibration after CT swap/firmware update): E, I, Q x(0.86-0.93 or 1.07-1.14) for 1-4 months.", "human_decision")
        scale_ch(j, s, e, f_); mark(evi, s, e, [j], mask=sup_b[s:e, [j]])
    for _ in range(DQ["clock"]):
        js = pick(wfault)
        if not js: continue
        j = js[0]; busy.add(j); L = int(rng.integers(7, 22)) * SPD; s = rstart(L, 5); e = min(s + L, T); k = int(rng.integers(2, 7))
        evi = new_event(SF, "clock_shift", "meter", md.feeder_id[j], [j], s, e, {"shift_intervals": k},
                        "RTC drift: every channel delayed by k half-hours (profile shifted vs neighbours).", "notify_operator")
        for c in ALL:
            R[c][s:e, j] = R[c][s - k:e - k, j].copy()
        mark(evi, s, e, [j])
    for _ in range(DQ["stuck"]):
        j = int(rng.integers(M)); L = int(rng.integers(6, 97)); s = rstart(L, 2); e = s + L
        evi = new_event(SF, "stuck_value", "meter", md.feeder_id[j], [j], s, e, {"frozen_intervals": L},
                        "Frozen register/firmware hang: all channels repeat the value of the previous interval.", "auto_correct")
        for c in ALL:
            R[c][s:e, j] = R[c][s - 1, j]
        mark(evi, s, e, [j])
    for _ in range(DQ["spike"]):
        j = int(rng.integers(M)); mode = str(rng.choice(["spike_gain", "negative_value", "saturation"])); L = int(rng.integers(1, 7)); s = rstart(L, 2); e = s + L
        evi = new_event(DC, mode, "meter", md.feeder_id[j], [j], s, e, {"mode": mode},
                        "Transmission/bit corruption on the energy field: x(8-40), sign flip, or saturation at 99.999 kWh.", "auto_correct")
        if mode == "spike_gain": R["E"][s:e, j] = (R["E"][s:e, j] + 0.05) * rng.uniform(8, 40)
        elif mode == "negative_value": R["E"][s:e, j] = -(R["E"][s:e, j] + 0.02)
        else: R["E"][s:e, j] = 99.999
        mark(evi, s, e, [j])
    for _ in range(DQ["unit"]):
        j = int(rng.integers(M)); L = int(rng.integers(2, 25)); s = rstart(L, 2); e = s + L
        evi = new_event(DC, "unit_error_x1000", "meter", md.feeder_id[j], [j], s, e, {"factor": 1000},
                        "Firmware unit bug: Wh reported as kWh (energy fields x1000) for 1-12 h.", "auto_correct")
        R["E"][s:e, j] *= 1000; R["X"][s:e, j] *= 1000; mark(evi, s, e, [j])

    # ----- falsification -----
    def fals_window(lo=14):
        L = int(rng.integers(20, 80)) * SPD; s = rstart(L, lo); return s, min(s + L, T)

    for _ in range(DQ["partial"]):
        js = pick(wcons)
        if not js: continue
        j = js[0]; busy.add(j); s, e = fals_window(); f_ = float(rng.uniform(0.4, 0.8))
        evi = new_event(FZ, "partial_reduction_bypass", "meter", md.feeder_id[j], [j], s, e, {"factor": round(f_, 2)},
                        "Physical bypass of part of the load: E, I, Q x(0.4-0.8) constant; tamper log at start w.p. 0.4. (P1 'partial reduction')", "human_decision")
        scale_ch(j, s, e, f_); mark(evi, s, e, [j], mask=sup_b[s:e, [j]]); tamper_at(j, s, "cover_open")
    for _ in range(DQ["selective"]):
        js = pick(wcons)
        if not js: continue
        j = js[0]; busy.add(j); s, e = fals_window(); f_ = float(rng.uniform(0.2, 0.55))
        hrs = (17, 24) if rng.random() < 0.5 else (11, 17)
        hm = (hod[s:e] >= hrs[0]) & (hod[s:e] < hrs[1]); fac = np.where(hm, f_, 1.0)
        evi = new_event(FZ, "selective_bypass", "meter", md.feeder_id[j], [j], s, e, {"factor": round(f_, 2), "hours": list(hrs)},
                        "Bypass only during high-load hours: E, I, Q x(0.2-0.55) inside the window of hours, untouched otherwise. (P1 'selective bypass')", "human_decision")
        scale_ch(j, s, e, fac); mark(evi, s, e, [j], mask=(hm & sup_b[s:e, j])[:, None]); tamper_at(j, s, "magnetic_tamper")
    for _ in range(DQ["zeroing"]):
        js = pick(wcons)
        if not js: continue
        j = js[0]; busy.add(j); s, e = fals_window(); h0 = int(rng.integers(10, 19)); h1 = h0 + int(rng.integers(5, 9))
        days_on = rng.random(NDAY) < 0.35
        zm = np.repeat(days_on, SPD)[s:e] & (hod[s:e] >= h0) & (hod[s:e] < h1)
        evi = new_event(FZ, "meter_zeroing", "meter", md.feeder_id[j], [j], s, e, {"hours": [h0, h1], "day_fraction": 0.35},
                        "Magnet/CT-shunt tampering: E, I, Q forced to ~0 for 5-8 h on ~35% of days while voltage keeps reporting.", "human_decision")
        for c in ("E", "X", "Q"): R[c][s:e, j] = np.where(zm, 0.0, R[c][s:e, j])
        R["I"][s:e, j] = np.where(zm, np.abs(rng.normal(0.03, 0.01, e - s)), R["I"][s:e, j])
        mark(evi, s, e, [j], mask=(zm & sup_b[s:e, j])[:, None])
        for t in np.where(zm & ~np.roll(zm, 1))[0][:40]:
            if rng.random() < 0.3: mevent[s + t, j] = ECODE["magnetic_tamper"]
    for _ in range(DQ["fdi_scale"]):
        js = pick(wcons)
        if not js: continue
        j = js[0]; busy.add(j); s, e = fals_window(); f_ = float(rng.uniform(0.3, 0.75))
        evi = new_event(FZ, "fdi_scale", "meter", md.feeder_id[j], [j], s, e, {"factor": round(f_, 2)},
                        "Cyber FDI after the meter: only the energy field is scaled x(0.3-0.75); current/voltage/PF stay true -> internal inconsistency.", "human_decision")
        R["E"][s:e, j] *= f_; mark(evi, s, e, [j], mask=((E_imp[s:e, j] > 0.002) & sup_b[s:e, j])[:, None])
    for _ in range(DQ["replay"]):
        js = pick(wcons)
        if not js: continue
        j = js[0]; busy.add(j); L = int(rng.integers(10, 40)) * SPD; s = rstart(L, 21); e = min(s + L, T)
        evi = new_event(FZ, "fdi_replay", "meter", md.feeder_id[j], [j], s, e, {"lag_days": 7},
                        "Replay attack: energy field replaced by the reported value of the same half-hour 7 days earlier; current/voltage true.", "human_decision")
        R["E"][s:e, j] = R["E"][s - 336:e - 336, j].copy(); mark(evi, s, e, [j], mask=sup_b[s:e, [j]])
    for _ in range(DQ["evasion"]):
        js = pick(wcons)
        if not js: continue
        j = js[0]; busy.add(j); s, e = fals_window(); base_f = float(rng.uniform(0.5, 0.8))
        walk = np.clip(base_f + ar1(NDAY, 0.9, 0.03, rng), 0.35, 0.95); fac = np.repeat(walk, SPD)[s:e] * (1 + rng.normal(0, 0.02, e - s))
        evi = new_event(FZ, "fdi_evasion", "meter", md.feeder_id[j], [j], s, e, {"base_factor": round(base_f, 2)},
                        "Evasive FDI: scaling factor wanders as a slow daily random walk + noise to mimic natural variance (P2 'evasion').", "human_decision")
        R["E"][s:e, j] *= fac; mark(evi, s, e, [j], mask=((E_imp[s:e, j] > 0.002) & sup_b[s:e, j])[:, None])
    for _ in range(DQ["inflate"]):
        js = pick(wcons)
        if not js: continue
        j = js[0]; busy.add(j); L = int(rng.integers(15, 45)) * SPD; s = rstart(L, 14); e = min(s + L, T); f_ = float(rng.uniform(1.35, 2.0))
        evi = new_event(FZ, "fdi_inflate", "meter", md.feeder_id[j], [j], s, e, {"factor": round(f_, 2)},
                        "FDI inflating a victim's consumption x(1.35-2.0) (energy field only): feeder balance gap turns negative.", "human_decision")
        R["E"][s:e, j] *= f_; mark(evi, s, e, [j], mask=((E_imp[s:e, j] > 0.002) & sup_b[s:e, j])[:, None])
    for _ in range(DQ["coord"]):
        js = pick(wcons, 3, same_feeder=True)
        if not js: continue
        for j in js: busy.add(j)
        L = int(rng.integers(10, 30)) * SPD; s = rstart(L, 14); e = min(s + L, T); f0 = float(rng.uniform(0.45, 0.7))
        evi = new_event(FZ, "coordinated_fdi", "meters_same_feeder", md.feeder_id[js[0]], js, s, e, {"factor_mean": round(f0, 2), "n_meters": len(js)},
                        "Coordinated FDI: 3 meters on one feeder scaled x~f0 over the same 15-40 day window (collective-defense case).", "human_decision")
        for j in js:
            R["E"][s:e, j] *= f0 * (1 + rng.uniform(-0.05, 0.05))
        mark(evi, s, e, js, mask=(E_imp[s:e][:, js] > 0.002) & sup_b[s:e][:, js])

    R_pre_E, R_pre_X = R["E"].copy(), R["X"].copy()               # before communication gaps
    # ----- communication loss -----
    pcom = {"PLC": 0.004, "GPRS": 0.0015, "NB-IoT": 0.0006}
    gap_mask = np.zeros((T, M), bool)
    for j in range(M):
        n_g = rng.poisson(pcom[md.comm_tech[j]] * T / 2.0)
        for st in rng.integers(0, T - 3, n_g):
            gap_mask[st:st + int(rng.integers(1, 3)), j] = True
    comm_w = np.array([{"PLC": 3, "GPRS": 1.5, "NB-IoT": 0.6}[t] for t in md.comm_tech]); comm_w = comm_w / comm_w.sum()
    comm_masks = []
    for _ in range(DQ["comm"]):
        j = int(rng.choice(M, p=comm_w)); L = int(np.clip(rng.lognormal(2.6, 0.9), 3, 96)); s = rstart(L, 2); e = s + L
        evi = new_event("comm_loss", "comm_dropout_burst", "meter", md.feeder_id[j], [j], s, e, {"intervals": L},
                        "Communication outage: all measured fields missing (NaN) for 1.5-48 h; PLC meters most exposed.",
                        "auto_correct" if L <= 12 else "notify_operator")
        mark(evi, s, e, [j]); gap_mask[s:e, j] = True
    # minor gaps are labelled normal/minor_gap
    mg = gap_mask & (lab_cls == 0)
    lab_sub[mg] = subcode("minor_gap")
    for c in ALL:
        R[c] = np.where(gap_mask, np.nan, R[c])
    mevent = np.where(gap_mask, 0, mevent)
    fa = rng.random((T, M)) < 0.0001                              # spurious tamper alarms (technician visits etc.)
    mevent = np.where(fa & (mevent == 0), ECODE["cover_open"], mevent)

    # ===== STAGE 1 again: feeder-head observer meter, then the labels =====
    fmeas = gb.feeder_measurements(S, EL, R, Pref, vf_bus, freq, loss_mult, thd_extra)
    frows = []
    for f, d0 in zip(feeders, fmeas):
        c = fcols[f]
        fe = feeder_evt[f].copy()
        # outages/shedding/cable events that cover the whole feeder or its feeder-head
        for evi, ev in enumerate(evs, start=1):
            if ev["subtype"] in ("outage", "load_shedding", "transformer_trip") and (ev["scope"] == "node" or ev["feeder_id"] == f):
                fe[ev["s"]:ev["e"]] = evi
            elif ev["subtype"] == "cable_fault" and ev["feeder_id"] == f:
                fe[ev["s"]:ev["e"]] = evi
            elif ev["subtype"] == "demand_surge" and ev["feeder_id"] == f:
                fe[ev["s"]:ev["e"]] = evi
        fe = np.where(fe == 0, freq_lab, fe)
        lab_f = np.array([0] + [LCODE[e_["label"]] for e_ in evs])[fe]
        sub_f = np.array(["none"] + [e_["subtype"] for e_ in evs])[fe]
        eid_f = np.array(["none"] + [e_["event_id"] for e_ in evs])[fe]
        disc = (E_imp[:, c].sum(1) - E_exp[:, c].sum(1)) - (np.nan_to_num(R_pre_E[:, c]).sum(1) - np.nan_to_num(R_pre_X[:, c]).sum(1))
        d0 = d0.assign(gt_label=np.array(LABELS)[lab_f], gt_subtype=sub_f, gt_event_id=eid_f,
                       gt_metering_discrepancy_kwh=disc,
                       gt_n_meters_falsified=(lab_cls[:, c] == LCODE["falsification"]).sum(1),
                       gt_n_meters_sensor_fault=(lab_cls[:, c] == LCODE["sensor_fault"]).sum(1))
        frows.append(d0[FEEDER_COLUMNS])
    feeder_df = pd.concat(frows, ignore_index=True)

    # ===== meter table =====
    lab_arr = np.array(LABELS)[lab_cls]
    sub_arr = np.array(SUBS)[lab_sub]
    eid_lut = np.array(["none"] + [e_["event_id"] for e_ in evs])
    resp_arr = np.array(RESP)[lab_rsp]
    trust = np.where(np.isin(lab_cls, [LCODE["sensor_fault"], LCODE["data_corruption"], LCODE["falsification"]]), 0, 1).astype(float)
    trust = np.where(np.isnan(R["E"]), np.nan, trust)
    pvx = md.has_pv.values[None, :].astype(bool)

    def fl(a): return a.ravel()
    tsr = np.repeat(ts.values, M)
    df = pd.DataFrame(dict(
        timestamp=tsr, node_id=nd, feeder_id=np.tile(md.feeder_id.values, T), meter_id=np.tile(mid, T),
        energy_kwh=fl(R["E"]), pv_export_kwh=fl(np.where(pvx, R["X"], 0.0)), reactive_kvarh=fl(R["Q"]),
        voltage_v=fl(R["V"]), voltage_min_v=fl(R["Vmin"]), voltage_max_v=fl(R["Vmax"]), current_a=fl(R["I"]),
        power_factor=fl(R["PF"]), thd_v_pct=fl(R["THD"]), meter_event_code=np.array(EVCODES)[mevent].ravel(),
        gt_label=fl(lab_arr), gt_subtype=fl(sub_arr), gt_event_id=eid_lut[lab_evt].ravel(),
        gt_data_trustworthy=fl(trust), gt_expected_response=fl(resp_arr),
        gt_true_energy_kwh=fl(E_imp), gt_true_voltage_v=fl(V)))
    df["pv_export_kwh"] = np.where(df.energy_kwh.isna(), np.nan, df.pv_export_kwh)
    for c, d_ in (("energy_kwh", 3), ("pv_export_kwh", 3), ("reactive_kvarh", 3), ("voltage_v", 1), ("voltage_min_v", 1),
                  ("voltage_max_v", 1), ("current_a", 2), ("power_factor", 3), ("thd_v_pct", 2),
                  ("gt_true_energy_kwh", 3), ("gt_true_voltage_v", 1)):
        df[c] = df[c].round(d_)
    for c, d_ in (("feeder_kwh_net", 3), ("feeder_voltage_v", 1), ("feeder_current_a", 2), ("feeder_thd_pct", 2), ("frequency_hz", 3),
                  ("sum_meters_import_kwh", 3), ("sum_meters_export_kwh", 3), ("energy_balance_gap_kwh", 3), ("energy_balance_gap_pct", 2),
                  ("gt_tech_loss_kwh", 3), ("gt_streetlight_kwh", 3), ("gt_metering_discrepancy_kwh", 3)):
        feeder_df[c] = feeder_df[c].round(d_)

    wdf = gb.weather_table(nd, wx)
    # events table
    cnt = np.bincount(lab_evt.ravel(), minlength=len(evs) + 1)
    erows = []
    for k, ev in enumerate(evs, start=1):
        erows.append(dict(event_id=ev["event_id"], node_id=nd, gt_label=ev["label"], gt_subtype=ev["subtype"], scope=ev["scope"],
                          feeder_id=ev["feeder_id"], meter_ids=";".join(mid[ev["cols"]]) if ev["cols"] else "",
                          start_ts=ts[ev["s"]], end_ts=ts[min(ev["e"], T) - 1], n_intervals=ev["e"] - ev["s"],
                          n_labelled_meter_rows=int(cnt[k]) if ev["scope"] != "node" or ev["subtype"] != "frequency_excursion" else int((freq_lab == k).sum()),
                          expected_response=ev["expected_response"], parameters=json.dumps(ev["params"]), generation_method=ev["method"]))
    return df, feeder_df, md, wdf, pd.DataFrame(erows)


FEEDER_COLUMNS = ["timestamp", "node_id", "feeder_id", "n_meters_total", "n_meters_reporting", "feeder_kwh_net", "feeder_voltage_v",
                  "feeder_current_a", "feeder_thd_pct", "frequency_hz", "sum_meters_import_kwh", "sum_meters_export_kwh",
                  "energy_balance_gap_kwh", "energy_balance_gap_pct", "gt_label", "gt_subtype", "gt_event_id",
                  "gt_tech_loss_kwh", "gt_streetlight_kwh", "gt_metering_discrepancy_kwh", "gt_n_meters_falsified",
                  "gt_n_meters_sensor_fault"]


# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="energymind_tn_dataset")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--contamination", type=float, default=1.0,
                    help="multiplier on the number of injected data-quality / falsification events")
    a = ap.parse_args()
    SEED, CONT = a.seed, a.contamination
    os.makedirs(os.path.join(a.out, "data"), exist_ok=True)
    heat = make_heat_episodes(np.random.default_rng(SEED))
    fds, mds, wxs, evs_all = [], [], [], []
    for ni, nd in enumerate(NODES):
        df, fdf, md, wdf, edf = inject_node(nd, ni, heat)
        df.to_csv(os.path.join(a.out, "data", f"{nd}_meter_readings.csv.gz"), index=False, compression="gzip")
        fds.append(fdf); mds.append(md); wxs.append(wdf); evs_all.append(edf)
        print(nd, df.shape, "non-normal row share: %.2f%%" % (100 * (df.gt_label != "normal").mean()))
    pd.concat(fds).to_csv(os.path.join(a.out, "data", "feeder_readings.csv.gz"), index=False, compression="gzip")
    pd.concat(wxs).to_csv(os.path.join(a.out, "data", "weather_calendar.csv.gz"), index=False, compression="gzip")
    pd.concat(mds).to_csv(os.path.join(a.out, "data", "meters_metadata.csv"), index=False)
    pd.concat(evs_all).to_csv(os.path.join(a.out, "data", "anomaly_events.csv"), index=False)
    print("done")
