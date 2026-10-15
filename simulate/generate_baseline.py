#!/usr/bin/env python3
"""
generate_baseline.py  -  STAGE 1: the HEALTHY network (no faults, no events)
===========================================================================
EnergyMind Tunisia / IEEE TSYP14.  Simulates a smart-metered Tunisian neighborhood:

  weather  ->  household load (+AC, heating, PV)  ->  feeder voltage drop  ->  meter measurements
                                                                          ->  feeder-head "observer" meter

Everything here describes how a HEALTHY system behaves.  Nothing in this file injects an anomaly.
`inject_anomalies.py` (STAGE 2) imports the functions below and calls them in order, inserting
genuine grid events, sensor faults, corruption and falsification between the steps.

Run on its own it writes an anomaly-free dataset:
    python generate_baseline.py --out ./clean_dataset --seed 2025

Three nodes (archetypes), one year (2025), 30-minute resolution, local time Africa/Tunis (UTC+1).
"""
import argparse, os
import numpy as np
import pandas as pd

YEAR = 2025
SPD = 48                      # 30-min intervals per day
NDAY = 365
T = NDAY * SPD
SEED_DEFAULT = 2025

# ----------------------------------------------------------------------------- calendar
ts = pd.date_range(f"{YEAR}-01-01", periods=T, freq="30min")
hod = (ts.hour.values + ts.minute.values / 60.0).astype(float)     # interval start hour
doy = ts.dayofyear.values
dow = ts.dayofweek.values                                          # Mon=0
# Tunisian public holidays 2025 (lunar dates +-1 day approximate)
HOL = pd.to_datetime(["2025-01-01", "2025-01-14", "2025-03-20", "2025-03-30", "2025-03-31",
                      "2025-04-01", "2025-04-09", "2025-05-01", "2025-06-06", "2025-06-07",
                      "2025-06-27", "2025-07-25", "2025-08-13", "2025-09-04", "2025-10-15"])
is_hol = np.isin(ts.normalize().values, HOL.values)
is_ram = (ts >= "2025-03-01") & (ts < "2025-03-30")
is_wkend = dow >= 5                                                # Sat/Sun


def G(h, mu, s):                                                   # circular gaussian bump on 24h clock
    d = (h - mu + 12.0) % 24.0 - 12.0
    return np.exp(-0.5 * (d / s) ** 2)


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def ar1(n, phi, sd, rng):
    x = np.zeros(n)
    e = rng.normal(0, sd, n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x



# ----------------------------------------------------------------------------- archetypes
NODES = {
    "N1": dict(name="Coastal urban suburb (Tunis/Ariana type)", lat=36.865, lon=10.195,
               feeders=[10, 10, 10], tmean=19.0, tamp=8.8, syn_sd=2.0, diur_w=3.5, diur_s=5.0,
               heat_gain=1.0, rh=68, ac_p=0.70, pv_p=0.22, wh_p=0.12, dist=(60, 350), kdrop=7.0,
               loss_f=1.0, node_factor=1.00,
               mix={"res_small": 0.12, "res_medium": 0.50, "res_large": 0.28, "small_commercial": 0.10}),
    "N2": dict(name="Interior hot town (Kairouan type)", lat=35.675, lon=10.10,
               feeders=[8, 7], tmean=20.2, tamp=9.8, syn_sd=2.3, diur_w=5.0, diur_s=7.5,
               heat_gain=1.2, rh=52, ac_p=0.80, pv_p=0.08, wh_p=0.08, dist=(100, 700), kdrop=9.5,
               loss_f=1.4, node_factor=0.95,
               mix={"res_small": 0.15, "res_medium": 0.40, "res_large": 0.40, "small_commercial": 0.05}),
    "N3": dict(name="Mixed residential/commercial (Sfax type)", lat=34.74, lon=10.76,
               feeders=[8, 7], tmean=20.3, tamp=8.0, syn_sd=1.9, diur_w=3.8, diur_s=5.0,
               heat_gain=1.0, rh=64, ac_p=0.75, pv_p=0.15, wh_p=0.10, dist=(50, 450), kdrop=8.0,
               loss_f=1.15, node_factor=1.05,
               mix={"res_small": 0.10, "res_medium": 0.40, "res_large": 0.25, "small_commercial": 0.25}),
}
CLS = {  # annual gross consumption targets (kWh) calibrated to STEG residential mean ~1.45-1.5 MWh/yr
    "res_small": dict(dw="apartment", area=(40, 65), occ=(1, 2), annual=850, units=(0, 1), kw=0.9, act=0.45),
    "res_medium": dict(dw="apartment", area=(65, 110), occ=(3, 5), annual=1150, units=(1, 2), kw=0.95, act=0.70),
    "res_large": dict(dw="villa", area=(140, 300), occ=(4, 7), annual=2000, units=(2, 4), kw=1.0, act=1.20),
    "small_commercial": dict(dw="shop", area=(20, 80), occ=(0, 0), annual=3300, units=(1, 2), kw=1.3, act=1.0),
}


# ----------------------------------------------------------------------------- weather
def make_heat_episodes(rng):
    eps = []
    for _ in range(4):
        eps.append((int(rng.integers(160, 236)), int(rng.integers(3, 8)), float(rng.uniform(4, 9))))
    return eps


def make_weather(nd, rng, heat):
    P = NODES[nd]
    d = np.arange(NDAY)
    seas = np.cos(2 * np.pi * (d - 204) / 365)
    dm = P["tmean"] + P["tamp"] * seas + ar1(NDAY, 0.7, P["syn_sd"], rng)
    for s, dur, mag in heat:
        for k in range(dur):
            if s + k < NDAY:
                dm[s + k] += mag * P["heat_gain"] * np.sin(np.pi * (k + 0.5) / dur) ** 0.5
    amp = P["diur_w"] + (P["diur_s"] - P["diur_w"]) * (0.5 + 0.5 * seas)
    temp = np.repeat(dm, SPD) + np.repeat(amp, SPD) * np.cos(2 * np.pi * (hod + 0.25 - 15.0) / 24.0)
    temp += pd.Series(rng.normal(0, 0.5, T)).rolling(4, min_periods=1).mean().values
    rh = np.clip(P["rh"] - 2.0 * (temp - 20) + ar1(T, 0.97, 1.2, rng), 15, 98)
    lat = np.radians(P["lat"])
    decl = np.radians(23.45 * np.sin(np.radians(360.0 / 365 * (284 + doy))))
    solar_t = hod + 0.25 - 1.0 + P["lon"] / 15.0
    cosz = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(np.radians(15 * (solar_t - 12)))
    clear = 1050.0 * np.clip(cosz, 0, None) ** 1.15
    kt_mean = 0.58 + 0.20 * (0.5 + 0.5 * np.cos(2 * np.pi * (d - 190) / 365))
    kt_d = np.clip(kt_mean + 0.17 * rng.standard_normal(NDAY), 0.12, 0.95)
    kt = np.repeat(kt_d, SPD) * np.clip(1 + pd.Series(rng.normal(0, 0.08, T)).rolling(3, min_periods=1).mean().values, 0.4, 1.1)
    ghi = np.clip(clear * np.clip(kt, 0.05, 1.0), 0, None)
    return dict(temp=temp, rh=rh, ghi=ghi, cosz=cosz)


# ----------------------------------------------------------------------------- meters
def make_meters(nd, rng):
    P = NODES[nd]
    rows, k = [], 0
    cnames, cprobs = list(P["mix"].keys()), list(P["mix"].values())
    for fi, n in enumerate(P["feeders"], start=1):
        fid = f"{nd}-F{fi}"
        bearing = rng.uniform(0, 2 * np.pi)
        dists = np.sort(rng.uniform(P["dist"][0], P["dist"][1], n))
        for j in range(n):
            k += 1
            c = rng.choice(cnames, p=cprobs)
            C = CLS[c]
            has_ac_p = {"res_small": P["ac_p"] - 0.2, "res_medium": P["ac_p"],
                        "res_large": min(P["ac_p"] + 0.15, 0.97), "small_commercial": 0.85}[c]
            has_ac = rng.random() < has_ac_p
            units = int(rng.integers(max(C["units"][0], 1), C["units"][1] + 1)) if has_ac else 0
            pv_p = P["pv_p"] * {"res_small": 0.0, "res_medium": 0.8, "res_large": 1.8, "small_commercial": 0.3}[c]
            has_pv = rng.random() < pv_p
            yr = int(rng.choice(np.arange(2016, 2025), p=np.array([5, 6, 8, 10, 12, 14, 16, 16, 13]) / 100.0))
            tech = "PLC" if (yr < 2019 and rng.random() < 0.7) else ("NB-IoT" if (yr >= 2022 and rng.random() < 0.6) else "GPRS")
            lat = P["lat"] + (dists[j] * np.cos(bearing) + rng.normal(0, 25)) / 111000.0
            lon = P["lon"] + (dists[j] * np.sin(bearing) + rng.normal(0, 25)) / (111000.0 * np.cos(np.radians(P["lat"])))
            rows.append(dict(
                meter_id=f"{nd}-M{k:03d}", node_id=nd, feeder_id=fid, phase=f"L{(k % 3) + 1}",
                customer_class=c, dwelling_type=C["dw"], floor_area_m2=int(rng.integers(*C["area"])),
                occupants=int(rng.integers(C["occ"][0], C["occ"][1] + 1)),
                distance_to_transformer_m=int(dists[j]), lat=round(lat, 5), lon=round(lon, 5),
                has_ac=int(has_ac), ac_units=units, ac_unit_kw=round(C["kw"] * rng.uniform(0.85, 1.2), 2) if has_ac else 0.0,
                has_pv=int(has_pv), pv_kwp=float(rng.choice([1.5, 2.0, 3.0, 4.5])) if has_pv else 0.0,
                has_electric_heater=int(c != "small_commercial" and rng.random() < 0.35),
                has_electric_water_heater=int(c in ("res_medium", "res_large") and rng.random() < P["wh_p"]),
                meter_install_year=yr, comm_tech=tech, meter_model=f"AMI-1P-CL1-{tech}", meter_accuracy_class=1.0,
                annual_kwh_target=round(C["annual"] * P["node_factor"] * (1 + 0.2 * has_ac) * rng.lognormal(0, 0.18)),
                fault_weight=round(1 + 0.35 * (2024 - yr), 2)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- baseline load
def gen_load(nd, md, wx, rng):
    M = len(md)
    temp = wx["temp"][:, None]
    shift = rng.normal(0, 0.45, M)[None, :]
    h = (hod[:, None] - shift) % 24
    is_shop = (md.customer_class == "small_commercial").values
    wk = (is_wkend | is_hol)[:, None]
    ram = np.asarray(is_ram)[:, None]
    a_wd = 0.12 + 0.55 * G(h, 7.3, 1.0) + 0.40 * G(h, 13.3, 1.3) + 1.0 * G(h, 20.6, 1.9) + 0.15 * G(h, 17.0, 1.2)
    a_we = 0.14 + 0.35 * G(h, 9.8, 1.8) + 0.55 * G(h, 13.5, 1.5) + 1.0 * G(h, 20.8, 2.1)
    a_rm = (0.10 + 0.55 * G(h, 3.8, 0.7) + 0.15 * G(h, 10.0, 2.0) + 0.25 * G(h, 14.5, 1.5)
            + 1.15 * G(h, 19.0, 1.1) + 0.95 * G(h, 22.3, 1.6))               # suhoor / iftar pattern
    a_res = np.where(ram, a_rm, np.where(wk, a_we, a_wd))
    open_wd = sig((h - 8.4) / 0.25) * (1 - sig((h - 20.5) / 0.25))
    open_rm = 0.6 * sig((h - 9) / 0.25) * (1 - sig((h - 14) / 0.25)) + sig((h - 19.5) / 0.25) * (1 - sig((h - 23.5) / 0.25))
    closed = ((dow == 6) | is_hol)[:, None]
    a_shop_open = np.where(ram, open_rm, open_wd)
    a_shop = 0.12 + 0.9 * np.where(closed, 0.05, a_shop_open)
    act = np.where(is_shop[None, :], a_shop, a_res)

    # occupancy (summer holidays, shop closures) and daily variability
    occ = np.ones((NDAY, M))
    for i in range(M):
        if is_shop[i]:
            if rng.random() < 0.35 and NDAY > 240:
                s = int(rng.integers(205, 235)); occ[s:s + 14, i] = 0.1
        elif rng.random() < 0.55 and NDAY > 270:
            s = int(rng.integers(150, 260)); occ[s:s + int(rng.integers(7, 22)), i] = 0.06
    z = np.zeros((NDAY, M)); e = rng.normal(0, 0.18, (NDAY, M))
    for d in range(1, NDAY):
        z[d] = 0.3 * z[d - 1] + e[d]
    dayf = np.exp(z)
    occT, dayT = np.repeat(occ, SPD, axis=0), np.repeat(dayf, SPD, axis=0)
    light = (1 + 0.18 * np.cos(2 * np.pi * (doy - 355) / 365))[:, None]

    base = np.where(is_shop, rng.uniform(0.25, 0.5, M), rng.uniform(0.05, 0.12, M))[None, :]
    amp = (rng.lognormal(np.log(0.9), 0.25, M) * np.array([CLS[c]["act"] for c in md.customer_class]))[None, :]
    fridge = base * (1 + 0.025 * np.clip(temp - 20, 0, None)) * (1 + 0.08 * rng.standard_normal((T, M)))
    burst = (rng.random((T, M)) < (0.04 + 0.10 * np.clip(act, 0, 1.2))) * np.clip(rng.exponential(0.55, (T, M)), 0, 2.8)
    non_ac = fridge + amp * act * dayT * occT * light * np.clip(1 + 0.10 * rng.standard_normal((T, M)), 0.5, 1.6) + burst * occT

    sched_res = np.clip(0.22 + 0.55 * G(h, 15.5, 2.4) + 0.95 * G(h, 0.2, 3.0), 0, 1)
    sched = np.where(is_shop[None, :], np.where(closed, 0.1, a_shop_open), sched_res)
    duty = sig((temp - 25.5) / 1.5) * sched * occT
    units_eff = 1 + np.clip(md.ac_units.values - 1, 0, None) * 0.6
    ac = (md.has_ac.values * md.ac_unit_kw.values * units_eff)[None, :] * duty * (0.8 + 0.3 * rng.random((T, M)))
    heat = (md.has_electric_heater.values * 1.2)[None, :] * sig((12.5 - temp) / 1.6) * np.clip(0.2 + 0.8 * G(h, 21, 3) + 0.3 * G(h, 7, 1.2), 0, 1) * occT * 0.7
    wh = (md.has_electric_water_heater.values * 1.0)[None, :] * (rng.random((T, M)) < 0.6 * (G(h, 6.7, 0.8) + G(h, 20.8, 1.3))) * (0.6 + 0.8 * sig((14 - temp) / 3)) * occT
    total = non_ac + ac + heat + wh
    scale = md.annual_kwh_target.values * (T / 17520.0) / (total.sum(0) * 0.5)      # calibrate to annual target
    return total * scale, ac * scale


def make_pv(md, wx):
    ghi, temp = wx["ghi"][:, None], wx["temp"][:, None]
    cell = temp + 0.03 * ghi
    pv = md.pv_kwp.values[None, :] * ghi / 1000.0 * 0.78 * (1 - 0.004 * np.clip(cell - 25, 0, None))
    return np.minimum(pv, md.pv_kwp.values[None, :])




# ----------------------------------------------------------------------------- per-node baseline state
def build_baseline(nd, ni, heat, seed=SEED_DEFAULT):
    """Weather, meter fleet, household load and PV for one node.  Returns the state used by every later step.
    The random generator is part of the state: later steps keep drawing from it (this keeps results reproducible)."""
    rng = np.random.default_rng([seed, ni + 1])
    P = NODES[nd]
    wx = make_weather(nd, rng, heat)
    md = make_meters(nd, rng)
    load, ac = gen_load(nd, md, wx, rng)
    pv = make_pv(md, wx)
    feeders = list(dict.fromkeys(md.feeder_id))
    fcols = {f: np.where(md.feeder_id.values == f)[0] for f in feeders}
    return dict(nd=nd, ni=ni, rng=rng, P=P, wx=wx, md=md, M=len(md), load=load, ac=ac, pv=pv,
                feeders=feeders, fcols=fcols, dist=md.distance_to_transformer_m.values.astype(float),
                mid=md.meter_id.values)


def apply_supply(S, supply, lmult):
    """Loads actually drawn.  supply: 1 = energised, 0 = cut (T x M).  lmult: load multiplier (T x M)."""
    sup_b = supply > 0
    load_e = S["load"] * lmult * sup_b
    ac_e = S["ac"] * lmult * sup_b
    pv_e = S["pv"] * sup_b
    net = load_e - pv_e                                         # kW, + import / - export
    imp_kw, exp_kw = np.clip(net, 0, None), np.clip(-net, 0, None)
    return dict(sup_b=sup_b, load_e=load_e, ac_e=ac_e, pv_e=pv_e, net=net, imp_kw=imp_kw, exp_kw=exp_kw)


def base_voltage_profile(S):
    """Slowly varying no-load voltage at the transformer (OU process around 234.5 V, slightly lower in summer)."""
    ou = ar1(T, np.exp(-0.5 / 3.0), 1.0 * np.sqrt(1 - np.exp(-1 / 3.0)), S["rng"])
    vnl_node = 234.5 + ou - 1.0 * (0.5 + 0.5 * np.cos(2 * np.pi * (doy - 204) / 365))
    loss_mult = np.ones(T)                                      # technical-loss multiplier (1 = healthy transformer)
    thd_extra = np.zeros(T)                                     # extra distortion (0 = healthy transformer)
    return vnl_node, loss_mult, thd_extra


def feeder_voltages(S, net, vnl_node):
    """Voltage at every meter: transformer voltage minus the drop along the feeder (grows with load and distance),
    minus the service-cable drop, plus the rise when PV exports."""
    P, M, rng = S["P"], S["M"], S["rng"]
    feeders, fcols, dist = S["feeders"], S["fcols"], S["dist"]
    V = np.zeros((T, M)); Pf = {}; Pref = {}; vf_bus = {}
    for f in feeders:
        c = fcols[f]
        Pf[f] = net[:, c].sum(1)
        Pref[f] = max(np.percentile(Pf[f], 98), 1.0)
        off = rng.normal(0, 0.5)
        vnl_f = vnl_node + off
        ratio = np.clip(Pf[f] / Pref[f], -0.5, 1.5)[:, None]
        drop = P["kdrop"] * ratio * (dist[c][None, :] / 300.0)
        local_i = net[:, c] * 1000.0 / 230.0
        V[:, c] = vnl_f[:, None] - drop - 0.12 * local_i
        vf_bus[f] = vnl_f - 2.0 * np.clip(Pf[f] / Pref[f], -0.5, 1.5)
    V = np.clip(V, 195, 262)
    vmin_x = np.zeros((T, M)); vmax_x = np.zeros((T, M)); thd_x = np.zeros((T, M))   # extra dips/rises/distortion (0 = healthy)
    return V, Pf, Pref, vf_bus, vmin_x, vmax_x, thd_x


def grid_frequency(S):
    """Grid frequency: 50 Hz with small AR(1) noise (sd about 0.02 Hz)."""
    return 50.0 + ar1(T, 0.8, 0.012, S["rng"])


def derive_electrical(S, EL, V, thd_x, thd_extra):
    """True electrical quantities from loads and voltage: power factor, current, energy, reactive energy, THD."""
    rng, M = S["rng"], S["M"]
    sup_b, load_e, ac_e = EL["sup_b"], EL["load_e"], EL["ac_e"]
    net, imp_kw, exp_kw = EL["net"], EL["imp_kw"], EL["exp_kw"]
    V = np.where(sup_b, V, 0.0)
    ac_frac = ac_e / (load_e + 1e-6)
    pf = 0.985 - 0.12 * ac_frac - 0.10 * np.exp(-imp_kw / 0.15) + rng.normal(0, 0.01, (T, M))
    pf = np.where(exp_kw > 0, 0.99, pf); pf = np.clip(pf, 0.55, 1.0)
    I_true = np.where(V > 1, np.abs(net) / pf * 1000.0 / np.maximum(V, 1.0), 0.0)
    E_imp, E_exp = imp_kw * 0.5, exp_kw * 0.5
    Q_true = np.abs(net) * 0.5 * np.tan(np.arccos(pf))
    thd_base = rng.uniform(1.8, 3.2, M)[None, :] + 1.2 * np.clip(load_e / 2.0, 0, 1.5) + 0.9 * (exp_kw > 0) + thd_x + thd_extra[:, None]
    thd_true = np.clip(thd_base + rng.normal(0, 0.2, (T, M)), 0.8, 9)
    return dict(V=V, pf=pf, I_true=I_true, E_imp=E_imp, E_exp=E_exp, Q_true=Q_true, thd_true=thd_true)


def measure_reported(S, D, vmin_x, vmax_x):
    """What the smart meters REPORT: true values + static gain error (class 1 meter, sd 0.4 %) + reading noise."""
    rng, M = S["rng"], S["M"]
    V, pf, I_true, E_imp, E_exp, Q_true, thd_true = (D[k] for k in ("V", "pf", "I_true", "E_imp", "E_exp", "Q_true", "thd_true"))
    g_i = rng.normal(0, 0.004, M)[None, :]
    R = {}
    R["E"] = E_imp * (1 + g_i) * (1 + rng.normal(0, 0.003, (T, M)))
    R["X"] = E_exp * (1 + g_i) * (1 + rng.normal(0, 0.003, (T, M)))
    R["Q"] = Q_true * (1 + g_i)
    R["I"] = I_true * (1 + g_i) * (1 + rng.normal(0, 0.012, (T, M)))
    R["V"] = np.where(V > 1, V + rng.normal(0, 0.25, (T, M)), 0.0)
    R["Vmin"] = np.where(V > 1, V - (0.6 + rng.gamma(2, 0.5, (T, M)) + 0.08 * I_true) - vmin_x, 0.0)
    R["Vmax"] = np.where(V > 1, V + 0.4 + rng.gamma(2, 0.4, (T, M)) + vmax_x, 0.0)
    R["PF"] = np.where(V > 1, np.minimum(pf + rng.normal(0, 0.004, (T, M)), 1.0), np.nan)
    R["THD"] = np.where(V > 1, thd_true + rng.normal(0, 0.15, (T, M)), np.nan)
    return R


def feeder_measurements(S, EL, R, Pref, vf_bus, freq, loss_mult, thd_extra):
    """Feeder-head "observer" meter: true net flow + technical losses + a small unmetered street-lighting load,
    compared with the sum of what the customer meters reported.  Returns one DataFrame per feeder."""
    nd, P, wx, rng = S["nd"], S["P"], S["wx"], S["rng"]
    sup_b, net = EL["sup_b"], EL["net"]
    out = []
    for f in S["feeders"]:
        c = S["fcols"][f]; n = len(c)
        net_f = net[:, c].sum(1)
        lamp = rng.uniform(0.03, 0.09) * (wx["cosz"] < 0.02)
        loss = np.abs(net_f) * (0.012 + 0.03 * np.abs(net_f) / Pref[f]) * P["loss_f"] * loss_mult
        feed_on = sup_b[:, c].any(1)
        meas = (net_f + (loss + lamp) * feed_on) * 0.5 * (1 + rng.normal(0, 0.003, T))
        meas = np.where(feed_on, meas, 0.0)
        sum_i = np.nansum(R["E"][:, c], 1); sum_x = np.nansum(R["X"][:, c], 1)
        nrep = (~np.isnan(R["E"][:, c])).sum(1)
        gap = meas - (sum_i - sum_x)
        thd_f = np.where(feed_on, np.clip(2.0 + 0.8 * np.clip(net_f / Pref[f], 0, 1.5) + thd_extra + rng.normal(0, 0.15, T), 1, 9), np.nan)
        vbus = np.where(feed_on, vf_bus[f] + rng.normal(0, 0.2, T), 0.0)
        out.append(pd.DataFrame(dict(
            timestamp=ts, node_id=nd, feeder_id=f, n_meters_total=n, n_meters_reporting=nrep,
            feeder_kwh_net=meas, feeder_voltage_v=vbus,
            feeder_current_a=np.where(feed_on, np.abs(net_f) * 1000.0 / (3 * 230.0 * 0.95), 0.0),
            feeder_thd_pct=thd_f, frequency_hz=np.where(feed_on, freq, np.nan),
            sum_meters_import_kwh=sum_i, sum_meters_export_kwh=sum_x,
            energy_balance_gap_kwh=gap,
            energy_balance_gap_pct=np.where(meas > 0.3, 100 * gap / np.maximum(meas, 0.3), 0.0),
            gt_tech_loss_kwh=np.where(feed_on, loss, 0) * 0.5, gt_streetlight_kwh=np.where(feed_on, lamp, 0) * 0.5)))
    return out


def weather_table(nd, wx):
    return pd.DataFrame(dict(timestamp=ts, node_id=nd, temp_c=wx["temp"].round(1), humidity_pct=wx["rh"].round(0),
                             ghi_wm2=wx["ghi"].round(0), is_weekend=is_wkend.astype(int), is_public_holiday=is_hol.astype(int),
                             is_ramadan=np.asarray(is_ram).astype(int)))


METER_DECIMALS = {"energy_kwh": 3, "pv_export_kwh": 3, "reactive_kvarh": 3, "voltage_v": 1, "voltage_min_v": 1,
                  "voltage_max_v": 1, "current_a": 2, "power_factor": 3, "thd_v_pct": 2,
                  "gt_true_energy_kwh": 3, "gt_true_voltage_v": 1}
FEEDER_DECIMALS = {"feeder_kwh_net": 3, "feeder_voltage_v": 1, "feeder_current_a": 2, "feeder_thd_pct": 2, "frequency_hz": 3,
                   "sum_meters_import_kwh": 3, "sum_meters_export_kwh": 3, "energy_balance_gap_kwh": 3,
                   "energy_balance_gap_pct": 2, "gt_tech_loss_kwh": 3, "gt_streetlight_kwh": 3}


# ----------------------------------------------------------------------------- stand-alone: anomaly-free dataset
def build_clean_node(nd, ni, heat, seed=SEED_DEFAULT):
    """The whole healthy pipeline with nothing injected: supply always on, healthy transformer, no events, no faults."""
    S = build_baseline(nd, ni, heat, seed)
    M = S["M"]
    EL = apply_supply(S, np.ones((T, M)), np.ones((T, M)))
    vnl_node, loss_mult, thd_extra = base_voltage_profile(S)
    V, Pf, Pref, vf_bus, vmin_x, vmax_x, thd_x = feeder_voltages(S, EL["net"], vnl_node)
    freq = grid_frequency(S)
    D = derive_electrical(S, EL, V, thd_x, thd_extra)
    R = measure_reported(S, D, vmin_x, vmax_x)
    feeders = feeder_measurements(S, EL, R, Pref, vf_bus, freq, loss_mult, thd_extra)
    fdf = pd.concat(feeders, ignore_index=True)
    pvx = S["md"].has_pv.values[None, :].astype(bool)
    fl = lambda a: a.ravel()
    df = pd.DataFrame(dict(
        timestamp=np.repeat(ts.values, M), node_id=nd, feeder_id=np.tile(S["md"].feeder_id.values, T), meter_id=np.tile(S["mid"], T),
        energy_kwh=fl(R["E"]), pv_export_kwh=fl(np.where(pvx, R["X"], 0.0)), reactive_kvarh=fl(R["Q"]),
        voltage_v=fl(R["V"]), voltage_min_v=fl(R["Vmin"]), voltage_max_v=fl(R["Vmax"]), current_a=fl(R["I"]),
        power_factor=fl(R["PF"]), thd_v_pct=fl(R["THD"]), meter_event_code="none",
        gt_label="normal", gt_subtype="none", gt_event_id="none", gt_data_trustworthy=1.0, gt_expected_response="none",
        gt_true_energy_kwh=fl(D["E_imp"]), gt_true_voltage_v=fl(D["V"])))
    for c, d_ in METER_DECIMALS.items(): df[c] = df[c].round(d_)
    for c, d_ in FEEDER_DECIMALS.items(): fdf[c] = fdf[c].round(d_)
    return df, fdf, S["md"], weather_table(nd, S["wx"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="clean_dataset")
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    heat = make_heat_episodes(np.random.default_rng(a.seed))
    fds, mds, wxs = [], [], []
    for ni, nd in enumerate(NODES):
        df, fdf, md, wdf = build_clean_node(nd, ni, heat, a.seed)
        df.to_csv(os.path.join(a.out, f"{nd}_meter_readings_clean.csv.gz"), index=False, compression="gzip")
        fds.append(fdf); mds.append(md); wxs.append(wdf)
        print(nd, df.shape)
    pd.concat(fds).to_csv(os.path.join(a.out, "feeder_readings_clean.csv.gz"), index=False, compression="gzip")
    pd.concat(mds).to_csv(os.path.join(a.out, "meters_metadata.csv"), index=False)
    pd.concat(wxs).to_csv(os.path.join(a.out, "weather_calendar.csv.gz"), index=False, compression="gzip")
    print("clean (anomaly-free) dataset written to", a.out)
