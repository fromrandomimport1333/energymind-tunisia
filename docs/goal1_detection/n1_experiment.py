"""Section 5 of the Goal 1 report: the detectors run on node N1 of the EnergyMind dataset.

Unit of analysis: one meter-day (48 half-hourly readings). 30 meters x 365 days.
Every detector is trained on clean meter-days only (all 48 rows gt_label == "normal");
labels are used only to score. Out-of-fold: 5 folds, clean days grouped by meter-week,
anomalous days grouped by gt_event_id, so no event is split across folds.

Run:  python n1_experiment.py [path/to/energymind_full_joined.csv.gz]
Writes data/n1_numbers.tex, data/n1_class_rows.tex, data/n1_event_rows.tex
"""
import os
import sys
import zlib

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.svm import OneClassSVM

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")
CANDIDATES = [os.path.join(HERE, "..", "..", "data", "energymind_full_joined.csv.gz"),            # inside the repo
              os.path.join(HERE, "..", "energymind-tunisia", "data", "energymind_full_joined.csv.gz")]
DEFAULT = next((c for c in CANDIDATES if os.path.exists(c)), CANDIDATES[0])
NODE, K, SEED, HIST = "N1", 5, 2025, 28          # 28-day trailing history for profiles
FPR_SINGLE, FPR_ENS = 0.01, 0.0025

COLS = ["timestamp", "node_id", "feeder_id", "meter_id", "energy_kwh", "pv_export_kwh", "voltage_v",
        "voltage_min_v", "voltage_max_v", "current_a", "power_factor", "thd_v_pct",
        "feeder_voltage_v", "energy_balance_gap_pct", "temp_c", "is_weekend",
        "gt_label", "gt_subtype", "gt_event_id"]

# ------------------------------------------------------------------ load N1 as (meter, day, slot) cubes
path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
df = pd.read_csv(path, usecols=COLS, parse_dates=["timestamp"], low_memory=False)
df = df[df.node_id == NODE].sort_values(["meter_id", "timestamp"]).reset_index(drop=True)
meters = df.meter_id.unique()
days = np.sort(df.timestamp.dt.floor("D").unique())
M, D, S = len(meters), len(days), 48
assert len(df) == M * D * S, "expected a complete meter x half-hour grid"
cube = lambda c: df[c].to_numpy(float).reshape(M, D, S)
E, X, V, VMIN, VMAX, I, PF, THD, FV, GAP, TEMP = map(cube, [
    "energy_kwh", "pv_export_kwh", "voltage_v", "voltage_min_v", "voltage_max_v", "current_a",
    "power_factor", "thd_v_pct", "feeder_voltage_v", "energy_balance_gap_pct", "temp_c"])
X = np.nan_to_num(X)
WKEND = cube("is_weekend")[:, :, 0]
feeder_of = df.groupby("meter_id", sort=False).feeder_id.first().to_numpy()
LAB = df.gt_label.to_numpy().reshape(M, D, S)
SUB = df.gt_subtype.to_numpy().reshape(M, D, S)
EVT = df.gt_event_id.to_numpy().reshape(M, D, S)
print(f"{NODE}: {M} meters x {D} days")

# ------------------------------------------------------------------ window labels
nan_frac = np.isnan(E).mean(axis=2)
abn = LAB != "normal"
clean = ~abn.any(axis=2)
day_sub = np.full((M, D), "none", dtype=object)
day_evt = np.full((M, D), "none", dtype=object)
day_lab = np.full((M, D), "normal", dtype=object)
for m, d in zip(*np.where(~clean)):
    s = pd.Series(SUB[m, d][abn[m, d]])
    top = s.value_counts().index[0]
    k = np.where(SUB[m, d] == top)[0][0]
    day_sub[m, d], day_evt[m, d], day_lab[m, d] = top, EVT[m, d, k], LAB[m, d, k]

# ------------------------------------------------------------------ features (no labels used)
def trailing_median(a, n=HIST):
    """For every day d: median over days d-n..d-1 (NaN for the first week)."""
    out = np.full(a.shape, np.nan)
    for d in range(7, a.shape[1]):
        out[:, d] = np.nanmedian(a[:, max(0, d - n):d], axis=1)
    return out

with np.errstate(all="ignore"):
    P = trailing_median(E)                                    # per-slot profile (M, D, S)
    r = np.log((E + 0.02) / (P + 0.02))
    eday = np.nansum(E, axis=2)
    eday[nan_frac > 0.5] = np.nan
    med_nb = np.full((M, D), np.nan)
    for f in np.unique(feeder_of):
        idx = np.where(feeder_of == f)[0]
        for m in idx:
            med_nb[m] = np.nanmedian(eday[[j for j in idx if j != m]], axis=0)
    q = np.log((eday + 0.1) / (med_nb + 0.1))                 # meter vs its feeder neighbours
    q_dev = q - trailing_median(q[:, :, None])[:, :, 0]
    expct = V * I * PF * 0.5 / 1000
    ok = (V > 1) & (X == 0) & (expct > 0.02)
    ratio = np.where(ok, E / expct, np.nan)
    phys_med = np.nanmedian(np.log(np.abs(ratio) + 1e-3), axis=2)
    phys_frac = np.nansum(np.abs(ratio - 1) > 0.05, axis=2) / np.maximum(ok.sum(axis=2), 1)
    dv = np.where((X == 0) & (V > 1), V - FV, np.nan)
    dv_med = np.nanmedian(dv, axis=2)
    dv_dev = dv_med - trailing_median(dv_med[:, :, None])[:, :, 0]
    v0_share = np.nanmean(V <= 1, axis=2)
    zero_share = np.nanmean((E == 0) & (V > 1) & (X == 0), axis=2)
    same = (np.diff(E, axis=2) == 0) & (np.diff(V, axis=2) == 0)   # a stuck meter freezes every channel
    run = np.zeros(same.shape)
    for s in range(same.shape[2]):
        run[:, :, s] = np.where(same[:, :, s], (run[:, :, s - 1] if s else 0) + 1, 0)
    repeat_max = run.max(axis=2) * (np.nanmean(V, axis=2) > 1)
    impossible = (np.nanmin(E, axis=2) < 0) | (np.nanmax(E, axis=2) >= 50)

    def corr(a, b):
        a = a - np.nanmean(a, axis=-1, keepdims=True); b = b - np.nanmean(b, axis=-1, keepdims=True)
        return np.nansum(a * b, axis=-1) / np.sqrt(np.nansum(a * a, axis=-1) * np.nansum(b * b, axis=-1))
    c0 = corr(E, P)
    lag_gain = np.nanmax(np.stack([corr(E[:, :, L:], P[:, :, :-L]) for L in (2, 4, 6)]), axis=0) - c0
    prev = np.concatenate([np.full((M, 7, S), np.nan), E[:, :-7]], axis=1)
    week_diff = np.nanmean(np.abs(E - prev), axis=2) / (np.nanmean(E, axis=2) + 0.01)
    feats = {
        "r_mean": np.nanmean(r, axis=2), "r_absmax": np.nanmax(np.abs(r), axis=2),
        "r_std": np.nanstd(r, axis=2), "q_dev": q_dev, "phys_med": phys_med, "phys_frac": phys_frac,
        "dv_med": dv_med, "dv_dev": dv_dev, "v0_share": v0_share, "zero_share": zero_share,
        "repeat_max": repeat_max, "impossible": impossible.astype(float), "lag_gain": lag_gain,
        "week_diff": week_diff, "thd_max": np.nanmax(THD, axis=2),
        "vmin_gap": np.nanmin(VMIN - FV, axis=2), "vmax_gap": np.nanmax(VMAX - FV, axis=2),
        "temp": np.nanmean(TEMP, axis=2), "weekend": WKEND,
    }

names = list(feats)
F = np.stack([feats[n] for n in names], axis=-1)              # (M, D, n_features)
use = (np.arange(D)[None, :] >= 7) & (nan_frac <= 0.5)        # profiles need a week; skip no-data days
use &= day_lab != "comm_loss"

# ------------------------------------------------------------------ folds
fold = np.zeros((M, D), int)
week = pd.DatetimeIndex(days).isocalendar().week.to_numpy()
for m in range(M):
    for d in range(D):
        key = f"{meters[m]}|{week[d]}" if clean[m, d] else str(day_evt[m, d])
        fold[m, d] = zlib.crc32(key.encode()) % K           # deterministic

# ------------------------------------------------------------------ rules (layer 1, fixed thresholds)
rules = (impossible | (phys_frac > 0.25) | (dv_med > 1.5) | (v0_share > 0) | (repeat_max >= 6))
rules &= use

# ------------------------------------------------------------------ learned detectors, out of fold
RESID = [names.index(n) for n in ["r_mean", "r_absmax", "q_dev", "phys_med", "dv_med", "dv_dev"]]
scores = {k: np.full((M, D), np.nan) for k in ["Robust z", "CUSUM", "Isolation Forest", "One-Class SVM", "Autoencoder"]}
thr = {k: np.zeros(K) for k in scores}
thr_e = {k: np.zeros(K) for k in scores}
rng = np.random.default_rng(SEED)


def cusum(z, k=0.5):
    """Two-sided CUSUM along days on a clipped standardized series; returns (S+, S-)."""
    z = np.clip(np.nan_to_num(z), -4, 4)
    sp = np.zeros_like(z); sm = np.zeros_like(z)
    for d in range(1, z.shape[1]):
        sp[:, d] = np.maximum(0, sp[:, d - 1] + z[:, d] - k)
        sm[:, d] = np.maximum(0, sm[:, d - 1] - z[:, d] - k)
    return sp, sm


# feeder daily energy-balance gap, CUSUM with a seasonal reference that freezes during alarms.
# The gap is seasonal (PV export), so the reference is the median of the last 28 non-alarm days.
# After an alarm the CUSUM restarts; while alarms keep coming (gap <= 7 days) the reference stays
# frozen, for at most 28 days, so a persistent bias re-alarms instead of being absorbed.
# h is chosen on clean feeder-days only (textbook h=5 alarms on ~45% of them: the gap is noisy).
feeders = np.unique(feeder_of)
fgap = np.stack([np.nanmedian(GAP[np.where(feeder_of == f)[0][0]], axis=1) for f in feeders])
fclean = np.stack([clean[feeder_of == f].all(axis=0) for f in feeders])
H_BAL, K_BAL, QUIET, MAX_FREEZE = 40.0, 1.0, 7, 28   # h set for ~5% alarm time on clean feeder-days
balance_flag = np.zeros((M, D), bool)                         # attributed meter, whole episode
f_alarm = np.zeros(fgap.shape, bool)                          # feeder alarm days
f_episode = np.zeros(fgap.shape, bool)
for fi, f in enumerate(feeders):
    idx = np.where(feeder_of == f)[0]
    sd = np.nanmedian(np.abs(np.diff(fgap[fi]))) * 1.4826 / np.sqrt(2)   # day-to-day noise
    pool, qpool = [], []
    sp = sm = 0.0
    last_alarm, freeze_start, suspect = -99, None, None
    for d in range(D):
        g = fgap[fi, d]
        if len(pool) >= 7 and not np.isnan(g):
            z = np.clip((g - np.median(pool[-HIST:])) / sd, -4, 4)
            sp, sm = max(0.0, sp + z - K_BAL), max(0.0, sm - z - K_BAL)
        if max(sp, sm) > H_BAL:
            f_alarm[fi, d] = True
            # attribution: meter whose share moved most in the direction that explains the gap
            ref = np.nanmedian(np.array(qpool[-HIST:]), axis=0)
            recent = np.nanmean(q[idx, max(0, d - 6):d + 1], axis=1)
            expl = (ref - recent) if sp >= sm else (recent - ref)   # gap up -> a meter reads low
            suspect = idx[np.nanargmax(np.nan_to_num(expl, nan=-9))]
            last_alarm, sp, sm = d, 0.0, 0.0
            freeze_start = d if freeze_start is None else freeze_start
        in_episode = d - last_alarm <= QUIET
        if in_episode and suspect is not None:
            f_episode[fi, d] = True
            balance_flag[suspect, d] = True
        frozen = in_episode and freeze_start is not None and d - freeze_start < MAX_FREEZE
        if not in_episode:
            freeze_start = None
        if not frozen and not np.isnan(g):
            pool.append(g); qpool.append(q[idx, d])


for k in range(K):
    tr = use & clean & (fold != k)
    te = use & (fold == k)
    trm = np.where(tr)
    # 70% of the training clean days fit the models, 30% set the thresholds
    perm = rng.permutation(len(trm[0]))
    cut = int(0.7 * len(perm))
    fit_i = (trm[0][perm[:cut]], trm[1][perm[:cut]])
    cal_i = (trm[0][perm[cut:]], trm[1][perm[cut:]])
    med = np.nanmedian(F[tr], axis=0)
    mad = np.nanmedian(np.abs(F[tr] - med), axis=0) * 1.4826 + 1e-6
    Z = np.clip(np.nan_to_num((F - med) / mad), -20, 20)    # robust scaling, NaN -> median

    s_z = np.abs(Z[:, :, RESID]).max(axis=-1)
    s_c = np.maximum(*cusum(Z[:, :, names.index("dv_dev")]))   # meter voltage vs feeder head
    Xfit = Z[fit_i]
    iforest = IsolationForest(n_estimators=200, random_state=SEED).fit(Xfit)
    sub = Xfit[rng.choice(len(Xfit), min(3000, len(Xfit)), replace=False)]
    svm = OneClassSVM(nu=0.01, gamma="scale").fit(sub)
    ae = MLPRegressor(hidden_layer_sizes=(12, 4, 12), max_iter=400, random_state=SEED).fit(Xfit, Xfit)
    flat = Z.reshape(-1, Z.shape[-1])
    s_if = -iforest.score_samples(flat).reshape(M, D)
    s_svm = -svm.decision_function(flat).reshape(M, D)
    s_ae = ((ae.predict(flat) - flat) ** 2).sum(axis=1).reshape(M, D)
    for name, s in zip(scores, [s_z, s_c, s_if, s_svm, s_ae]):
        scores[name][te] = s[te]
        thr[name][k] = np.quantile(s[cal_i], 1 - FPR_SINGLE)
        thr_e[name][k] = np.quantile(s[cal_i], 1 - FPR_ENS)
    print(f"fold {k} done")

flags = {"Rules (layer 1)": rules}
ens = np.zeros((M, D), bool)
for name in scores:
    t1 = thr[name][fold]; te_ = thr_e[name][fold]
    flags[name] = use & (scores[name] > t1)
    ens |= use & (scores[name] > te_)
flags["CUSUM"] = flags["CUSUM"] | (use & balance_flag)        # balance CUSUM + attribution
ens |= use & balance_flag
flags["Detection stack"] = rules | ens

# ------------------------------------------------------------------ metrics
cls = {"normal": "NORMAL", "grid_event": "GENUINE", "equipment_failure": "GENUINE",
       "sensor_fault": "METER", "data_corruption": "METER", "falsification": "SUSPICIOUS"}
day_cls = np.vectorize(lambda l: cls.get(l, "NO_DATA"))(day_lab)
ev_ids = sorted(set(day_evt[use & ~clean]) - {"none"})
ev_info = {}
for e in ev_ids:
    mask = use & (day_evt == e)
    dd = np.where(mask.any(axis=0))[0]
    ev_info[e] = dict(cls=day_cls[mask][0], sub=day_sub[mask][0], start=dd.min(), end=dd.max(),
                      feeder=feeder_of[np.where(mask.any(axis=1))[0][0]], mask=mask)


def share(flag, e):
    """Fraction of the event's meter-days that are flagged."""
    return flag[ev_info[e]["mask"]].mean()


def detected(flag, e):
    """An event counts as detected when the flag is a reliable signal, not a chance hit:
    >= 20% of its meter-days flagged (any flag for events of 5 days or less)."""
    i = ev_info[e]
    short = i["end"] - i["start"] < 5
    return (flag & i["mask"]).any() if short else share(flag, e) >= 0.2


def first_flag(flag, e):
    hit = (flag & ev_info[e]["mask"]).any(axis=0)
    return int(np.where(hit)[0].min() - ev_info[e]["start"]) if hit.any() else None


def pct(x):
    return f"{100 * x:.0f}"


rows = []
for name, fl in flags.items():
    fpr = fl[use & clean].mean()
    cells = []
    for c in ["GENUINE", "METER", "SUSPICIOUS"]:
        evs = [e for e in ev_ids if ev_info[e]["cls"] == c]
        n_det = sum(bool(detected(fl, e)) for e in evs)
        win = fl[use & (day_cls == c)].mean()
        cells += [rf"\hc{{{pct(win)}}}", f"{n_det}/{len(evs)}"]
    rows.append(f"{name} & " + " & ".join(cells) + f" & {100 * fpr:.1f}\\% \\\\")
with open(os.path.join(OUT, "n1_class_rows.tex"), "w") as fh:
    fh.write("% generated by n1_experiment.py\n" + "\n".join(rows) + "\n\\bottomrule\n")

# feeder-level energy balance on its own: share of event days whose daily gap leaves the
# feeder's clean 1-99% band (on clean feeder-days this is 2% by construction)
gap_sd = np.mean([np.nanstd(fgap[fi][fclean[fi]]) for fi in range(len(feeders))])
f_fpr = f_episode[fclean].mean()                              # feeder in alarm on clean feeder-days

SHORT = {"Rules (layer 1)": "R", "Robust z": "z", "CUSUM": "C", "Isolation Forest": "IF",
         "One-Class SVM": "SVM", "Autoencoder": "AE"}
RARE = ["partial_reduction_bypass", "selective_bypass", "meter_zeroing", "fdi_scale", "fdi_evasion",
        "fdi_inflate", "fdi_replay", "coordinated_fdi", "bias_step", "gain_drift", "clock_shift",
        "voltage_offset_drift", "stuck_value"]
erows = []
for e in sorted([e for e in ev_ids if ev_info[e]["sub"] in RARE],
                key=lambda e: (RARE.index(ev_info[e]["sub"]), ev_info[e]["start"])):
    i = ev_info[e]
    by = [SHORT[n] for n in SHORT if detected(flags[n], e)]
    st = flags["Detection stack"]
    delay = first_flag(st, e)
    fi = list(feeders).index(i["feeder"])
    hit = np.where(f_alarm[fi][i["start"]:i["end"] + 1])[0]
    bal = f"day {hit[0]}" if hit.size else "--"              # first feeder balance alarm
    start = pd.Timestamp(days[i["start"]]).strftime("%d %b")
    sub = i["sub"].replace("_", r"\_")
    erows.append(f"\\texttt{{{sub}}} & {i['feeder'][-2:]} & {start} & {i['end'] - i['start'] + 1} & "
                 f"{', '.join(by) if by else '--'} & {('day ' + str(delay)) if delay is not None else '--'} & "
                 f"{pct(share(st, e))}\\% & {bal} \\\\")
with open(os.path.join(OUT, "n1_event_rows.tex"), "w") as fh:
    fh.write("% generated by n1_experiment.py\n" + "\n".join(erows) + "\n\\bottomrule\n")

stack = flags["Detection stack"]
num = {
    "nOneDays": f"{int(use.sum()):,}".replace(",", "{,}"),
    "nOneClean": f"{int((use & clean).sum()):,}".replace(",", "{,}"),
    "nOneEvents": str(len(ev_ids)),
    "stackFPR": f"{100 * stack[use & clean].mean():.1f}",
    "gapSd": f"{gap_sd:.1f}",
    "balFPR": f"{100 * f_fpr:.1f}",
    "nRare": str(len(erows)),
    "nRareHit": str(sum(bool(detected(stack, e)) for e in ev_ids if ev_info[e]["sub"] in RARE)),
}
with open(os.path.join(OUT, "n1_numbers.tex"), "w") as fh:
    fh.write("% generated by n1_experiment.py -- do not edit\n")
    for k_, v in num.items():
        fh.write(f"\\newcommand{{\\{k_}}}{{{v}}}\n")
print(num)
print("\n".join(rows)); print("\n".join(erows))
