"""Generate the data files behind the illustrative plots (Figures 2, 4, 5, 6) of the Goal 1 report.

The results on the challenge dataset (Section 5) come from n1_experiment.py.

All data are synthetic, with fixed seeds.  Run:  python gen.py   (needs numpy, scikit-learn)
Outputs go to ./data ; data/numbers.tex holds the numbers quoted in the text and captions.
"""
import os
import numpy as np
from PIL import Image
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT, exist_ok=True)
numbers = {}


def save(name, header, *cols):
    arr = np.column_stack(cols)
    np.savetxt(os.path.join(OUT, name), arr, fmt="%.4f", header=" ".join(header), comments="")


def nan_except(x, mask):
    y = np.full_like(x, np.nan)
    y[mask] = x[mask]
    return y


# ---------------------------------------------------------------- Figure 2: anomaly types
rng = np.random.default_rng(2)
t = np.arange(0, 72.01, 0.5)                        # 30-min samples over 3 days (as in the dataset)
base = 230 + 3.0 * np.sin(2 * np.pi * (t - 6) / 24) + rng.normal(0, 0.45, t.size)

# point: one impossible value at 36 h
pt = np.isclose(t, 36.0)
v_point = base.copy(); v_point[pt] = 246.5
save("types_point.dat", ["t", "v", "anom", "ref"], t, base, nan_except(v_point, pt), nan_except(base, pt))

# contextual: the day-time level appears at night (27 h)
ctx = np.isclose(t, 27.0)
v_ctx = base.copy(); v_ctx[ctx] = 232.0
save("types_context.dat", ["t", "v", "anom", "ref"], t, np.where(ctx, np.nan, base),
     nan_except(v_ctx, ctx), nan_except(base, ctx))

# collective, stuck: frozen value between 48 h and 56 h
st = (t >= 48) & (t <= 56)
v_st = base.copy(); v_st[st] = np.where(t[st] < 53, 229.2, 230.6)
save("types_stuck.dat", ["t", "v", "anom"], t, np.where(st, np.nan, base), nan_except(v_st, st))

# collective, drift: slow gain error from 24 h on
dr = t >= 24
v_dr = base + np.where(dr, (t - 24) * 0.16, 0.0)
save("types_drift.dat", ["t", "v", "anom"], t, base, nan_except(v_dr, dr))

# ---------------------------------------------------------------- Figure 4 left: masking
rng = np.random.default_rng(4)
normal = rng.normal(0, 1, 60)
outliers = np.array([4.5, 8.3, 10.6, 12.9])
x = np.concatenate([normal, outliers])
z_ord = np.abs((x - x.mean()) / x.std(ddof=1))
mad = np.median(np.abs(x - np.median(x)))
z_rob = np.abs((x - np.median(x)) / (1.4826 * mad))
idx = np.arange(x.size)
save("mask_normal.dat", ["i", "z"], idx[:60], z_ord[:60])
save("mask_out.dat", ["i", "zord", "zrob"], idx[60:], z_ord[60:], z_rob[60:])
numbers["zOrd"] = ", ".join(f"{v:.1f}" for v in z_ord[60:])
numbers["zRob"] = ", ".join(f"{v:.1f}" for v in z_rob[60:])
numbers["zOrdHidden"] = str(int((z_ord[60:] < 3).sum()))

# ---------------------------------------------------------------- Figure 4 right: CUSUM
k, h, shift_at = 0.5, 4.0, 80
for seed in range(1000):
    rng = np.random.default_rng(seed)
    sig = rng.normal(0, 1, 160)
    sig[shift_at:] += 1.0
    S = np.zeros_like(sig)
    for i in range(1, sig.size):
        S[i] = max(0.0, S[i - 1] + sig[i] - k)
    alarms = np.where(S > h)[0]
    if alarms.size and alarms[0] == 84 and np.abs(sig[shift_at:]).mean() < 1.6:
        break
alarm = int(alarms[0])
save("cusum.dat", ["i", "x", "S"], np.arange(sig.size), sig, np.minimum(S, 13))
save("cusum_alarm.dat", ["i", "S"], np.array([alarm]), np.array([S[alarm]]))
numbers["cusumAlarm"] = str(alarm)
numbers["cusumDelay"] = str(alarm - shift_at)

# ---------------------------------------------------------------- Figures 5 and 6: IF and OCSVM
rng = np.random.default_rng(6)
cov = np.array([[2.0, 0.75], [0.75, 0.6]])
Xn = rng.multivariate_normal([0, 0], cov, 300)
Xo = np.array([[-3.9, 3.5], [0.2, 4.7], [4.2, 3.8], [-4.0, -3.6], [2.5, -4.2], [3.9, -3.3]])
iforest = IsolationForest(n_estimators=200, max_samples=256, random_state=0).fit(Xn)
ocsvm = OneClassSVM(nu=0.05, kernel="rbf", gamma=0.35).fit(Xn)

gx, gy = np.linspace(-6, 6, 97), np.linspace(-5.5, 5.5, 89)
GX, GY = np.meshgrid(gx, gy)
G = np.column_stack([GX.ravel(), GY.ravel()])
s_grid = -iforest.score_samples(G)
f_grid = ocsvm.decision_function(G)
f_grid = np.clip(f_grid / np.abs(f_grid).max(), -1, 1)


def heatmap(values, lo, hi, stops, name):
    """Write a grid of values as a PNG (top row = highest y) using a piecewise-linear colormap."""
    v = np.clip((values.reshape(GX.shape) - lo) / (hi - lo), 0, 1)[::-1]
    stops = np.array(stops, float)
    pos = np.linspace(0, 1, len(stops))
    rgb = np.stack([np.interp(v, pos, stops[:, c]) for c in range(3)], axis=-1)
    img = Image.fromarray(rgb.astype(np.uint8)).resize((970, 890), Image.BICUBIC)
    img.save(os.path.join(OUT, name))


IFMAP = [(255, 248, 238), (245, 190, 120), (230, 135, 30), (200, 80, 30), (150, 40, 30)]
SVMMAP = [(150, 40, 30), (200, 80, 70), (255, 255, 255), (120, 200, 185), (15, 118, 110)]
s_lo, s_hi = np.round(s_grid.min(), 2), np.round(s_grid.max(), 2)
heatmap(s_grid, s_lo, s_hi, IFMAP, "if_map.png")
heatmap(f_grid, -1, 1, SVMMAP, "svm_map.png")
save("cloud.dat", ["x", "y"], Xn[:, 0], Xn[:, 1])
save("outliers.dat", ["x", "y"], Xo[:, 0], Xo[:, 1])
s_out, s_norm = -iforest.score_samples(Xo), -iforest.score_samples(Xn)
numbers["ifOutMin"] = f"{s_out.min():.2f}"
numbers["ifOutMax"] = f"{s_out.max():.2f}"
numbers["ifNormMean"] = f"{s_norm.mean():.2f}"
numbers["ifSMin"] = f"{s_lo:.2f}"
numbers["ifSMax"] = f"{s_hi:.2f}"


def c_n(n):
    return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n


# Figure 5: path lengths of 250 normal points and 250 anomalies drawn outside the dense core
Xn5 = rng.multivariate_normal([0, 0], cov, 250)
Xa5 = []
inv = np.linalg.inv(cov)
while len(Xa5) < 250:
    p = rng.uniform([-6, -5], [6, 5])
    if p @ inv @ p > 16:                      # Mahalanobis distance > 4
        Xa5.append(p)
Xa5 = np.array(Xa5)
if5 = IsolationForest(n_estimators=200, max_samples=250, random_state=0).fit(Xn5)
cpsi = c_n(250)
path = lambda X: -np.log2(-if5.score_samples(X)) * cpsi   # E[h(x)] from s = 2^(-E[h]/c)
pn, pa = path(Xn5), path(Xa5)
edges = np.arange(2.0, 14.51, 0.5)
dn, _ = np.histogram(pn, edges, density=True)
da, _ = np.histogram(pa, edges, density=True)
save("pathlen.dat", ["mid", "normal", "anom"], (edges[:-1] + edges[1:]) / 2, dn, da)
numbers["pathNormal"] = f"{pn.mean():.1f}"
numbers["pathAnom"] = f"{pa.mean():.1f}"

with open(os.path.join(OUT, "numbers.tex"), "w") as fh:
    fh.write("% generated by gen.py -- do not edit\n")
    for key, val in numbers.items():
        fh.write(f"\\newcommand{{\\{key}}}{{{val}}}\n")
print(numbers)
