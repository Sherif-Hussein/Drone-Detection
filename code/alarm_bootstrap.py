#!/usr/bin/env python3
"""
Frame-level alarm ("is a drone in view?") from each camera and from three fusion rules, for all seeds, with
session-level bootstrap intervals (same definitions as presence_fusion.py; the learned rule is fitted on validation).
Statistic per resample: mean over seeds of the AUC (and of the AUC difference to the thermal camera).

python alarm_bootstrap.py PREP RUNS OUT.json [N_BOOT]
"""
import json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import antiuav_pipeline as P
from fusion_core import Platt, proben_pair, logit, sigmoid
from analyze_antiuav import session_of_seq

PREP, RUNS, OUT = sys.argv[1:4]
NB = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
gt, meta = P.load_prepared(PREP)
E = np.zeros((0, 5))


def auc(s, y):
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def table(D, cir, crgb, sp):
    ks = sorted(k for k, v in meta["split_of"].items() if v == sp)
    ir, rgb = P.apply_cal(D[f"ir_{sp}"], cir), P.apply_cal(D[f"rgb_{sp}"], crgb)
    s = lambda d: float(d[:, 4].max()) if len(d) else 0.0
    pi = np.array([s(ir.get(k, E)) for k in ks]); pr = np.array([s(rgb.get(k, E)) for k in ks])
    y = np.array([gt["IR"][k] is not None or gt["RGB"][k] is not None for k in ks]).astype(int)
    br = np.array([meta["scene"][k]["rgb_brightness"] for k in ks])
    return ks, pi, pr, y, br


scores, y_test, keys = {}, None, None
for seed in (0, 1, 2):
    R = json.load(open(os.path.join(RUNS, f"results_antiuav_s{seed}.json")))
    D = pickle.load(open(os.path.join(RUNS, f"detections_s{seed}.pkl"), "rb"))
    c = R["calibration"]; prior = c["prior"]
    cir, crgb = Platt(c["ir"]["A"], c["ir"]["B"]), Platt(c["rgb"]["A"], c["rgb"]["B"])
    _, vi, vr, vy, vb = table(D, cir, crgb, "val")
    keys, ti, tr, ty, tb = table(D, cir, crgb, "test")
    X = lambda a, b, br: np.c_[np.ones(len(a)), logit(np.maximum(a, 1e-3)), logit(np.maximum(b, 1e-3)), br / 255.0,
                               logit(np.maximum(b, 1e-3)) * br / 255.0]
    w = np.zeros(5); Xv = X(vi, vr, vb)
    for _ in range(5000):
        w -= 0.05 * (Xv.T @ (sigmoid(Xv @ w) - vy) / len(vy) + 1e-3 * w)
    rules = {"Thermal only": ti, "RGB only": tr, "MAX": np.maximum(ti, tr), "ProbEn": proben_pair(ti, tr, prior),
             "Condition-aware (learned)": sigmoid(X(ti, tr, tb) @ w)}
    for n, v in rules.items():
        scores.setdefault(n, []).append(v)
    y_test = ty

sess = np.array([session_of_seq(P.seq_of(k)) for k in keys])
names = sorted(set(sess))
idx = {s: np.flatnonzero(sess == s) for s in names}
rng = np.random.default_rng(0)
point = {n: float(np.mean([auc(v, y_test) for v in vs])) for n, vs in scores.items()}
boot = {n: [] for n in scores}; dboot = {n: [] for n in scores if n != "Thermal only"}
for _ in range(NB):
    pick = np.concatenate([idx[names[i]] for i in rng.integers(0, len(names), len(names))])
    yy = y_test[pick]
    if yy.sum() in (0, len(yy)):
        continue
    a = {n: np.mean([auc(v[pick], yy) for v in vs]) for n, vs in scores.items()}
    for n in scores:
        boot[n].append(a[n])
    for n in dboot:
        dboot[n].append(a[n] - a["Thermal only"])
res = {"n_test_frames": int(len(y_test)), "n_drone_free_frames": int((y_test == 0).sum()),
       "drone_free_frames_per_session": {s: int((y_test[idx[s]] == 0).sum()) for s in names},
       "n_sessions": len(names), "seeds": [0, 1, 2], "valid_resamples": len(boot["Thermal only"]), "AUC": {}, "AUC_minus_thermal": {}}
for n in scores:
    b = np.array(boot[n]); res["AUC"][n] = {"mean_over_seeds": point[n], "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                                            "per_seed": [auc(v, y_test) for v in scores[n]]}
for n in dboot:
    b = np.array(dboot[n]); d = point[n] - point["Thermal only"]
    res["AUC_minus_thermal"][n] = {"diff": d, "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                                   "p_two_sided": float(min(1.0, 2 * min(np.mean(b <= 0), np.mean(b >= 0))))}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
