"""Frame-level drone-presence (alarm) decision from each camera and from their fusion, in the spirit of the image-level
IR-VL fusion of Aboalia et al. (2025). Presence score of a camera = max calibrated detection score in the frame; label =
drone annotated in either camera. Fused rules: MAX, ProbEn and a logistic regression fitted on validation frames.
Reports AUC, accuracy at the validation-optimal threshold and recall at a 1% / 5% frame false-alarm rate.
python presence_fusion.py PREP RUNS SEED
"""
import json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import antiuav_pipeline as P
from fusion_core import Platt, proben_pair, logit, sigmoid

PREP, RUNS, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
gt, meta = P.load_prepared(PREP)
R = json.load(open(os.path.join(RUNS, f"results_antiuav_s{seed}.json")))
D = pickle.load(open(os.path.join(RUNS, f"detections_s{seed}.pkl"), "rb"))
c = R["calibration"]; prior = c["prior"]
cir, crgb = Platt(c["ir"]["A"], c["ir"]["B"]), Platt(c["rgb"]["A"], c["rgb"]["B"])


def table(sp):
    ks = sorted(k for k, v in meta["split_of"].items() if v == sp)
    ir, rgb = P.apply_cal(D[f"ir_{sp}"], cir), P.apply_cal(D[f"rgb_{sp}"], crgb)
    s = lambda d: float(d[:, 4].max()) if len(d) else 0.0
    pi = np.array([s(ir.get(k, np.zeros((0, 5)))) for k in ks]); pr = np.array([s(rgb.get(k, np.zeros((0, 5)))) for k in ks])
    y = np.array([gt["IR"][k] is not None or gt["RGB"][k] is not None for k in ks]).astype(int)
    br = np.array([meta["scene"][k]["rgb_brightness"] for k in ks])
    return pi, pr, y, br


def auc(s, y):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def recall_at_far(s, y, far):
    neg = np.sort(s[y == 0])[::-1]
    if not len(neg):
        return float("nan")
    t = neg[min(int(np.floor(far * len(neg))), len(neg) - 1)]
    return float((s[y == 1] > t).mean())


vi, vr, vy, vb = table("val"); ti, tr, ty, tb = table("test")
X = lambda a, b, br: np.c_[np.ones(len(a)), logit(np.maximum(a, 1e-3)), logit(np.maximum(b, 1e-3)), br / 255.0,
                           logit(np.maximum(b, 1e-3)) * br / 255.0]
w = np.zeros(5); Xv = X(vi, vr, vb)
for _ in range(5000):
    w -= 0.05 * (Xv.T @ (sigmoid(Xv @ w) - vy) / len(vy) + 1e-3 * w)
rules = {"IR only": (vi, ti), "RGB only": (vr, tr), "MAX": (np.maximum(vi, vr), np.maximum(ti, tr)),
         "ProbEn": (proben_pair(vi, vr, prior), proben_pair(ti, tr, prior)),
         "Learned (cond.-aware)": (sigmoid(Xv @ w), sigmoid(X(ti, tr, tb) @ w))}
out = {"n_test_frames": int(len(ty)), "test_positive_share": float(ty.mean())}
for n, (sv, st) in rules.items():
    th = np.unique(sv); accs = [((sv >= t) == vy).mean() for t in th]; t = th[int(np.argmax(accs))]
    out[n] = {"AUC": auc(st, ty), "accuracy": float(((st >= t) == ty).mean()), "recall@1%FAR": recall_at_far(st, ty, 0.01),
              "recall@5%FAR": recall_at_far(st, ty, 0.05),
              "accuracy_night": float(((st >= t) == ty)[tb < 50].mean()), "accuracy_day": float(((st >= t) == ty)[tb >= 100].mean())}
    print(f"{n:22s}", {k: round(v, 4) for k, v in out[n].items()})
json.dump(out, open(os.path.join(RUNS, f"presence_fusion_s{seed}.json"), "w"), indent=1)
