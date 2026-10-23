"""Image-level (global) fusion in the spirit of the IR-VL model (Aboalia et al., 2025): the RGB camera contributes a
frame-level 'drone present' probability, so no box has to be transferred between the unregistered cameras.
Here the RGB evidence is the maximum calibrated RGB detection score of the frame (decision-level proxy of a global
feature). Variants (IR frame):
  top1   : the top-scoring IR box is re-scored by ProbEn with the RGB presence probability
  all    : every IR box is re-scored
  learned: logistic regression on validation frames over [logit p_ir, logit p_rgb_presence, is_top1, interaction]
python image_level_fusion.py PREP RUNS SEED
"""
import json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import antiuav_pipeline as P
from fusion_core import Platt, proben_pair, logit, sigmoid, iou_matrix
from analyze_antiuav import frame_records, ap_from, bootstrap, seq_pack

PREP, RUNS, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
gt, meta = P.load_prepared(PREP)
R = json.load(open(os.path.join(RUNS, f"results_antiuav_s{seed}.json")))
D = pickle.load(open(os.path.join(RUNS, f"detections_s{seed}.pkl"), "rb"))
cal = R["calibration"]; prior = cal["prior"]
cir, crgb = Platt(cal["ir"]["A"], cal["ir"]["B"]), Platt(cal["rgb"]["A"], cal["rgb"]["B"])
E = np.zeros((0, 5))


def keys(sp):
    return sorted(k for k, v in meta["split_of"].items() if v == sp)


def presence(rgb, k):
    d = rgb.get(k, E)
    return float(d[:, 4].max()) if len(d) else 0.0


def feats(d, pr):
    o = np.argsort(-d[:, 4]); top = np.zeros(len(d)); top[o[0]] = 1
    li, lr = logit(d[:, 4]), logit(np.full(len(d), max(pr, 1e-3)))
    return np.c_[np.ones(len(d)), li, lr, top, li * top, lr * top]


def ap(d, g, kk):
    r = frame_records(d, g, kk)
    return ap_from(np.concatenate([v[0] for v in r.values()]), np.concatenate([v[1] for v in r.values()]), sum(v[2] for v in r.values()))


ir = {sp: P.apply_cal(D[f"ir_{sp}"], cir) for sp in ("val", "test")}
rgb = {sp: P.apply_cal(D[f"rgb_{sp}"], crgb) for sp in ("val", "test")}
# learned rule fitted on validation
X, y = [], []
for k in keys("val"):
    d = ir["val"].get(k, E)
    if not len(d):
        continue
    g = gt["IR"].get(k)
    X.append(feats(d, presence(rgb["val"], k)))
    y.append(iou_matrix(d[:, :4], np.asarray(g, float)[None, :4])[:, 0] >= 0.5 if g is not None else np.zeros(len(d), bool))
X, y = np.concatenate(X), np.concatenate(y).astype(float)
w = np.zeros(X.shape[1])
for _ in range(4000):
    w -= 0.05 * (X.T @ (sigmoid(X @ w) - y) / len(y) + 1e-3 * w)
kt = keys("test"); G = gt["IR"]
out = {"IR only": ir["test"]}
for name in ("top1", "all", "learned"):
    f = {}
    for k in kt:
        d = ir["test"].get(k, E).copy()
        if len(d):
            pr = presence(rgb["test"], k)
            if name == "learned":
                d[:, 4] = sigmoid(feats(d, pr) @ w)
            else:
                idx = [int(np.argmax(d[:, 4]))] if name == "top1" else range(len(d))
                for i in idx:
                    d[i, 4] = proben_pair(d[i, 4], pr, prior) if pr > 0 else d[i, 4] * (1 - prior)
        f[k] = d
    out[name] = f
sf = {}
for k in kt:
    sf.setdefault(P.seq_of(k), []).append(k)
res = {}
for n, d in out.items():
    res[n] = {"AP50": ap(d, G, kt)}
    if n != "IR only":
        res[n]["boot_vs_IR"] = bootstrap([seq_pack(frame_records(d, G, kt), sf)], [seq_pack(frame_records(out["IR only"], G, kt), sf)], sorted(sf), 300)
    print(n, {kk: (round(v, 4) if isinstance(v, float) else {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()}) for kk, v in res[n].items()}, flush=True)
json.dump(res, open(os.path.join(RUNS, f"image_level_fusion_s{seed}.json"), "w"), indent=1)
