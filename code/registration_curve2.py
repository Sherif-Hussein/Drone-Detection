"""Registration-precision analysis, revised (test set, IR frame).

Three ways of degrading the per-frame ground-truth (oracle) registration, each giving the AP50 change of fusion
over the thermal detector alone as a function of the median registration error:
  gauss     isotropic Gaussian offset noise, fine sigma grid, 3 draws per sigma
  empirical the real per-frame residuals of the windowed self-calibrated map (heavy-tailed, temporally correlated),
            scaled by a factor c; c = 1 reproduces the windowed map's error distribution
  coverage  a fraction f of frames gets the oracle registration, the rest the windowed map
Crossing points (where the gain reaches zero) get 95 % intervals from a bootstrap that resamples recording sessions.

python registration_curve2.py PREP RUNS SEED [WINDOW] [N_BOOT]
"""
import json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import antiuav_pipeline as P
from fusion_core import AxisAffine, Platt, fuse_anchored
from analyze_antiuav import frame_records, ap_from, seq_pack, ap_of_seqs, session_of_seq
import registration as REG

PREP, RUNS, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
W = int(sys.argv[4]) if len(sys.argv) > 4 else 3
NB = int(sys.argv[5]) if len(sys.argv) > 5 else 1000
gt, meta = P.load_prepared(PREP)
R = json.load(open(os.path.join(RUNS, f"results_antiuav_s{seed}.json")))
D = pickle.load(open(os.path.join(RUNS, f"detections_s{seed}.pkl"), "rb"))
cal = R["calibration"]; prior = cal["prior"]
cir, crgb = Platt(cal["ir"]["A"], cal["ir"]["B"]), Platt(cal["rgb"]["A"], cal["rgb"]["B"])
E = np.zeros((0, 5))
keys = sorted(k for k, v in meta["split_of"].items() if v == "test")
fo = {s: ks for s, ks in meta["frames_of"].items() if meta["split_of"][ks[0]] == "test"}
seqs = sorted(fo)
smaps = {s: AxisAffine.from_dict(d) for s, d in D["tmaps"].items()}
ir, rgb = P.apply_cal(D["ir_test"], cir), P.apply_cal(D["rgb_test"], crgb)
base_pack = seq_pack(frame_records(ir, gt["IR"], keys), fo)
METHODS = ("box-avg ProbEn", "anchored ProbEn + unmatched RGB")


def packs(fm):
    """Per-sequence record packs of both fusion methods under the per-frame maps fm, plus the median map error."""
    ri = REG.to_frame_pf(rgb, fm, "rgb2ir")
    d = {"box-avg ProbEn": P.fuse_all(ir, ri, keys, "proben", prior),
         "anchored ProbEn + unmatched RGB": {k: fuse_anchored(ir.get(k, E), ri.get(k, E), "proben", prior, gamma=1.0) for k in keys}}
    return {m: seq_pack(frame_records(v, gt["IR"], keys), fo) for m, v in d.items()}, REG.frame_map_error(fm, gt, keys)


def gain(pk, sq):
    return ap_of_seqs(pk, sq) - ap_of_seqs(base_pack, sq)


oracle = REG.oracle_frame_maps(gt, fo, smaps)
win = REG.windowed_maps(D["ir_test"], D["rgb_test"], fo, smaps, window=W)
resid = {k: (win[k].bx - m.bx, win[k].by - m.by) for k, m in oracle.items() if k in win}  # offset residual per frame

curves = {"gauss": [], "empirical": [], "coverage": []}
# each entry: (x, err_px, [packs per draw])
for sigma in np.arange(0, 40.1, 2.5):
    draws, errs = [], []
    for rep in range(3 if sigma else 1):
        rng = np.random.default_rng(100 + rep)
        fm = {k: AxisAffine(m.ax, m.bx + rng.normal(0, sigma), m.ay, m.by + rng.normal(0, sigma)) for k, m in oracle.items()}
        pk, e = packs(fm); draws.append(pk); errs.append(e)
    curves["gauss"].append((float(sigma), float(np.mean(errs)), draws))
    print("gauss", sigma, round(np.mean(errs), 1), {m: round(np.mean([gain(d[m], seqs) for d in draws]), 4) for m in METHODS}, flush=True)
for c in np.arange(0, 1.51, 0.1):
    fm = {k: AxisAffine(m.ax, m.bx + c * resid.get(k, (0, 0))[0], m.ay, m.by + c * resid.get(k, (0, 0))[1]) for k, m in oracle.items()}
    pk, e = packs(fm)
    curves["empirical"].append((float(c), float(e), [pk]))
    print("empirical", round(c, 1), round(e, 1), {m: round(gain(pk[m], seqs), 4) for m in METHODS}, flush=True)
for f in np.arange(0, 1.01, 0.1):
    draws, errs = [], []
    for rep in range(3 if 0 < f < 1 else 1):
        rng = np.random.default_rng(200 + rep)
        fm = {k: (oracle[k] if rng.random() < f else win.get(k, oracle[k])) for k in oracle}
        pk, e = packs(fm); draws.append(pk); errs.append(e)
    curves["coverage"].append((float(f), float(np.mean(errs)), draws))
    print("coverage", round(f, 1), round(np.mean(errs), 1), {m: round(np.mean([gain(d[m], seqs) for d in draws]), 4) for m in METHODS}, flush=True)


def crossing(xs, gs):
    """First x at which the gain falls from >0 to <=0 (linear interpolation); None if it never does."""
    for i in range(1, len(xs)):
        if gs[i - 1] > 0 >= gs[i]:
            return float(xs[i - 1] + (xs[i] - xs[i - 1]) * gs[i - 1] / (gs[i - 1] - gs[i]))
    return None


def rising(xs, gs):
    """First x at which the gain rises from <=0 to >0 (for the coverage curve)."""
    for i in range(1, len(xs)):
        if gs[i - 1] <= 0 < gs[i]:
            return float(xs[i - 1] + (xs[i] - xs[i - 1]) * (-gs[i - 1]) / (gs[i] - gs[i - 1]))
    return None


cl = {}
for s in seqs:
    cl.setdefault(session_of_seq(s), []).append(s)
names = sorted(cl)
rng = np.random.default_rng(seed)
picks = [seqs] + [[s for i in rng.integers(0, len(names), len(names)) for s in cl[names[i]]] for _ in range(NB)]
out = {"seed": seed, "window": W, "n_sessions": len(names), "base_ir_ap50": ap_of_seqs(base_pack, seqs), "curves": {}, "crossings": {}}
for name, rows in curves.items():
    out["curves"][name] = [{"x": x, "map_err_px": e, **{m: float(np.mean([gain(d[m], seqs) for d in ds])) for m in METHODS}}
                           for x, e, ds in rows]
    out["crossings"][name] = {}
    for m in METHODS:
        vals = []
        for sq in picks:
            gs = [np.mean([gain(d[m], sq) for d in ds]) for _, _, ds in rows]
            xs = [e for _, e, _ in rows] if name != "coverage" else [x for x, _, _ in rows]
            vals.append(rising(xs, gs) if name == "coverage" else crossing(xs, gs))
        pt, bs = vals[0], [v for v in vals[1:] if v is not None]
        out["crossings"][name][m] = {"point": pt, "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if bs else None,
                                     "frac_resamples_with_crossing": len(bs) / NB,
                                     "x": "median error (RGB px)" if name != "coverage" else "fraction of frames with exact registration"}
        print("crossing", name, m, out["crossings"][name][m], flush=True)
out["windowed_median_err_px"] = REG.frame_map_error(win, gt, keys)
json.dump(out, open(os.path.join(RUNS, f"registration_curve2_s{seed}.json"), "w"), indent=1)
print("saved")
