"""Fusion gain as a function of cross-camera registration error (test set, IR frame).

Perfect per-frame registration (ground-truth centres) is degraded with Gaussian offset noise of standard deviation
sigma (RGB pixels, per axis); the curve shows how precise registration must be before fusion beats the thermal
detector alone. The real registration methods (sequence map, windowed map) are added as points.

python registration_curve.py PREP RUNS SEED [WINDOW]
"""
import json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import antiuav_pipeline as P
from fusion_core import AxisAffine, Platt, fuse_anchored
from analyze_antiuav import frame_records, ap_from
import registration as REG

PREP, RUNS, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
W = int(sys.argv[4]) if len(sys.argv) > 4 else 3
gt, meta = P.load_prepared(PREP)
R = json.load(open(os.path.join(RUNS, f"results_antiuav_s{seed}.json")))
D = pickle.load(open(os.path.join(RUNS, f"detections_s{seed}.pkl"), "rb"))
cal = R["calibration"]; prior = cal["prior"]
cir, crgb = Platt(cal["ir"]["A"], cal["ir"]["B"]), Platt(cal["rgb"]["A"], cal["rgb"]["B"])
E = np.zeros((0, 5))
keys = sorted(k for k, v in meta["split_of"].items() if v == "test")
fo = {s: ks for s, ks in meta["frames_of"].items() if meta["split_of"][ks[0]] == "test"}
smaps = {s: AxisAffine.from_dict(d) for s, d in D["tmaps"].items()}
ir, rgb = P.apply_cal(D["ir_test"], cir), P.apply_cal(D["rgb_test"], crgb)


def ap(d):
    r = frame_records(d, gt["IR"], keys)
    return ap_from(np.concatenate([v[0] for v in r.values()]), np.concatenate([v[1] for v in r.values()]), sum(v[2] for v in r.values()))


base = ap(ir)


def gains(fm):
    ri = REG.to_frame_pf(rgb, fm, "rgb2ir")
    return {"box-avg ProbEn": ap(P.fuse_all(ir, ri, keys, "proben", prior)) - base,
            "anchored ProbEn + unmatched RGB": ap({k: fuse_anchored(ir.get(k, E), ri.get(k, E), "proben", prior, gamma=1.0) for k in keys}) - base,
            "map_err_px": REG.frame_map_error(fm, gt, keys)}


oracle = REG.oracle_frame_maps(gt, fo, smaps)
curve = []
for sigma in (0, 2.5, 5, 7.5, 10, 15, 20, 30, 40):
    draws = []
    for rep in range(3 if sigma else 1):
        rng = np.random.default_rng(100 + rep)
        fm = {k: AxisAffine(m.ax, m.bx + rng.normal(0, sigma), m.ay, m.by + rng.normal(0, sigma)) for k, m in oracle.items()}
        draws.append(gains(fm))
    row = {k: float(np.mean([d[k] for d in draws])) for k in draws[0]}
    row["sigma"] = sigma
    curve.append(row); print(row, flush=True)
points = {"sequence map": gains({k: smaps[P.seq_of(k)] for k in keys}),
          f"windowed map (w={W})": gains(REG.windowed_maps(D["ir_test"], D["rgb_test"], fo, smaps, window=W))}
print(points)
json.dump({"seed": seed, "base_ir_ap50": base, "curve": curve, "points": points},
          open(os.path.join(RUNS, f"registration_curve_s{seed}.json"), "w"), indent=1)
