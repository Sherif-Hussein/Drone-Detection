"""Does per-frame registration unlock the complementarity? Window chosen on VALIDATION, then reported on test.

python explore_registration.py PREP RUNS SEED
"""
import json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import antiuav_pipeline as P
from fusion_core import AxisAffine, Platt, fuse_anchored
from analyze_antiuav import frame_records, ap_from
import registration as REG

PREP, RUNS, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
gt, meta = P.load_prepared(PREP)
R = json.load(open(os.path.join(RUNS, f"results_antiuav_s{seed}.json")))
D = pickle.load(open(os.path.join(RUNS, f"detections_s{seed}.pkl"), "rb"))
mtv = json.load(open(os.path.join(RUNS, "maps_trainval_refit.json")))
gmap = AxisAffine.from_dict(mtv["global"])
sc_all = [v for s in mtv["fitted"] for v in (mtv["per_seq"][s]["ax"], mtv["per_seq"][s]["ay"])]
srange = (0.85 * np.percentile(sc_all, 2), 1.15 * np.percentile(sc_all, 98))
cal = R["calibration"]; prior = cal["prior"]
cir, crgb = Platt(cal["ir"]["A"], cal["ir"]["B"]), Platt(cal["rgb"]["A"], cal["rgb"]["B"])
E = np.zeros((0, 5))


def ap(d, g, keys, thr=0.5):
    r = frame_records(d, g, keys, thr)
    return ap_from(np.concatenate([v[0] for v in r.values()]), np.concatenate([v[1] for v in r.values()]), sum(v[2] for v in r.values()))


out = {}
for split in ("val", "test"):
    keys = sorted(k for k, v in meta["split_of"].items() if v == split)
    fo = {s: ks for s, ks in meta["frames_of"].items() if meta["split_of"][ks[0]] == split}
    smaps, _ = P.test_mappings(D[f"ir_{split}"], D[f"rgb_{split}"], meta, gmap, split=split, scale_range=srange)
    ir, rgb = P.apply_cal(D[f"ir_{split}"], cir), P.apply_cal(D[f"rgb_{split}"], crgb)
    variants = {"sequence map": {k: smaps[P.seq_of(k)] for k in keys}}
    for w in (1, 2, 3, 5, 8):
        variants[f"window {w}"] = REG.windowed_maps(D[f"ir_{split}"], D[f"rgb_{split}"], fo, smaps, window=w)
    variants["oracle per-frame"] = REG.oracle_frame_maps(gt, fo, smaps)
    base_ir, base_rgb = ap(ir, gt["IR"], keys), ap(rgb, gt["RGB"], keys)
    out[(split, "IR only")] = base_ir; out[(split, "RGB only")] = base_rgb
    for vn, fm in variants.items():
        rgb_ir = REG.to_frame_pf(rgb, fm, "rgb2ir"); ir_rgb = REG.to_frame_pf(ir, fm, "ir2rgb")
        res = {"map_err_px": REG.frame_map_error(fm, gt, keys),
               "RGB mapped AP50": ap(rgb_ir, gt["IR"], keys)}
        for gam in (0.0, 0.5, 1.0):
            f = {k: fuse_anchored(ir.get(k, E), rgb_ir.get(k, E), "proben", prior, gamma=gam) for k in keys}
            res[f"IRframe anchored g{gam}"] = ap(f, gt["IR"], keys) - base_ir
            f = {k: fuse_anchored(rgb.get(k, E), ir_rgb.get(k, E), "proben", prior, gamma=gam) for k in keys}
            res[f"RGBframe anchored g{gam}"] = ap(f, gt["RGB"], keys) - base_rgb
        res["IRframe box-avg"] = ap(P.fuse_all(ir, rgb_ir, keys, "proben", prior), gt["IR"], keys) - base_ir
        out[(split, vn)] = res
        print(split, f"{vn:16s}", {k: round(v, 4) for k, v in res.items()}, flush=True)
    print(split, "single IR", round(base_ir, 4), "RGB", round(base_rgb, 4), flush=True)
json.dump({"|".join(k): v for k, v in out.items()}, open(os.path.join(RUNS, f"registration_s{seed}.json"), "w"), indent=1)
