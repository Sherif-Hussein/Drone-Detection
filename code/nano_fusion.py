#!/usr/bin/env python3
"""
End-to-end evaluation of the recommended lightweight station (reviewer request): a YOLO11-nano thermal detector and a
YOLO11-nano RGB detector (three seeds each), maps self-calibrated from the nano detections themselves, and anchored
late fusion with calibration and settings tuned by leave-one-session-out cross-validation, exactly as for the
YOLO11-small pair (loso_cv.py). Also: the fixed configuration of the reference design (WBF, tau=2, gamma=0.25), the
thermal nano alone, complete loss of either camera, and the same-session permutation test.

python nano_fusion.py PREP RUNS NANO3_DIR OUT.json [best|last]
"""
import itertools, json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import antiuav_pipeline as P
from fusion_core import AxisAffine, fuse_anchored
from analyze_antiuav import frame_records, ap_from, seq_pack, session_of_seq, cluster_bootstrap, holm
from session_permutation import permutation_test

PREP, RUNS, NANO, OUT = sys.argv[1:5]
CKPT = sys.argv[5] if len(sys.argv) > 5 else "best"
E = np.zeros((0, 5))
gt, meta = P.load_prepared(PREP)
keys = sorted(k for k, v in meta["split_of"].items() if v == "test")
fo = {}
for k in keys:
    fo.setdefault(P.seq_of(k), []).append(k)
seqs = sorted(fo)
sess = sorted({session_of_seq(s) for s in seqs})
keys_of = {h: [k for k in keys if session_of_seq(P.seq_of(k)) == h] for h in sess}
GRID = list(itertools.product(("proben", "max", "wbf"), (0.5, 1.0, 2.0), (0.0, 0.25, 1.0)))
DEFAULT, FIXED = ("proben", 1.0, 0.0), ("wbf", 2.0, 0.25)
MTV = json.load(open(os.path.join(RUNS, "maps_trainval_refit.json")))
gmap = AxisAffine.from_dict(MTV["global"])
fitted = [AxisAffine.from_dict(MTV["per_seq"][s]) for s in MTV["fitted"] if s in MTV["per_seq"]]
sc = [v for m in fitted for v in (m.ax, m.ay)]
scale_range = (0.85 * float(np.percentile(sc, 2)), 1.15 * float(np.percentile(sc, 98)))
N = pickle.load(open(os.path.join(NANO, "nano3_trial_test_dets.pkl"), "rb"))
sfx = "" if CKPT == "best" else "_last"
seeds = [0, 1, 2]


def ap(d, ks):
    r = frame_records(d, gt["IR"], ks)
    return ap_from(np.concatenate([r[k][0] for k in ks]), np.concatenate([r[k][1] for k in ks]), sum(r[k][2] for k in ks))


def anchored(a, b, ks, cfg, prior):
    rule, tau, gam = cfg
    return {k: fuse_anchored(a.get(k, E), b.get(k, E), rule, prior, tau, gam) for k in ks}


names = ("thermal nano alone", "anchored nano pair, CV-tuned", "anchored nano pair, ProbEn default", "anchored nano pair, MAX default",
         "anchored nano pair, fixed design", "RGB camera lost (thermal only)")
outs = {n: [] for n in names}
res = {"checkpoint": CKPT, "per_seed": {}}
for s in seeds:
    ir_raw, rgb_raw = N[f"ir_n640_s{s}{sfx}"], N[f"rgb_n960_s{s}{sfx}"]
    tmaps, tinfo = P.test_mappings(ir_raw, rgb_raw, meta, gmap, scale_range=scale_range)
    err = P.mapping_error(tmaps, gt, meta)
    rgb_ir_raw = P.to_frame(rgb_raw, tmaps, "rgb2ir")
    cv = {n: {} for n in names}
    for h in sess:
        tr = [k for k in keys if k not in set(keys_of[h])]
        cir, prior = P.calibrate(ir_raw, gt["IR"], tr)
        crgb, _ = P.calibrate(rgb_raw, gt["RGB"], tr)
        ir, rgb = P.apply_cal(ir_raw, cir), P.to_frame(P.apply_cal(rgb_raw, crgb), tmaps, "rgb2ir")
        best = max(GRID, key=lambda c: ap(anchored(ir, rgb, tr, c, prior), tr))
        hk = keys_of[h]
        cv["thermal nano alone"].update({k: ir.get(k, E) for k in hk})
        cv["anchored nano pair, CV-tuned"].update(anchored(ir, rgb, hk, best, prior))
        cv["anchored nano pair, ProbEn default"].update(anchored(ir, rgb, hk, DEFAULT, prior))
        cv["anchored nano pair, MAX default"].update(anchored(ir, rgb, hk, ("max", 1.0, 0.0), prior))
        cv["anchored nano pair, fixed design"].update(anchored(ir, rgb, hk, FIXED, prior))
        cv["RGB camera lost (thermal only)"].update(anchored(ir, {}, hk, FIXED, prior))
    # thermal camera lost: the station falls back to the RGB detector in its own image frame
    r_rgb = frame_records(rgb_raw, gt["RGB"], keys)
    rgb_alone = ap_from(np.concatenate([r_rgb[k][0] for k in keys]), np.concatenate([r_rgb[k][1] for k in keys]), sum(r_rgb[k][2] for k in keys))
    res["per_seed"][s] = {"ap50": {n: ap(d, keys) for n, d in cv.items()}, "thermal_lost_rgb_frame_ap50": float(rgb_alone),
                          "median_map_err_px": float(np.median(list(err.values()))) if err else None,
                          "self_calibrated_share": float(np.mean([v["self_calibrated"] for v in tinfo.values()]))}
    print("seed", s, {n: round(v, 4) for n, v in res["per_seed"][s]["ap50"].items()}, "map err", res["per_seed"][s]["median_map_err_px"], flush=True)
    for n in names:
        outs[n].append(seq_pack(frame_records(cv[n], gt["IR"], keys), fo))
res["ap50_mean"] = {n: float(np.mean([res["per_seed"][s]["ap50"][n] for s in seeds])) for n in names}
res["thermal_lost_rgb_frame_mean"] = float(np.mean([res["per_seed"][s]["thermal_lost_rgb_frame_ap50"] for s in seeds]))
res["ap50_sd"] = {n: float(np.std([res["per_seed"][s]["ap50"][n] for s in seeds], ddof=1)) for n in names}
# against the thermal nano alone, and against the YOLO11-small pair (main study, same protocol)
comps = {n: cluster_bootstrap(outs[n], outs["thermal nano alone"], seqs, 2000) for n in names[1:5]}
perm = {n: permutation_test(outs[n], outs["thermal nano alone"], seqs) for n in names[1:5]}
adj = holm({n: v["p_perm"] for n, v in perm.items()})
for n in comps:
    comps[n].update({"p_perm": perm[n]["p_perm"], "p_perm_holm": adj[n]})
    print(n, round(comps[n]["diff"], 4), [round(c, 4) for c in comps[n]["ci95"]], "p_perm", perm[n]["p_perm"], "holm", round(adj[n], 3),
          "equiv" if comps[n]["equivalent"] else "", flush=True)
res["vs_thermal_nano"] = comps
json.dump(res, open(OUT, "w"), indent=1)
print({n: round(v, 4) for n, v in res["ap50_mean"].items()})
