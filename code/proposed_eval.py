"""End-to-end evaluation of the proposed configuration (seed 0, IR frame): thermal detectors from the ablation
(E0 baseline, E1 extra thermal scenes, E5 feature channels) fused with the RGB detector by anchored late fusion.
Calibration and fusion settings are fitted by leave-one-session-out cross-validation on the test sessions
(as in loso_cv.py); the fixed setting WBF, tau = 2, gamma = 0.25 (the setting chosen most often there) is also reported.

python proposed_eval.py PREP RUNS ABLATION_DIR OUT.json
"""
import itertools, json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import antiuav_pipeline as P
from fusion_core import AxisAffine, fuse_anchored
from analyze_antiuav import frame_records, ap_from, seq_pack, session_of_seq, cluster_bootstrap, holm, suite

PREP, RUNS, ABL, OUT = sys.argv[1:5]
E = np.zeros((0, 5))
gt, meta = P.load_prepared(PREP)
scene = meta["scene"]
keys = sorted(k for k, v in meta["split_of"].items() if v == "test")
fo = {}
for k in keys:
    fo.setdefault(P.seq_of(k), []).append(k)
seqs = sorted(fo)
sess = sorted({session_of_seq(s) for s in seqs})
keys_of = {h: [k for k in keys if session_of_seq(P.seq_of(k)) == h] for h in sess}
D = pickle.load(open(os.path.join(RUNS, "detections_s0.pkl"), "rb"))
tmaps = {q: AxisAffine.from_dict(d) for q, d in D["tmaps"].items()}
rgb_raw = D["rgb_test"]
rgb_ir_raw = P.to_frame(rgb_raw, tmaps, "rgb2ir")
GRID = list(itertools.product(("proben", "max", "wbf"), (0.5, 1.0, 2.0), (0.0, 0.25, 1.0)))
FIXED = ("wbf", 2.0, 0.25)


def ap(d, ks):
    r = frame_records(d, gt["IR"], ks)
    return ap_from(np.concatenate([r[k][0] for k in ks]), np.concatenate([r[k][1] for k in ks]), sum(r[k][2] for k in ks))


def anchored(a, b, ks, cfg, prior):
    return {k: fuse_anchored(a.get(k, E), b.get(k, E), cfg[0], prior, cfg[1], cfg[2]) for k in ks}


thermal = {"E0 (baseline)": pickle.load(open(os.path.join(ABL, "ablation_ir_base_s0.pkl"), "rb"))["test"],
           "E1 (+ Anti-UAV410 scenes)": pickle.load(open(os.path.join(ABL, "ablation_ir_a410_s0.pkl"), "rb"))["test"],
           "E5 (+ CLAHE/top-hat channels)": pickle.load(open(os.path.join(ABL, "ablation_ir_fe_s0.pkl"), "rb"))["test"],
           "Main-study IR detector": D["ir_test"]}
res, packs = {}, {}
for name, ir_raw in thermal.items():
    out = {n: {} for n in ("alone", "fixed", "cv")}
    for h in sess:
        tk = [k for k in keys if k not in set(keys_of[h])]
        cir, prior = P.calibrate(ir_raw, gt["IR"], tk)
        crgb, _ = P.calibrate(rgb_raw, gt["RGB"], tk)
        ir, rgb = P.apply_cal(ir_raw, cir), P.apply_cal(rgb_ir_raw, crgb)
        best = max(GRID, key=lambda c: ap(anchored(ir, rgb, tk, c, prior), tk))
        hk = keys_of[h]
        out["alone"].update({k: ir.get(k, E) for k in hk})
        out["fixed"].update(anchored(ir, rgb, hk, FIXED, prior))
        out["cv"].update(anchored(ir, rgb, hk, best, prior))
    res[name] = {}
    for n, d in out.items():
        s = suite(d, gt["IR"], keys)
        res[name][n] = {k: float(s[k]) for k in ("AP50", "AP50_95", "AP75", "AP50_S(<32^2)", "AP50_night", "AP50_day", "P", "R", "F1") if k in s}
        packs[(name, n)] = [seq_pack(frame_records(d, gt["IR"], keys), fo)]
    print(name, {n: round(v["AP50"], 4) for n, v in res[name].items()}, flush=True)
comps = [(("E1 (+ Anti-UAV410 scenes)", "alone"), ("E0 (baseline)", "alone")),
         (("E1 (+ Anti-UAV410 scenes)", "cv"), ("E1 (+ Anti-UAV410 scenes)", "alone")),
         (("E1 (+ Anti-UAV410 scenes)", "fixed"), ("E1 (+ Anti-UAV410 scenes)", "alone")),
         (("E1 (+ Anti-UAV410 scenes)", "cv"), ("E0 (baseline)", "alone")),
         (("E0 (baseline)", "cv"), ("E0 (baseline)", "alone")),
         (("E5 (+ CLAHE/top-hat channels)", "cv"), ("E5 (+ CLAHE/top-hat channels)", "alone")),
         (("Main-study IR detector", "cv"), ("Main-study IR detector", "alone"))]
boot = {f"{a[0]}|{a[1]} - {b[0]}|{b[1]}": cluster_bootstrap(packs[a], packs[b], seqs, 2000) for a, b in comps}
adj = holm({k: v["p_two_sided"] for k, v in boot.items()})
for k, v in boot.items():
    v["p_holm"] = adj[k]
    print(k, round(v["diff"], 4), [round(x, 4) for x in v["ci95"]], "p_holm", round(v["p_holm"], 3))
json.dump({"metrics": res, "bootstrap": boot, "fixed_cfg": list(FIXED)}, open(OUT, "w"), indent=1)
