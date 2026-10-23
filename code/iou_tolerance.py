"""How much of the late-fusion loss is localisation? AP at several IoU thresholds for mapped RGB boxes, anchored fusion
that adds unmatched RGB boxes (gamma=1), with the self-calibrated and the GT-fitted (oracle) test maps. IR frame."""
import json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import antiuav_pipeline as P
from fusion_core import AxisAffine, Platt, fuse_anchored
from analyze_antiuav import frame_records, ap_from
PREP, RUNS, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
gt, meta = P.load_prepared(PREP)
R = json.load(open(os.path.join(RUNS, f"results_antiuav_s{seed}.json"))); D = pickle.load(open(os.path.join(RUNS, f"detections_s{seed}.pkl"), "rb"))
cal = R["calibration"]; prior = cal["prior"]; E = np.zeros((0, 5))
kt = sorted(k for k, v in meta["split_of"].items() if v == "test")
tm = {s: AxisAffine.from_dict(d) for s, d in D["tmaps"].items()}; om = {**tm, **P.gt_mappings(gt, meta, ("test",))}
ir = P.apply_cal(D["ir_test"], Platt(cal["ir"]["A"], cal["ir"]["B"])); rgb = P.apply_cal(D["rgb_test"], Platt(cal["rgb"]["A"], cal["rgb"]["B"]))
def ap(d, thr):
    r = frame_records(d, gt["IR"], kt, thr)
    return ap_from(np.concatenate([v[0] for v in r.values()]), np.concatenate([v[1] for v in r.values()]), sum(v[2] for v in r.values()))
out = {}
for mn, mp in (("self", tm), ("oracle", om)):
    ri = P.to_frame(rgb, mp, "rgb2ir")
    M = {"IR only": ir, f"RGB mapped ({mn})": ri,
         f"Anchored ProbEn, gamma=1 ({mn})": {k: fuse_anchored(ir.get(k, E), ri.get(k, E), "proben", prior, gamma=1.0) for k in kt},
         f"Late ProbEn box avg ({mn})": P.fuse_all(ir, ri, kt, "proben", prior)}
    for n, d in M.items():
        out[n] = {t: round(ap(d, t), 3) for t in (0.1, 0.3, 0.5)}
        print(f"{n:42s}", out[n])
json.dump(out, open(os.path.join(RUNS, f"iou_tolerance_s{seed}.json"), "w"), indent=1)
