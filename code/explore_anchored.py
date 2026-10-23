"""Select the anchored-fusion settings on VALIDATION sequences only (self-calibrated val maps, as at test time)."""
import json, os, pickle, sys, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import antiuav_pipeline as P
from fusion_core import AxisAffine, Platt, fuse_anchored
from analyze_antiuav import frame_records, ap_from

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


def ap(d, g, keys):
    r = frame_records(d, g, keys)
    return ap_from(np.concatenate([v[0] for v in r.values()]), np.concatenate([v[1] for v in r.values()]), sum(v[2] for v in r.values()))


out = {}
for split in ("val", "test"):
    keys = sorted(k for k, v in meta["split_of"].items() if v == split)
    maps, _ = P.test_mappings(D[f"ir_{split}"], D[f"rgb_{split}"], meta, gmap, split=split, scale_range=srange)
    ir, rgb = P.apply_cal(D[f"ir_{split}"], cir), P.apply_cal(D[f"rgb_{split}"], crgb)
    refs = {"IR": (ir, P.to_frame(rgb, maps, "rgb2ir"), gt["IR"]), "RGB": (rgb, P.to_frame(ir, maps, "ir2rgb"), gt["RGB"])}
    for ref, (a, b, g) in refs.items():
        base = ap(a, g, keys)
        out[(split, ref, "single")] = base
        for rule, tau, gam in itertools.product(("proben", "max", "wbf"), (0.5, 1.0, 2.0), (0.0, 0.1, 0.25, 0.5, 1.0)):
            f = {k: fuse_anchored(a.get(k, E), b.get(k, E), rule, prior, tau, gam) for k in keys}
            out[(split, ref, rule, tau, gam)] = ap(f, g, keys)
for ref in ("IR", "RGB"):
    cands = [(k, v) for k, v in out.items() if k[0] == "val" and k[1] == ref and len(k) == 5]
    cands.sort(key=lambda x: -x[1])
    print(f"--- {ref} frame: single val {out[('val', ref, 'single')]:.4f} test {out[('test', ref, 'single')]:.4f}")
    for k, v in cands[:6]:
        print(f"  val {v:.4f}  test {out[('test',) + k[1:]]:.4f}  {k[2:]}")
    for gam in (0.0, 1.0):
        k = ("val", ref, "proben", 1.0, gam); print(f"  [proben tau1 gamma{gam}] val {out[k]:.4f} test {out[('test',) + k[1:]]:.4f}")
json.dump({"|".join(map(str, k)): v for k, v in out.items()}, open(os.path.join(RUNS, f"anchored_grid_s{seed}.json"), "w"), indent=0)
