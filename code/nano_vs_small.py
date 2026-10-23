#!/usr/bin/env python3
"""YOLO11n (trained on CPU only) vs YOLO11s (GPU, main study) thermal detectors on the Anti-UAV300 test set:
AP metrics, and session-cluster bootstrap of the AP50 difference with the +-0.01 equivalence test, against the same-seed
YOLO11s and against the mean of the three YOLO11s seeds.

python nano_vs_small.py PREP RUNS NANO_DETS.pkl OUT.json
"""
import json, os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import antiuav_pipeline as P
from analyze_antiuav import frame_records, seq_pack, cluster_bootstrap, suite

PREP, RUNS, NANO, OUT = sys.argv[1:5]
gt, meta = P.load_prepared(PREP)
keys = sorted(k for k, v in meta["split_of"].items() if v == "test")
fo = {}
for k in keys:
    fo.setdefault(P.seq_of(k), []).append(k)
seqs = sorted(fo)
nano = pickle.load(open(NANO, "rb"))
small = [pickle.load(open(os.path.join(RUNS, f"detections_s{s}.pkl"), "rb"))["ir_test"] for s in (0, 1, 2)]
pk = lambda d: seq_pack(frame_records(d, gt["IR"], keys), fo)
pn, ps = pk(nano), [pk(d) for d in small]
keep = ("AP50", "AP50_95", "AP75", "AP50_S(<32^2)", "AP50_<24", "AP50_night", "AP50_day")
res = {"nano": {k: float(v) for k, v in suite(nano, gt["IR"], keys).items() if k in keep},
       "small_per_seed": [{k: float(v) for k, v in suite(d, gt["IR"], keys).items() if k in keep} for d in small],
       "nano_minus_small_seed0": cluster_bootstrap([pn], [ps[0]], seqs, 2000),
       "nano_minus_small_3seed_mean": cluster_bootstrap([pn, pn, pn], ps, seqs, 2000)}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items()}, indent=1)[:2500])
