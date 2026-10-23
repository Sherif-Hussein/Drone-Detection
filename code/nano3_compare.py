#!/usr/bin/env python3
"""
Three-seed YOLO11-nano against the three YOLO11-small seeds of the main study, per camera (reviewer request).
The statistic is the difference of the three-seed means (seeds are not paired across architectures). Intervals:
session-cluster bootstrap; p: exact session permutation test; equivalence at +-0.01 and +-0.02. Both checkpoint rules
(best validation, last epoch) are reported, and small-drone AP50 (thermal <24/<32 px, RGB <48 px; sequence resampling).

python nano3_compare.py PREP RUNS NANO3_DIR OUT.json
"""
import json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from session_permutation import setup, permutation_test, MARGINS
from analyze_antiuav import frame_records, frame_records_sized, seq_pack, cluster_bootstrap, suite, ap_of_seqs

PREP, RUNS, NANO, OUT = sys.argv[1:5]
gt, keys, fo, seqs = setup(PREP)
N = pickle.load(open(os.path.join(NANO, "nano3_trial_test_dets.pkl"), "rb"))
S = [pickle.load(open(os.path.join(RUNS, f"detections_s{s}.pkl"), "rb")) for s in (0, 1, 2)]
pk = lambda d, fr: seq_pack(frame_records(d, gt[fr], keys), fo)
bins = {"IR": {"lt24": (0, 24), "lt32": (0, 32)}, "RGB": {"lt48": (0, 48)}}
res = {}
for cam, nkey, skey in (("IR", "ir_n640", "ir_test"), ("RGB", "rgb_n960", "rgb_test")):
    small = [x[skey] for x in S]
    r = {"small_ap50": [float(suite(d, gt[cam], keys)["AP50"]) for d in small]}
    for ck, sfx in (("best", ""), ("last", "_last")):
        nano = [N[f"{nkey}_s{s}{sfx}"] for s in (0, 1, 2) if f"{nkey}_s{s}{sfx}" in N]
        if len(nano) < 3:
            continue
        b = cluster_bootstrap([pk(d, cam) for d in nano], [pk(d, cam) for d in small], seqs, 2000)
        p = permutation_test([pk(d, cam) for d in nano], [pk(d, cam) for d in small], seqs)
        aps = [suite(d, gt[cam], keys) for d in nano]
        r[ck] = {"ap50": [float(a["AP50"]) for a in aps], "ap75": [float(a["AP75"]) for a in aps],
                 "ap50_mean": float(np.mean([a["AP50"] for a in aps])), "ap50_sd": float(np.std([a["AP50"] for a in aps], ddof=1)),
                 "diff": b["diff"], "ci95": b["ci95"], "ci90": b["ci90"], "p_perm": p["p_perm"],
                 "equivalent": {str(m): bool(-m < b["ci90"][0] and b["ci90"][1] < m) for m in MARGINS},
                 "bins": {bn: float(np.mean([ap_of_seqs(seq_pack(frame_records_sized(d, gt[cam], keys, rg), fo), seqs) for d in nano]))
                          for bn, rg in bins[cam].items()},
                 "small_bins": {bn: float(np.mean([ap_of_seqs(seq_pack(frame_records_sized(d, gt[cam], keys, rg), fo), seqs) for d in small]))
                                for bn, rg in bins[cam].items()}}
        print(cam, ck, "nano", [round(x, 3) for x in r[ck]["ap50"]], "small", [round(x, 3) for x in r["small_ap50"]],
              "d %+.4f [%+.4f,%+.4f] p_perm %.3f" % (b["diff"], *b["ci95"], p["p_perm"]), "equiv", r[ck]["equivalent"],
              "bins", {k: round(v, 3) for k, v in r[ck]["bins"].items()}, "small bins", {k: round(v, 3) for k, v in r[ck]["small_bins"].items()},
              flush=True)
    res[cam] = r
json.dump(res, open(OUT, "w"), indent=1)
