#!/usr/bin/env python3
"""
E1 on the full training set (reviewer request): thermal YOLO11-small trained on all Anti-UAV300 training sessions plus
100 Anti-UAV410 scenes, selected on the official validation split like the main study, three seeds; against the
three main-study thermal detectors. Difference of three-seed means, session-cluster bootstrap and exact session
permutation test.

python e1full_compare.py PREP RUNS E1_DIR OUT.json
"""
import json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from session_permutation import setup, permutation_test
from analyze_antiuav import frame_records, seq_pack, cluster_bootstrap, suite

PREP, RUNS, E1, OUT = sys.argv[1:5]
gt, keys, fo, seqs = setup(PREP)
e1 = [pickle.load(open(os.path.join(E1, f"ablation_ir_a410_s{s}.pkl"), "rb"))["test"] for s in (0, 1, 2)]
main = [pickle.load(open(os.path.join(RUNS, f"detections_s{s}.pkl"), "rb"))["ir_test"] for s in (0, 1, 2)]
pk = lambda d: seq_pack(frame_records(d, gt["IR"], keys), fo)
b = cluster_bootstrap([pk(d) for d in e1], [pk(d) for d in main], seqs, 2000)
p = permutation_test([pk(d) for d in e1], [pk(d) for d in main], seqs)
res = {"e1_full_ap50": [float(suite(d, gt["IR"], keys)["AP50"]) for d in e1],
       "main_ap50": [float(suite(d, gt["IR"], keys)["AP50"]) for d in main],
       "diff": b["diff"], "ci95": b["ci95"], "p_perm": p["p_perm"], "p_boot": b["p_two_sided"]}
res["e1_full_mean"], res["e1_full_sd"] = float(np.mean(res["e1_full_ap50"])), float(np.std(res["e1_full_ap50"], ddof=1))
print(json.dumps(res, indent=1))
json.dump(res, open(OUT, "w"), indent=1)
