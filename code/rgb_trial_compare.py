#!/usr/bin/env python3
"""Small-drone trial vs the main-study YOLO11s detector of the same camera (RGB at 960 px, or thermal at 640 px): AP metrics per variant and session-cluster
bootstrap of the AP50 difference, overall and for small drones (<48 and <64 px native; the COCO S bin, <32 px, holds
only 20 test drones, too few to estimate; small drones come almost only from two evening sessions, so size-bin
intervals resample sequences and are indicative), against the same-seed YOLO11s and
(overall) the mean of its three seeds.

Thermal (MOD=IR): small bins are <32 and <24 px, and AP75 is also compared (box placement).

python rgb_trial_compare.py PREP RUNS TRIAL_DIR OUT.json [RGB|IR]
"""
import json, os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import antiuav_pipeline as P
import analyze_antiuav as A
from analyze_antiuav import frame_records, frame_records_sized, seq_pack, cluster_bootstrap, suite, COCO_BINS

PREP, RUNS, TRIAL, OUT = sys.argv[1:5]
MOD = sys.argv[5] if len(sys.argv) > 5 else "RGB"
STEM = "rgb_small_trial" if MOD == "RGB" else "ir_small_trial"
gt, meta = P.load_prepared(PREP)
keys = sorted(k for k, v in meta["split_of"].items() if v == "test")
fo = {}
for k in keys:
    fo.setdefault(P.seq_of(k), []).append(k)
seqs = sorted(fo)
trial = pickle.load(open(os.path.join(TRIAL, STEM + "_test_dets.pkl"), "rb"))
info = json.load(open(os.path.join(TRIAL, STEM + ".json")))
small = [pickle.load(open(os.path.join(RUNS, f"detections_s{s}.pkl"), "rb"))["rgb_test" if MOD == "RGB" else "ir_test"] for s in (0, 1, 2)]
BINS = {"lt48": (0, 48), "lt64": (0, 64)} if MOD == "RGB" else {"lt32": (0, 32), "lt24": (0, 24)}
_sess = A.session_of_seq


def seq_bootstrap(pa, pb):
    """Size bins: small RGB drones come almost only from two evening sessions, so resample sequences instead (indicative)."""
    A.session_of_seq = lambda q: q
    try:
        return {**cluster_bootstrap(pa, pb, seqs, 2000), "resampling_unit": "sequence"}
    finally:
        A.session_of_seq = _sess


pk = lambda d, t=0.5: seq_pack(frame_records(d, gt[MOD], keys, t), fo)
pks = lambda d, rg: seq_pack(frame_records_sized(d, gt[MOD], keys, rg), fo)
keep = ("AP50", "AP50_95", "AP75", "AP50_S(<32^2)", "AP50_M(32^2-96^2)", "AP50_<24", "AP50_24-32", "P", "R", "n_gt_S(<32^2)")
ref = [pk(d) for d in small]; ref_s = {b: [pks(d, rg) for d in small] for b, rg in BINS.items()}
from analyze_antiuav import ap_of_seqs
bin_ap = lambda d: {b: float(ap_of_seqs(pks(d, rg), seqs)) for b, rg in BINS.items()}
res = {"reference_bins_per_seed": [bin_ap(d) for d in small], "reference_yolo11s_per_seed": [{k: float(v) for k, v in suite(d, gt[MOD], keys).items() if k in keep} for d in small],
       "variants": {}}
for name, d in trial.items():
    p = pk(d)
    res["variants"][name] = {
        **{k: v for k, v in info[name].items() if k != "metrics"},
        "metrics": {k: float(v) for k, v in suite(d, gt[MOD], keys).items() if k in keep}, "bins": bin_ap(d),
        "d_AP50_vs_ref_seed0": cluster_bootstrap([p], [ref[0]], seqs, 2000),
        "d_AP50_vs_ref_3seed_mean": cluster_bootstrap([p, p, p], ref, seqs, 2000),
        "d_AP75_vs_ref_seed0": cluster_bootstrap([pk(d, 0.75)], [pk(small[0], 0.75)], seqs, 2000),
        **{f"d_AP50_{b}_vs_ref_seed0": seq_bootstrap([pks(d, rg)], [ref_s[b][0]]) for b, rg in BINS.items()},
        **{f"d_AP50_{b}_vs_ref_3seed_mean": seq_bootstrap([pks(d, rg)] * 3, ref_s[b]) for b, rg in BINS.items()},
    }
json.dump(res, open(OUT, "w"), indent=1)
print("ref YOLO11s per seed", [{k: round(v, 3) for k, v in r.items()} for r in res["reference_bins_per_seed"]])
for n, v in res["variants"].items():
    a = v["d_AP50_vs_ref_seed0"]
    line = f"{n} params={v.get('params')} AP50 {v['metrics']['AP50']:.3f} d {a['diff']:+.3f} [{a['ci95'][0]:+.3f},{a['ci95'][1]:+.3f}] p={a['p_two_sided']:.2f}"
    for b in BINS:
        c = v[f"d_AP50_{b}_vs_ref_seed0"]
        line += f" | {b} {v['bins'][b]:.3f} d {c['diff']:+.3f} [{c['ci95'][0]:+.3f},{c['ci95'][1]:+.3f}] p={c['p_two_sided']:.2f}"
    print(line)
