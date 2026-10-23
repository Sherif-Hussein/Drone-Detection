#!/usr/bin/env python3
"""Shared-weights (joint thermal+RGB) YOLO11n vs separate single-camera detectors, on each camera: AP50 differences by
session-cluster bootstrap against the separate YOLO11n of the same seed and recipe and against the YOLO11s detectors
(three seeds), plus AP50 for small drones (thermal <32/<24 px, RGB <48/<64 px; sequence resampling, indicative).

python joint_compare.py PREP RUNS IR_TRIAL RGB_TRIAL JOINT_DIR STEM OUT.json
"""
import json, os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import antiuav_pipeline as P
import analyze_antiuav as A
from analyze_antiuav import frame_records, frame_records_sized, seq_pack, cluster_bootstrap, suite, ap_of_seqs

PREP, RUNS, IRT, RGBT, JD, STEM, OUT = sys.argv[1:8]
gt, meta = P.load_prepared(PREP)
keys = sorted(k for k, v in meta["split_of"].items() if v == "test")
fo = {}
for k in keys:
    fo.setdefault(P.seq_of(k), []).append(k)
seqs = sorted(fo)
J = pickle.load(open(os.path.join(JD, STEM + "_test_dets.pkl"), "rb"))
info = json.load(open(os.path.join(JD, STEM + ".json")))
IR = pickle.load(open(os.path.join(IRT, "ir_small_trial_test_dets.pkl"), "rb"))
RGB = pickle.load(open(os.path.join(RGBT, "rgb_small_trial_test_dets.pkl"), "rb"))
MAIN = [pickle.load(open(os.path.join(RUNS, f"detections_s{s}.pkl"), "rb")) for s in (0, 1, 2)]
BINS = {"IR": {"lt32": (0, 32), "lt24": (0, 24)}, "RGB": {"lt48": (0, 48), "lt64": (0, 64)}}
_sess = A.session_of_seq


def seq_boot(pa, pb):
    A.session_of_seq = lambda q: q
    try:
        return {**cluster_bootstrap(pa, pb, seqs, 2000), "resampling_unit": "sequence"}
    finally:
        A.session_of_seq = _sess


pk = lambda d, m: seq_pack(frame_records(d, gt[m], keys), fo)
pks = lambda d, m, rg: seq_pack(frame_records_sized(d, gt[m], keys, rg), fo)
keep = ("AP50", "AP50_95", "AP75", "P", "R")
prefix = [k for k in J if k.endswith("_on_IR")][0][:-len("_on_IR")]
res = {"model": prefix, "train_seconds": info[prefix + "_on_IR"].get("train_seconds"), "cameras": {}}
for cam, key, sep, sep_name in (("IR", prefix + "_on_IR", IR["ir_n640"], "ir_n640"),
                                ("RGB", prefix + "_on_RGB", RGB["n960"], "n960")):
    d = J[key]
    small = [m["ir_test" if cam == "IR" else "rgb_test"] for m in MAIN]
    r = {"metrics": {k: float(v) for k, v in suite(d, gt[cam], keys).items() if k in keep},
         "gpu_latency_ms": info[key].get("latency_ms_batch1_0"),
         "bins": {b: float(ap_of_seqs(pks(d, cam, rg), seqs)) for b, rg in BINS[cam].items()},
         "separate_nano_bins": {b: float(ap_of_seqs(pks(sep, cam, rg), seqs)) for b, rg in BINS[cam].items()},
         "vs_separate_nano": cluster_bootstrap([pk(d, cam)], [pk(sep, cam)], seqs, 2000),
         "vs_small_seed0": cluster_bootstrap([pk(d, cam)], [pk(small[0], cam)], seqs, 2000),
         "vs_small_3seed_mean": cluster_bootstrap([pk(d, cam)] * 3, [pk(x, cam) for x in small], seqs, 2000),
         **{f"vs_separate_nano_{b}": seq_boot([pks(d, cam, rg)], [pks(sep, cam, rg)]) for b, rg in BINS[cam].items()}}
    res["cameras"][cam] = r
    print(cam, {k: round(v, 3) for k, v in r["metrics"].items()}, "bins", {k: round(v, 3) for k, v in r["bins"].items()},
          "| vs nano %+.3f [%+.3f,%+.3f] p=%.3f" % (r["vs_separate_nano"]["diff"], *r["vs_separate_nano"]["ci95"], r["vs_separate_nano"]["p_two_sided"]),
          "| vs small(3) %+.3f [%+.3f,%+.3f] p=%.3f" % (r["vs_small_3seed_mean"]["diff"], *r["vs_small_3seed_mean"]["ci95"], r["vs_small_3seed_mean"]["p_two_sided"]))
json.dump(res, open(OUT, "w"), indent=1)
