#!/usr/bin/env python3
"""Evaluate the single-detector ablation with the paper's evaluator: test AP50, AP50:95, AP by size and light, and the
sequence-bootstrap difference to the baseline variant. Selection uses the session-disjoint selection AP50:95.

python evaluate_ablation.py --prep PREP --runs DIR_WITH_ablation_files --frame IR --base ir_base --out OUT.json
"""
import argparse, glob, json, os, pickle, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import antiuav_pipeline as P  # noqa: E402
from analyze_antiuav import bootstrap, frame_records, seq_pack, suite  # noqa: E402
from analyze_antiuav2 import LIGHT, light_ap  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True); ap.add_argument("--runs", required=True)
    ap.add_argument("--frame", default="IR"); ap.add_argument("--base", default="ir_base")
    ap.add_argument("--out", required=True); ap.add_argument("--bootstrap", type=int, default=1000)
    a = ap.parse_args()
    gt, meta = P.load_prepared(a.prep)
    G, scene = gt[a.frame], meta["scene"]
    kt = sorted(k for k, v in meta["split_of"].items() if v == "test")
    sf = {}
    for k in kt:
        sf.setdefault(P.seq_of(k), []).append(k)
    info = {}
    for f in glob.glob(os.path.join(a.runs, "ablation_s*.json")):
        info.update(json.load(open(f))["variants"])
    dets = {os.path.basename(f)[len("ablation_"):].rsplit("_s", 1)[0]: pickle.load(open(f, "rb"))["test"]
            for f in glob.glob(os.path.join(a.runs, "ablation_*_s*.pkl"))}
    out = {}
    pack = lambda d: [seq_pack(frame_records(d, G, kt), sf)]
    for v, d in sorted(dets.items()):
        m = suite(d, G, kt)
        m.update({"AP50_" + k: x for k, x in light_ap(d, G, kt, scene).items() if not k.endswith("_frames")})
        m["selection_AP50_95"] = info.get(v, {}).get("val_map50_95")
        m["latency_ms"] = info.get(v, {}).get("latency_ms")
        if v != a.base and a.base in dets:
            m["boot_vs_base"] = bootstrap(pack(d), pack(dets[a.base]), sorted(sf), a.bootstrap)
        out[v] = m
        b = m.get("boot_vs_base")
        print(f"{v:10s} sel {m['selection_AP50_95'] if m['selection_AP50_95'] is None else round(m['selection_AP50_95'], 3)}  "
              f"AP50 {m['AP50']:.4f}  AP50:95 {m['AP50_95']:.4f}  S {m['AP50_S(<32^2)']:.3f}  night {m['AP50_night']:.3f}  "
              f"day {m['AP50_day']:.3f}" + (f"  d {b['diff']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]" if b else ""), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
