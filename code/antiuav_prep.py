#!/usr/bin/env python3
"""
CPU stage of the Anti-UAV300 study: everything that depends only on the data and ground truth.

  - sample every --step-th synchronised IR/RGB frame (IR 640x512, RGB full resolution + 960-wide copy)
  - YOLO datasets for the IR and RGB detectors
  - per-sequence IR->RGB maps fitted on train/val ground truth (test maps are fitted later from detections)
  - native-resolution RGB crops for the cascade verifier (train/val)
  - pixel-level fusion images in the IR frame (train/val)
  - dataset statistics

python antiuav_prep.py --data DATA_ROOT --work /tmp/skyfuse_au --step 40
"""
import argparse, json, os, time

import numpy as np

import antiuav_pipeline as P


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True); p.add_argument("--work", required=True)
    p.add_argument("--step", type=int, default=40); p.add_argument("--max_seqs", type=int, default=None)
    a = p.parse_args()
    t0 = time.time()
    prep = os.path.join(a.work, "prep")
    seqs = P.discover(a.data)
    info = {"n_sequences": len(seqs), "by_split": {s: sum(x["split"] == s for x in seqs) for s in ("train", "val", "test")},
            "example": seqs[0]["files"] if seqs else None}
    print(json.dumps(info, indent=1), flush=True)
    if not seqs:
        raise SystemExit("No paired RGB/IR sequences found under " + a.data)
    gt, meta = P.prepare(seqs, prep, step=a.step, max_seqs=a.max_seqs)
    gmaps = P.gt_mappings(gt, meta, ("train", "val"))
    gmap = P.global_mapping(gmaps)
    vmaps = dict(gmaps)
    for s, ks in meta["frames_of"].items():
        if meta["split_of"][ks[0]] in ("train", "val"):
            vmaps.setdefault(s, gmap)
    with open(os.path.join(prep, "maps_trainval.json"), "w") as f:
        json.dump({"global": gmap.to_dict(), "per_seq": {s: m.to_dict() for s, m in vmaps.items()},
                   "fitted": sorted(gmaps)}, f)
    print("[prep] crops", flush=True)
    P.build_crops(prep, os.path.join(a.work, "crops"), gt, meta, seed=0)
    print("[prep] pixel fusion", flush=True)
    P.build_pixel_fusion(prep, os.path.join(a.work, "pixfuse"), gt, meta, vmaps, ("train", "val"))
    st = P.data_stats(gt, meta)
    st["discovery"] = info
    st["mapping_global"] = gmap.to_dict()
    st["prep_minutes"] = round((time.time() - t0) / 60, 1)
    with open(os.path.join(prep, "data_stats.json"), "w") as f:
        json.dump(st, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(json.dumps({k: v for k, v in st.items() if k in ("sequences", "frames", "exist_rate", "light_bins", "prep_minutes")}, indent=1))


if __name__ == "__main__":
    main()
