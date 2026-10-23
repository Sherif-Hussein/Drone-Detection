#!/usr/bin/env python3
"""
Extract the Anti-UAV410 thermal sequences that are NOT part of Anti-UAV300 (the 100 new sequences) as extra IR training
data. Leakage guard: sequences whose name is in Anti-UAV300 (any split) are skipped, and so is any sequence from one of
the 10 Anti-UAV300 test recording sessions. Every --step-th frame is written with its YOLO label.

python a410_extra_prep.py --data A410_ROOT --au300 au300_names.json --out /tmp/skyfuse_au/a410x --step 40
"""
import argparse, glob, json, os, shutil

import cv2

from antiuav_pipeline import read_labels, _write_yolo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--au300", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--step", type=int, default=40)
    a = ap.parse_args()
    au = json.load(open(a.au300))
    au_names, test_sessions = set(au["names"]), set(au["test_sessions"])
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True); os.makedirs(os.path.join(a.out, "labels"), exist_ok=True)
    seq_dirs = sorted({os.path.dirname(p) for p in glob.glob(os.path.join(a.data, "**", "IR_label.json"), recursive=True)})
    used, skipped, paths = [], {"in_au300": 0, "test_session": 0}, []
    for d in seq_dirs:
        name = os.path.basename(d)
        if name in au_names:
            skipped["in_au300"] += 1; continue
        if "_".join(name.split("_")[:2]) in test_sessions:
            skipped["test_session"] += 1; continue
        labels = read_labels(os.path.join(d, "IR_label.json"))
        frames = sorted(glob.glob(os.path.join(d, "*.jpg")))
        n = min(len(frames), len(labels))
        for i in range(0, n, a.step):
            k = f"a410_{name}__{i:05d}"
            im = cv2.imread(frames[i])
            if im is None:
                continue
            H, W = im.shape[:2]
            ip = os.path.join(a.out, "images", k + ".jpg")
            shutil.copyfile(frames[i], ip)
            _write_yolo(os.path.join(a.out, "labels", k + ".txt"), labels[i], W, H)
            paths.append(ip)
        used.append(name)
    with open(os.path.join(a.out, "extra_train.txt"), "w") as f:
        f.write("\n".join(paths) + "\n")
    info = {"sequences_used": len(used), "frames": len(paths), "skipped": skipped, "names": used}
    json.dump(info, open(os.path.join(a.out, "a410x_info.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != "names"}, indent=1))


if __name__ == "__main__":
    main()
