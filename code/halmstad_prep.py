#!/usr/bin/env python3
"""
Prepare the Halmstad multi-sensor drone dataset for YOLO training.

- Reads the 650 videos (IR 320x256, visible 640x512) and their MATLAB groundTruth
  labels (via mcos-decoder).
- Samples every `--step`-th frame, writes JPEG frames and YOLO label files.
- Builds two split protocols per sensor:
    clip   : 70/15/15 by clip, stratified by class x distance bin (leakage-free)
    random : 70/15/15 by frame, ignoring clips (the leaky protocol many papers use)
- Writes per-distance test lists for the clip protocol.

Usage:
  python halmstad_prep.py --root DATA_DIR --out OUT_DIR --step 30 --seed 0
  DATA_DIR must contain Video_dataset_description.xlsx, labels/*.mat, videos/*.mp4
"""
import argparse, json, os, random
from collections import defaultdict

import cv2
import pandas as pd
from mcos_decoder import load_groundtruth

CLASSES = ["AIRPLANE", "BIRD", "DRONE", "HELICOPTER"]


def stratified_clip_split(clips, seed, frac=(0.70, 0.15, 0.15)):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for c in clips:
        groups[(c["class"], c["distance"])].append(c["clip"])
    split = {}
    for _, names in sorted(groups.items()):
        names = sorted(names)
        rng.shuffle(names)
        n = len(names)
        n_te = max(1, round(n * frac[2])) if n >= 3 else 0
        n_va = max(1, round(n * frac[1])) if n >= 3 else 0
        for i, nm in enumerate(names):
            split[nm] = "test" if i < n_te else ("val" if i < n_te + n_va else "train")
    return split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--step", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    d = pd.read_excel(os.path.join(a.root, "Video_dataset_description.xlsx"))
    d.columns = [c.strip() for c in d.columns]
    dist_col = [c for c in d.columns if c.startswith("DISTANCE")][0]
    clips = []
    for _, r in d.iterrows():
        s = "IR" if r["SENSOR"] == "IR" else "V"
        clips.append({"clip": f"{s}_{r['CLASS']}_{int(r['NUMBER']):03d}", "sensor": s,
                      "class": r["CLASS"], "distance": str(r[dist_col]).strip()})

    meta = {"step": a.step, "seed": a.seed, "classes": CLASSES, "sensors": {}}
    for s in ["IR", "V"]:
        sc = [c for c in clips if c["sensor"] == s]
        img_dir = os.path.join(a.out, s, "images"); lab_dir = os.path.join(a.out, s, "labels")
        os.makedirs(img_dir, exist_ok=True); os.makedirs(lab_dir, exist_ok=True)
        frames = []  # (img_path, clip, class, distance, has_target, sqrt_area)
        for c in sc:
            gt = load_groundtruth(os.path.join(a.root, "labels", c["clip"] + "_LABELS.mat"))
            cap = cv2.VideoCapture(os.path.join(a.root, "videos", c["clip"] + ".mp4"))
            k = 0
            while True:
                if k % a.step != 0:
                    if not cap.grab():
                        break
                    k += 1
                    continue
                ok, im = cap.read()
                if not ok:
                    break
                if k < len(gt):
                    H, W = im.shape[:2]
                    name = f"{c['clip']}_{k:04d}"
                    ip = os.path.join(img_dir, name + ".jpg")
                    if not os.path.exists(ip):
                        cv2.imwrite(ip, im, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    b = gt[k]
                    with open(os.path.join(lab_dir, name + ".txt"), "w") as f:
                        if b is not None and b[2] > 0 and b[3] > 0:
                            x, y, w, h = b
                            cx = min(max((x + w / 2) / W, 0), 1); cy = min(max((y + h / 2) / H, 0), 1)
                            f.write(f"{CLASSES.index(c['class'])} {cx:.6f} {cy:.6f} {min(w / W, 1):.6f} {min(h / H, 1):.6f}\n")
                    sq = (b[2] * b[3]) ** 0.5 if b is not None else 0.0
                    frames.append((ip, c["clip"], c["class"], c["distance"], b is not None, round(sq, 2)))
                k += 1
            cap.release()

        # clip protocol
        cs = stratified_clip_split(sc, a.seed)
        lists = defaultdict(list)
        for f in frames:
            lists["clip_" + cs[f[1]]].append(f[0])
            if cs[f[1]] == "test":
                lists["clip_test_" + f[3].lower()].append(f[0])
        # random-frame protocol (the leaky practice), drawn only from clips of the clip-level train+val sets so
        # that the clip-level test clips stay unseen by both models. random_train has exactly as many frames as
        # clip_train; the remaining frames are split evenly into random_val and random_test (frames of
        # clips that also contribute training frames). Each model is then scored on its own test set and on
        # the unseen clip-level test set.
        rng = random.Random(a.seed)
        idx = [i for i, f in enumerate(frames) if cs[f[1]] != "test"]; rng.shuffle(idx)
        n_tr = len(lists["clip_train"]); n_rest = len(idx) - n_tr
        for j, i in enumerate(idx):
            sp = "train" if j < n_tr else ("val" if j < n_tr + n_rest // 2 else "test")
            lists["random_" + sp].append(frames[i][0])
        for k_, v in lists.items():
            with open(os.path.join(a.out, s, k_ + ".txt"), "w") as f:
                f.write("\n".join(sorted(v)) + "\n")
        # yaml files
        def yaml(name, tr, va, te):
            with open(os.path.join(a.out, s, name + ".yaml"), "w") as f:
                f.write(f"path: {os.path.abspath(os.path.join(a.out, s))}\ntrain: {tr}.txt\nval: {va}.txt\ntest: {te}.txt\n")
                f.write("names:\n" + "".join(f"  {i}: {c.lower()}\n" for i, c in enumerate(CLASSES)))
        yaml("clip", "clip_train", "clip_val", "clip_test")
        yaml("random", "random_train", "random_val", "random_test")
        yaml("random_unseen", "random_train", "random_val", "clip_test")
        for dist in ["close", "medium", "distant"]:
            yaml(f"clip_test_{dist}", "clip_train", f"clip_test_{dist}", f"clip_test_{dist}")
        pd.DataFrame(frames, columns=["image", "clip", "class", "distance", "has_target", "sqrt_area"]).assign(
            clip_split=lambda x: x["clip"].map(cs)).to_csv(os.path.join(a.out, s, "frames.csv"), index=False)
        meta["sensors"][s] = {k_: len(v) for k_, v in sorted(lists.items())}
        meta["sensors"][s]["clips_by_split"] = {sp: sum(1 for v in cs.values() if v == sp) for sp in ["train", "val", "test"]}
    with open(os.path.join(a.out, "prep_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
