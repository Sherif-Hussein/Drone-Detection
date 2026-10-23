#!/usr/bin/env python3
"""
CPU inference benchmark on this PC for ONE configuration (run each configuration in its own process):
model (yolo11n/yolo11s) x camera (thermal 640x512 / RGB 960x540) x backend (pytorch / openvino FP32), batch size 1,
on real Anti-UAV300 frames from the local smoke subset. Appends a JSON line with median and 90th-percentile latency.

python inference_bench.py --frames SMOKE_PREP_DIR --model yolo11n --cam thermal --backend openvino --out LOG.jsonl
"""
import argparse, glob, json, os, shutil, time

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="yolo11n"); ap.add_argument("--cam", default="thermal")
    ap.add_argument("--backend", default="pytorch"); ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--workdir", default=".")
    a = ap.parse_args()
    a.out, a.frames = os.path.abspath(a.out), os.path.abspath(a.frames)
    from ultralytics import YOLO
    sub, sz, shape = (("IR", 640, (512, 640)) if a.cam == "thermal" else ("RGBs", 960, (544, 960)))
    ims = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(a.frames, sub, "images", "test_*.jpg")))[:25]]
    os.chdir(a.workdir)
    tag = f"{a.model}_{a.cam}"
    if not os.path.exists(tag + ".pt"):
        shutil.copy(a.model + ".pt", tag + ".pt")
    model = YOLO(tag + ".pt")
    if a.backend == "openvino":
        model = YOLO(model.export(format="openvino", imgsz=shape, verbose=False), task="detect")
    elif a.backend == "onnx":
        model = YOLO(model.export(format="onnx", imgsz=shape, simplify=True, verbose=False), task="detect")
    for im in ims[:5]:
        model.predict(im, imgsz=sz, device="cpu", verbose=False)
    t = []
    for i in range(a.runs):
        s = time.perf_counter()
        model.predict(ims[i % len(ims)], imgsz=sz, device="cpu", verbose=False)
        t.append((time.perf_counter() - s) * 1000)
    r = {"model": a.model, "cam": a.cam, "backend": a.backend, "median_ms": float(np.median(t)),
         "p90_ms": float(np.percentile(t, 90)), "fps": 1000 / float(np.median(t)), "runs": a.runs}
    with open(a.out, "a") as f:
        f.write(json.dumps(r) + "\n")
    print(r)


if __name__ == "__main__":
    main()
