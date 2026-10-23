#!/usr/bin/env python3
"""
End-to-end CPU cost on this laptop (reviewer request), with trained single-class weights and ONNX Runtime through the
Ultralytics predictor (letterboxing, inference and duplicate removal included), on real Anti-UAV300 test frames.

  single : one detector, one stream, batch 1 -> median end-to-end ms per frame
  pair   : thermal and RGB detectors in two concurrent processes on the same CPU -> per-stream ms and frames per second
  memory : peak additional RAM of one process that holds (a) two separate detectors or (b) one shared detector with a
           dynamic input size, and runs both cameras' frames

python e2e_bench.py --mode single|pair|memory ... --out LOG.jsonl
"""
import argparse, glob, json, os, sys, time

import numpy as np
import psutil

E2E_DIR = os.path.join(os.environ.get("TEMP", "/tmp"), "e2e_onnx")


def frames(cam, n=40):
    root = {"thermal": os.environ["E2E_IR_DIR"], "rgb": os.environ["E2E_RGB_DIR"]}[cam]
    import cv2
    return [cv2.imread(p) for p in sorted(glob.glob(os.path.join(root, "test_*.jpg")))[:n]]


def onnx_model(weights, imgsz, dynamic=False):
    from ultralytics import YOLO
    os.makedirs(E2E_DIR, exist_ok=True)
    tag = os.path.splitext(os.path.basename(weights))[0] + ("_dyn" if dynamic else f"_{imgsz[0]}x{imgsz[1]}")
    out = os.path.join(E2E_DIR, tag + ".onnx")
    if not os.path.exists(out):
        src = os.path.join(E2E_DIR, tag + ".pt")
        import shutil; shutil.copy(weights, src)
        p = YOLO(src).export(format="onnx", imgsz=imgsz if not dynamic else max(imgsz), dynamic=dynamic, simplify=True, verbose=False)
        os.replace(p, out)
    return YOLO(out, task="detect")


def run_stream(model, ims, imgsz, n):
    for im in ims[:5]:
        model.predict(im, imgsz=imgsz, device="cpu", verbose=False)
    t = []
    for i in range(n):
        s = time.perf_counter(); model.predict(ims[i % len(ims)], imgsz=imgsz, device="cpu", verbose=False)
        t.append((time.perf_counter() - s) * 1000)
    return t


SHAPE = {"thermal": (512, 640), "rgb": (544, 960)}
SIZE = {"thermal": 640, "rgb": 960}


def peak():
    mi = psutil.Process().memory_info()
    return getattr(mi, "peak_wset", mi.rss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--w", help="weights (single)"); ap.add_argument("--cam", default="thermal")
    ap.add_argument("--w_ir"); ap.add_argument("--w_rgb"); ap.add_argument("--w_shared"); ap.add_argument("--tag", default="")
    ap.add_argument("--runs", type=int, default=60)
    a = ap.parse_args()
    a.out = os.path.abspath(a.out)
    if a.mode == "single":
        m = onnx_model(a.w, SHAPE[a.cam]); t = run_stream(m, frames(a.cam), SIZE[a.cam], a.runs)
        r = {"mode": "single", "tag": a.tag, "cam": a.cam, "e2e_ms_median": float(np.median(t)), "e2e_ms_p90": float(np.percentile(t, 90))}
    elif a.mode == "pair":
        import multiprocessing as mp
        for w, cam in ((a.w_ir, "thermal"), (a.w_rgb, "rgb")):
            onnx_model(w, SHAPE[cam])  # export once before the timed run
        with mp.get_context("spawn").Pool(2) as pool:
            res = pool.starmap(_pair_worker, [(a.w_ir, "thermal", a.runs), (a.w_rgb, "rgb", a.runs)])
        r = {"mode": "pair", "tag": a.tag, **{f"{cam}_e2e_ms_median": ms for cam, ms, _ in res},
             **{f"{cam}_ram_extra_MB": mb for cam, _, mb in res}, "pair_ram_extra_MB": sum(mb for *_, mb in res),
             "pair_fps": 1000 / max(ms for _, ms, _ in res)}
    elif a.mode == "memory":
        import gc
        ims = {c: frames(c, 10) for c in ("thermal", "rgb")}
        gc.collect(); base = psutil.Process().memory_info().rss; p0 = peak()
        if a.w_shared:
            m = onnx_model(a.w_shared, SHAPE["rgb"], dynamic=True)
            for c in ("thermal", "rgb"):
                run_stream(m, ims[c], SIZE[c], 15)
            kind = "shared (one dynamic-shape network)"
        else:
            mi, mr = onnx_model(a.w_ir, SHAPE["thermal"]), onnx_model(a.w_rgb, SHAPE["rgb"])
            run_stream(mi, ims["thermal"], 640, 15); run_stream(mr, ims["rgb"], 960, 15)
            kind = "two separate networks"
        r = {"mode": "memory", "tag": a.tag, "kind": kind, "ram_extra_MB": (max(peak(), p0) - base) / 2 ** 20}
    else:
        raise SystemExit("unknown mode")
    with open(a.out, "a") as f:
        f.write(json.dumps(r) + "\n")
    print(json.dumps(r))


def _pair_worker(w, cam, n):
    import gc
    ims = frames(cam)
    gc.collect(); base = psutil.Process().memory_info().rss; p0 = peak()
    m = onnx_model(w, SHAPE[cam])
    t = run_stream(m, ims, SIZE[cam], n)
    return cam, float(np.median(t)), (max(peak(), p0) - base) / 2 ** 20


if __name__ == "__main__":
    main()
