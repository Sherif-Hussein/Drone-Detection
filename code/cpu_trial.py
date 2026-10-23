#!/usr/bin/env python3
"""
CPU-only trial: train a small thermal detector (YOLO11n by default) on the prepared Anti-UAV300 IR set and score it on
the official test split with the same evaluator as the paper. Training time is capped (Ultralytics `time`, hours) so
the run fits in one CPU session; per-epoch time is logged to judge training without a GPU.

python cpu_trial.py --work /tmp/skyfuse_au --out OUT [--model yolo11n.pt] [--epochs 30] [--hours 10] [--imgsz 640]
"""
import argparse, json, os, time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="yolo11n.pt"); ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hours", type=float, default=10.0); ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    import torch
    from ultralytics import YOLO
    import antiuav_pipeline as P
    from analyze_antiuav import suite
    torch.set_num_threads(os.cpu_count())
    prep = os.path.join(a.work, "prep")
    gt, meta = P.load_prepared(prep)
    data = os.path.join(prep, "IR", "data.yaml")
    t0 = time.time()
    m = YOLO(a.model)
    m.train(data=data, imgsz=a.imgsz, epochs=a.epochs, time=a.hours, device="cpu", batch=a.batch, workers=a.workers,
            seed=0, deterministic=True, patience=10, project=a.out, name="train", exist_ok=True, plots=False, verbose=False)
    train_h = (time.time() - t0) / 3600
    csv = os.path.join(a.out, "train", "results.csv")
    rows = open(csv).read().strip().splitlines() if os.path.exists(csv) else []
    epochs_done = max(len(rows) - 1, 0)
    best = os.path.join(a.out, "train", "weights", "best.pt")
    m = YOLO(best if os.path.exists(best) else os.path.join(a.out, "train", "weights", "last.pt"))
    keys = sorted(k for k, v in meta["split_of"].items() if v == "test")
    paths = [os.path.join(prep, "IR", "images", k + ".jpg") for k in keys]
    t1 = time.time()
    dets = P.infer(m, paths, a.imgsz, device="cpu", batch=8)
    infer_ms = (time.time() - t1) / len(paths) * 1000
    s = suite(dets, gt["IR"], keys)
    res = {"model": a.model, "imgsz": a.imgsz, "cpu_threads": os.cpu_count(), "epochs_done": epochs_done,
           "train_hours": train_h, "hours_per_epoch": train_h / max(epochs_done, 1),
           "test_AP50": float(s["AP50"]), "test_AP50_95": float(s.get("AP50_95", np.nan)), "test_AP75": float(s.get("AP75", np.nan)),
           "infer_ms_per_frame_cpu_batch8": infer_ms, "n_test_frames": len(keys),
           "reference_main_study_yolo11s_gpu": 0.882}
    json.dump(res, open(os.path.join(a.out, "cpu_trial.json"), "w"), indent=1)
    import pickle
    pickle.dump(dets, open(os.path.join(a.out, "cpu_trial_test_dets.pkl"), "wb"))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
