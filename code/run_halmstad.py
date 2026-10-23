#!/usr/bin/env python3
"""
Train and evaluate single-modality YOLO detectors on Halmstad (resumable).

Each config: sensor (IR|V), protocol (clip|random), imgsz, epochs.
Writes results/<name>.json with overall, per-class and (clip protocol) per-distance test metrics,
plus CPU/GPU latency. Skips configs whose result file already exists; resumes interrupted training.

Usage: python run_halmstad.py --data halm_s30 --out results_halm --configs IR:clip:320:12 IR:random:320:12 ...
"""
import argparse, json, os, time

from ultralytics import YOLO

CLASSES = ["airplane", "bird", "drone", "helicopter"]


def metrics_dict(m):
    box = m.box
    per = {}
    for i, c in enumerate(CLASSES):
        try:
            idx = list(box.ap_class_index).index(i)
            per[c] = {"ap50": float(box.ap50[idx]), "ap50_95": float(box.ap[idx])}
        except ValueError:
            per[c] = None
    return {"map50": float(box.map50), "map50_95": float(box.map), "precision": float(box.mp),
            "recall": float(box.mr), "per_class": per}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budget_h", type=float, default=0)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    T0 = time.time()
    for cfg in a.configs:
        if a.budget_h and time.time() - T0 > a.budget_h * 3600:
            print("budget reached, skipping", cfg); continue
        sensor, proto, imgsz, epochs, seed = (cfg.split(":") + [str(a.seed)])[:5]
        imgsz, epochs, seed = int(imgsz), int(epochs), int(seed)
        name = f"{sensor}_{proto}_{imgsz}_e{epochs}_s{seed}"
        res_path = os.path.join(a.out, name + ".json")
        if os.path.exists(res_path):
            print("skip", name); continue
        data = os.path.join(a.data, sensor, f"{proto}.yaml")
        proj = os.path.abspath(os.path.join(a.out, "runs"))
        last = os.path.join(proj, name, "weights", "last.pt")
        best = os.path.join(proj, name, "weights", "best.pt")
        t0 = time.time()
        if os.path.exists(last) and not os.path.exists(os.path.join(proj, name, "DONE")):
            YOLO(last).train(resume=True)
        elif not os.path.exists(os.path.join(proj, name, "DONE")):
            YOLO(a.model).train(data=data, imgsz=imgsz, epochs=epochs, batch=a.batch, workers=2,
                                device=a.device, project=proj, name=name, exist_ok=True, seed=seed,
                                deterministic=True, patience=100, plots=False, verbose=False)
        open(os.path.join(proj, name, "DONE"), "w").close()
        train_s = time.time() - t0
        model = YOLO(best if os.path.exists(best) else last)
        out = {"name": name, "sensor": sensor, "protocol": proto, "imgsz": imgsz, "epochs": epochs,
               "seed": seed, "model": a.model, "train_seconds": round(train_s, 1)}
        out["test"] = metrics_dict(model.val(data=data, split="test", imgsz=imgsz, batch=a.batch,
                                             device=a.device, plots=False, verbose=False))
        if proto == "random":  # same model scored on clips it has never seen
            out["test_unseen_clips"] = metrics_dict(model.val(data=os.path.join(a.data, sensor, "random_unseen.yaml"),
                                                              split="test", imgsz=imgsz, batch=a.batch, device=a.device,
                                                              plots=False, verbose=False))
        if True:
            out["test_by_distance"] = {}
            for dist in ["close", "medium", "distant"]:
                dy = os.path.join(a.data, sensor, f"clip_test_{dist}.yaml")
                if not os.path.exists(os.path.join(a.data, sensor, f"clip_test_{dist}.txt")):
                    continue
                out["test_by_distance"][dist] = metrics_dict(
                    model.val(data=dy, split="test", imgsz=imgsz, batch=a.batch, device=a.device,
                              plots=False, verbose=False))
        # latency on 50 test images, batch 1
        with open(os.path.join(a.data, sensor, f"{proto}_test.txt")) as f:
            imgs = [l.strip() for l in f if l.strip()][:50]
        model.predict(imgs[:3], imgsz=imgsz, device=a.device, verbose=False)
        t = time.time()
        for im in imgs:
            model.predict(im, imgsz=imgsz, device=a.device, verbose=False)
        out["latency_ms_per_frame"] = round(1000 * (time.time() - t) / len(imgs), 2)
        out["latency_device"] = a.device
        # training curve
        rc = os.path.join(proj, name, "results.csv")
        if os.path.exists(rc):
            out["results_csv"] = open(rc).read()
        with open(res_path, "w") as f:
            json.dump(out, f, indent=2)
        print(json.dumps({k: v for k, v in out.items() if k != "results_csv"}, indent=2))


if __name__ == "__main__":
    main()
