#!/usr/bin/env python3
"""
Small-drone trial (RGB, or thermal with --modality IR): does a quarter-size RGB detector match YOLO11s, and do a higher input size or an extra
P2 (stride-4) head help small drones? Trains seed-0 variants with the main-study recipe (30 epochs, patience 10,
cosine LR) on the prepared Anti-UAV300 RGB set and scores the official test split with the paper's evaluator,
in full-resolution RGB coordinates. Reference: the main-study YOLO11s RGB detector at 960 px.

python rgb_small_trial.py --work /tmp/skyfuse_au --out OUT --devices 0,1
"""
import argparse, json, os, pickle, subprocess, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# name, model, image set (RGBs = 960-px-wide copy, RGB = native 1920 px, IR = native 640x512), imgsz, batch
VARIANTS = {"RGB": [
    ("n960", "yolo11n.pt", "RGBs", 960, 16),
    ("n1280", "yolo11n.pt", "RGB", 1280, 16),
    ("nP2_960", "yolo11n-p2.yaml", "RGBs", 960, 16),
    ("sP2_960", "yolo11s-p2.yaml", "RGBs", 960, 8),
    ("s1280", "yolo11s.pt", "RGB", 1280, 8),
], "IR": [
    ("ir_s640", "yolo11s.pt", "IR", 640, 16),  # same-run replicate of the main-study thermal detector (weights for cueing)
    ("ir_n640", "yolo11n.pt", "IR", 640, 16),
    ("ir_nP2_640", "yolo11n-p2.yaml", "IR", 640, 16),
    ("ir_sP2_640", "yolo11s-p2.yaml", "IR", 640, 16),
], "JOINT": [
    # one detector trained on thermal and RGB frames together (shared weights), applied to each camera separately
    ("joint_n960", "yolo11n.pt", "JOINT", 960, 16),
], "JOINTPAD": [
    # as JOINT, but thermal frames are padded (not enlarged) onto a 960x768 canvas, so thermal drones keep their size
    ("jointpad_n960", "yolo11n.pt", "JOINTPAD", 960, 16),
], "NANO3": [
    # three seeds of the separate nano detectors (seed 0 re-trained so that its last-epoch checkpoint exists)
    *[(f"ir_n640_s{sd}", "yolo11n.pt", "IR", 640, 16, {"seed": sd}) for sd in (0, 1, 2)],
    *[(f"rgb_n960_s{sd}", "yolo11n.pt", "RGBs", 960, 16, {"seed": sd}) for sd in (0, 1, 2)],
], "LOWRES": [
    # 320-px inputs, as in the published single-camera YOLOv5s baselines (yolov5su = Ultralytics anchor-free YOLOv5s)
    ("ir_n320", "yolo11n.pt", "IR", 320, 16), ("rgb_n320", "yolo11n.pt", "RGBs", 320, 16),
    ("ir_v5s320", "yolov5su.pt", "IR", 320, 16), ("rgb_v5s320", "yolov5su.pt", "RGBs", 320, 16),
]}


def pad_thermal(prep, W=960, H=768, fill=114):
    """Thermal training/validation frames centred on a WxH grey canvas (native scale), with shifted YOLO labels."""
    import cv2
    src, dst = os.path.join(prep, "IR"), os.path.join(prep, "IRpad")
    os.makedirs(os.path.join(dst, "images"), exist_ok=True); os.makedirs(os.path.join(dst, "labels"), exist_ok=True)
    for sp in ("train", "val_sub"):
        out = []
        for ip in open(os.path.join(src, sp + ".txt")).read().split():
            k = os.path.splitext(os.path.basename(ip))[0]
            op = os.path.join(dst, "images", k + ".jpg")
            if not os.path.exists(op):
                im = cv2.imread(ip); h, w = im.shape[:2]
                ox, oy = (W - w) // 2, (H - h) // 2
                c = np.full((H, W, 3), fill, np.uint8); c[oy:oy + h, ox:ox + w] = im
                cv2.imwrite(op, c, [cv2.IMWRITE_JPEG_QUALITY, 95])
                lp = os.path.join(src, "labels", k + ".txt"); rows = []
                if os.path.exists(lp):
                    for ln in open(lp).read().splitlines():
                        if ln.strip():
                            c_, x, y, bw, bh = ln.split()
                            rows.append(f"{c_} {(float(x) * w + ox) / W:.6f} {(float(y) * h + oy) / H:.6f} "
                                        f"{float(bw) * w / W:.6f} {float(bh) * h / H:.6f}")
                open(os.path.join(dst, "labels", k + ".txt"), "w").write("\n".join(rows) + ("\n" if rows else ""))
            out.append(op)
        open(os.path.join(dst, sp + ".txt"), "w").write("\n".join(out) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--devices", default="0"); ap.add_argument("--epochs", default="30"); ap.add_argument("--only", default=None, help="comma-separated variant names")
    ap.add_argument("--modality", default="RGB", choices=["RGB", "IR", "JOINT", "JOINTPAD", "NANO3", "LOWRES"])
    a = ap.parse_args()
    import antiuav_pipeline as P
    from analyze_antiuav import suite
    prep = os.path.join(a.work, "prep")
    runs = os.path.join(a.out, "runs")
    gt, meta = P.load_prepared(prep)
    if a.modality == "JOINTPAD":
        pad_thermal(prep)
    if a.modality in ("JOINT", "JOINTPAD"):  # union of the thermal and the 960-px RGB training/validation lists
        jd = os.path.join(prep, a.modality); os.makedirs(jd, exist_ok=True)
        for sp in ("train", "val_sub"):
            with open(os.path.join(jd, sp + ".txt"), "w") as f:
                for m in (("IR" if a.modality == "JOINT" else "IRpad"), "RGBs"):
                    f.write(open(os.path.join(prep, m, sp + ".txt")).read().strip() + "\n")
        with open(os.path.join(jd, "data.yaml"), "w") as f:
            f.write(f"path: {os.path.abspath(jd)}\ntrain: train.txt\nval: val_sub.txt\nnames:\n  0: drone\n")
    var = [v for v in VARIANTS[a.modality] if a.only is None or v[0] in a.only.split(",")]
    jobs = []
    var = [v if len(v) == 6 else v + ({},) for v in var]
    for name, model, imgs, imgsz, batch, extra in var:
        mpath = os.path.join(HERE, model) if model.endswith(".yaml") else model  # custom yaml lives next to this script
        j = {"name": name, "model": mpath, "data": os.path.join(prep, imgs, "data.yaml"), "imgsz": imgsz, "batch": batch, **extra}
        if model.endswith(".yaml"):  # P2 variants: transfer the matching COCO weights of the same scale
            j["args"] = {"pretrained": model.replace("-p2.yaml", ".pt")}
        jobs.append(j)
    devs = a.devices.split(",")
    common = [sys.executable, os.path.join(HERE, "train_queue.py"), "--project", runs, "--seed", "0", "--epochs", a.epochs,
              "--patience", "10", "--workers", "4", "--jobs", json.dumps(jobs), "--claim_dir", os.path.join(runs, "claims")]
    t0 = time.time()
    procs = [subprocess.Popen(common + ["--device", d]) for d in devs]
    codes = [p.wait() for p in procs]
    print("training exit codes", codes, round((time.time() - t0) / 60, 1), "min", flush=True)

    from ultralytics import YOLO
    import torch
    dev = "cpu" if devs[0] == "cpu" else 0
    sync = torch.cuda.synchronize if dev == 0 else (lambda: None)
    keys = sorted(k for k, v in meta["split_of"].items() if v == "test")
    res = {}
    dets_all = {}
    import shutil

    def evaluate(m, wp, name, train_name, model, imgs, gmod, imgsz, batch):
        paths = [os.path.join(prep, imgs, "images", k + ".jpg") for k in keys]
        scale = meta["rgb_scale"] if imgs == "RGBs" else None
        d = P.infer(m, paths, imgsz, device=dev, scale=scale)
        dets_all[name] = d
        s = suite(d, gt[gmod], keys)
        for _ in range(5):  # warm-up
            m.predict(paths[0], imgsz=imgsz, device=dev, verbose=False)
        sync(); t = time.time()
        for _ in range(50):
            m.predict(paths[0], imgsz=imgsz, device=dev, verbose=False)
        sync()
        lat = (time.time() - t) / 50 * 1000
        done = os.path.join(runs, train_name, "DONE")
        csv = os.path.join(runs, train_name, "results.csv")
        res[name] = {"model": model, "images": imgs, "camera": gmod, "imgsz": imgsz, "batch": batch,
                     "params": int(sum(p.numel() for p in m.model.parameters())),
                     "train_seconds": float(open(done).read() or 0) if os.path.exists(done) else None,
                     "epochs_done": max(len(open(csv).read().strip().splitlines()) - 1, 0) if os.path.exists(csv) else None,
                     "latency_ms_batch1_" + str(dev): lat, "metrics": {k: float(v) for k, v in s.items()}}
        print(name, json.dumps({k: round(v, 3) for k, v in res[name]["metrics"].items() if k.startswith("AP")}), flush=True)

    for name, model, imgs, imgsz, batch, extra in var:
        w = os.path.join(runs, name, "weights")
        wp = os.path.join(w, "best.pt") if os.path.exists(os.path.join(w, "best.pt")) else os.path.join(w, "last.pt")
        if not os.path.exists(wp):
            res[name] = {"error": "no weights"}; continue
        m = YOLO(wp)
        if imgs in ("JOINT", "JOINTPAD"):  # the joint model is scored on each camera at that camera's main-study input size
            evals = [(name + "_on_IR", "IR", "IR", 640), (name + "_on_IR960", "IR", "IR", 960),
                     (name + "_on_RGB", "RGBs", "RGB", 960)]
        else:
            evals = [(name, imgs, "IR" if imgs == "IR" else "RGB", imgsz)]
        for ename, eimgs, gmod, esz in evals:
            evaluate(m, wp, ename, name, model, eimgs, gmod, esz, batch)
        lp = os.path.join(w, "last.pt")  # last-epoch checkpoint: selection-free alternative to best-validation
        if a.modality in ("NANO3", "LOWRES") and os.path.exists(lp) and lp != wp:
            ml = YOLO(lp)
            for ename, eimgs, gmod, esz in evals:
                evaluate(ml, lp, ename + "_last", name, model, eimgs, gmod, esz, batch)
        os.makedirs(os.path.join(a.out, "weights"), exist_ok=True)
        shutil.copy(wp, os.path.join(a.out, "weights", name + ".pt"))
    stem = {"RGB": "rgb_small_trial", "IR": "ir_small_trial", "JOINT": "joint_trial", "JOINTPAD": "jointpad_trial",
            "NANO3": "nano3_trial", "LOWRES": "lowres_trial"}[a.modality]
    json.dump(res, open(os.path.join(a.out, stem + ".json"), "w"), indent=1)
    pickle.dump(dets_all, open(os.path.join(a.out, stem + "_test_dets.pkl"), "wb"), protocol=4)
    print(json.dumps(res, indent=1)[:6000])


if __name__ == "__main__":
    main()
