#!/usr/bin/env python3
"""
Single-detector ablation on Anti-UAV300 (one change at a time, seed 0): which detector or preprocessing change
improves the thermal (or RGB) detector? Trains the listed variants on two GPUs (shared job queue), then saves raw
val/test detections and Ultralytics validation metrics for offline evaluation.

Variants (name -> settings):
  ir_fe   : thermal frames as 3 engineered channels [raw, CLAHE, white top-hat]
  ir_960  : thermal frames up-sampled to a 960-pixel network input
  ir_m    : YOLO11-medium instead of YOLO11-small
  rgb_1280: RGB detector at 1280 pixels (from a 1280-wide copy of the full-resolution frame)
  rgb_m   : RGB detector with YOLO11-medium

python run_ablation.py --work /tmp/skyfuse_au --out OUT --variants ir_fe ir_960 ir_m [--devices 0,1] [--epochs 30]
"""
import argparse, json, os, pickle, shutil, subprocess, sys, time

import cv2
import numpy as np

import antiuav_pipeline as P

HERE = os.path.dirname(os.path.abspath(__file__))


def fe_image(im, clip=2.0, tophat=31):
    """Feature-engineered thermal image: raw, CLAHE (local contrast) and white top-hat (small bright blobs)."""
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(g)
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tophat, tophat)))
    hi = max(float(np.percentile(th, 99.9)), 1.0)
    th = np.clip(th.astype(np.float32) * (255.0 / hi), 0, 255).astype(np.uint8)
    return cv2.merge([g, c, th])


_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_RECT31 = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))


def fe_image_fast(im):
    """Fast version used in the final model (about 1 ms per 640x512 frame on a laptop CPU): CLAHE and a separable
    rectangular white top-hat (correlation 0.98 with the elliptical one) with a fixed gain instead of a percentile."""
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im
    th = cv2.convertScaleAbs(cv2.morphologyEx(g, cv2.MORPH_TOPHAT, _RECT31), alpha=4.0)
    return cv2.merge([g, _CLAHE.apply(g), th])


def derived_set(prep, out, src, fn, label_scale=None):
    """Write a derived copy of a prepared image set (images via fn, labels copied), with the same lists."""
    os.makedirs(os.path.join(out, "images"), exist_ok=True); os.makedirs(os.path.join(out, "labels"), exist_ok=True)
    lists = {}
    for sp in ("train", "val", "test", "val_sub"):
        paths = [l.strip() for l in open(os.path.join(prep, src, f"{sp}.txt")) if l.strip()]
        new = []
        for p in paths:
            k = os.path.splitext(os.path.basename(p))[0]
            ip = os.path.join(out, "images", k + ".jpg")
            if not os.path.exists(ip):
                cv2.imwrite(ip, fn(cv2.imread(p), k), [cv2.IMWRITE_JPEG_QUALITY, 92])
                lp = p.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep).replace("/images/", "/labels/")[:-4] + ".txt"
                with open(lp) as f, open(os.path.join(out, "labels", k + ".txt"), "w") as g:
                    g.write(f.read())  # YOLO labels are normalised, so they are valid for any image size
            new.append(ip)
        lists[sp] = new
        with open(os.path.join(out, f"{sp}.txt"), "w") as f:
            f.write("\n".join(new) + "\n")
    with open(os.path.join(out, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(out)}\ntrain: train.txt\nval: val_sub.txt\ntest: test.txt\nnames:\n  0: drone\n")
    return lists


def session_of(k):
    """Recording session of a frame key: '<split>_<date>_<time>_...' -> '<date>_<time>'."""
    return "_".join(P.seq_of(k).split("_", 1)[1].split("_")[:2])


def sinus_remap(im, k):
    """Thermal intensity remapping (sinusoidal), a training-time augmentation for thermal-contrast variation."""
    rng = np.random.default_rng(abs(hash(k)) % (2 ** 32))
    g = (cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im).astype(np.float32) / 255.0
    a, f, ph = rng.uniform(0.08, 0.25), rng.uniform(0.5, 1.5), rng.uniform(0, 2 * np.pi)
    y = np.clip(g + a * np.sin(2 * np.pi * f * g + ph), 0, 1)
    y = (y - y.min()) / max(float(y.max() - y.min()), 1e-6)
    return cv2.cvtColor((y * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def write_session_split(d, select, extra_train=()):
    """Training lists without the selected sessions; selection lists = all frames of the selected sessions."""
    trv = [l.strip() for sp in ("train", "val") for l in open(os.path.join(d, f"{sp}.txt")) if l.strip()]
    key = lambda p: os.path.splitext(os.path.basename(p))[0]
    tr = sorted([p for p in trv if session_of(key(p)) not in select] + list(extra_train))
    sel = sorted(p for p in trv if session_of(key(p)) in select)
    for name, lst in (("train_sd", tr), ("sel", sel), ("sel_sub", sel[::2])):
        with open(os.path.join(d, f"{name}.txt"), "w") as f:
            f.write("\n".join(lst) + "\n")
    with open(os.path.join(d, "data_sd.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(d)}\ntrain: train_sd.txt\nval: sel_sub.txt\ntest: test.txt\nnames:\n  0: drone\n")
    return os.path.join(d, "data_sd.yaml"), len(tr), len(sel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--variants", nargs="+", required=True); ap.add_argument("--devices", default="0,1")
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="train every variant for each of these seeds")
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--select_sessions", type=int, default=6,
                    help="hold out this many whole train/val recording sessions for model selection (0 = official val)")
    a = ap.parse_args()
    prep = os.path.join(a.work, "prep"); os.makedirs(a.out, exist_ok=True)
    gt, meta = P.load_prepared(prep)
    t0 = time.time()
    jobs, setups = [], {}
    trv_sessions = sorted({session_of(k) for k, sp in meta["split_of"].items() if sp in ("train", "val")})
    rng = np.random.default_rng(1234)
    select = set(rng.choice(trv_sessions, size=a.select_sessions, replace=False).tolist()) if a.select_sessions else set()
    split_info = {"selection_sessions": sorted(select), "n_trainval_sessions": len(trv_sessions)}
    print("[split] selection sessions:", sorted(select), flush=True)
    for v in a.variants:
        if v == "ir_base":
            setups[v] = (os.path.join(prep, "IR", "data.yaml"), 640, "yolo11s.pt", os.path.join(prep, "IR"), None)
        elif v == "ir_a410":  # extra thermal scenes: the 100 Anti-UAV410 sequences not in Anti-UAV300
            xl = os.path.join(a.work, "a410x", "extra_train.txt")
            if not os.path.exists(xl):
                print("[skip] ir_a410: no extra data at", xl, flush=True); continue
            extra = [l.strip() for l in open(xl) if l.strip()]
            d = os.path.join(a.work, "abl_ir_a410"); os.makedirs(d, exist_ok=True)  # own list files (images stay in place)
            for sp in ("train", "val", "test"):
                with open(os.path.join(d, f"{sp}.txt"), "w") as f:
                    f.write(open(os.path.join(prep, "IR", f"{sp}.txt")).read())
            setups[v] = (None, 640, "yolo11s.pt", d, None, {"_extra_train": extra})
            split_info["a410_extra_frames"] = len(extra)
        elif v == "ir_sin":
            d = os.path.join(a.work, "abl_ir_sin")
            os.makedirs(os.path.join(d, "images"), exist_ok=True); os.makedirs(os.path.join(d, "labels"), exist_ok=True)
            extra = []
            for p in [l.strip() for sp in ("train", "val") for l in open(os.path.join(prep, "IR", f"{sp}.txt")) if l.strip()]:
                k = os.path.splitext(os.path.basename(p))[0]
                if session_of(k) in select:
                    continue
                ip = os.path.join(d, "images", k + "_sin.jpg")
                if not os.path.exists(ip):
                    cv2.imwrite(ip, sinus_remap(cv2.imread(p), k), [cv2.IMWRITE_JPEG_QUALITY, 95])
                    with open(os.path.join(prep, "IR", "labels", k + ".txt")) as f, open(os.path.join(d, "labels", k + "_sin.txt"), "w") as g:
                        g.write(f.read())
                extra.append(ip)
            for sp in ("train", "val", "test"):
                with open(os.path.join(d, f"{sp}.txt"), "w") as f:
                    f.write(open(os.path.join(prep, "IR", f"{sp}.txt")).read())
            setups[v] = (None, 640, "yolo11s.pt", d, None, {"_extra_train": extra})
        elif v in ("ir_combo", "ir_a410fe", "ir_a410fe_ell"):
            # extra Anti-UAV410 scenes + engineered channels; ir_combo also uses the stronger augmentation (E3);
            # ir_a410fe_ell uses the original elliptical, percentile-normalised top-hat instead of the fast one
            fn = fe_image if v == "ir_a410fe_ell" else (lambda im: fe_image_fast(im))
            d = os.path.join(a.work, f"abl_{v}")
            if not os.path.exists(os.path.join(d, "data.yaml")):
                derived_set(prep, d, "IR", lambda im, k: fn(im))
            extra = []
            xl = os.path.join(a.work, "a410x", "extra_train.txt")
            for p in ([l.strip() for l in open(xl) if l.strip()] if os.path.exists(xl) else []):
                k = os.path.splitext(os.path.basename(p))[0]
                ip = os.path.join(d, "images", k + ".jpg")
                if not os.path.exists(ip):
                    cv2.imwrite(ip, fn(cv2.imread(p)), [cv2.IMWRITE_JPEG_QUALITY, 92])
                    lp = p.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep).replace("/images/", "/labels/")[:-4] + ".txt"
                    with open(lp) as f, open(os.path.join(d, "labels", k + ".txt"), "w") as g:
                        g.write(f.read())
                extra.append(ip)
            split_info[f"{v}_extra_frames"] = len(extra)
            cfg = {"_extra_train": extra}
            if v == "ir_combo":
                cfg["args"] = {"scale": 0.9, "multi_scale": 0.25, "mixup": 0.1, "degrees": 5.0, "translate": 0.2,
                               "hsv_v": 0.6, "close_mosaic": 15}
            setups[v] = (None, 640, "yolo11s.pt", d, None, cfg)
        elif v == "ir_fe":
            d = os.path.join(a.work, "abl_ir_fe")
            if not os.path.exists(os.path.join(d, "data.yaml")):
                derived_set(prep, d, "IR", lambda im, k: fe_image(im))
            setups[v] = (os.path.join(d, "data.yaml"), 640, "yolo11s.pt", d, None)
        elif v == "ir_aug":  # stronger training-time augmentation only (inference unchanged)
            setups[v] = (os.path.join(prep, "IR", "data.yaml"), 640, "yolo11s.pt", os.path.join(prep, "IR"), None,
                         {"args": {"scale": 0.9, "multi_scale": 0.25, "mixup": 0.1, "degrees": 5.0, "translate": 0.2,
                                   "hsv_v": 0.6, "close_mosaic": 15}})
        elif v == "ir_e80":  # longer training only
            setups[v] = (os.path.join(prep, "IR", "data.yaml"), 640, "yolo11s.pt", os.path.join(prep, "IR"), None,
                         {"epochs": 80, "patience": 20})
        elif v == "ir_960":
            setups[v] = (os.path.join(prep, "IR", "data.yaml"), 960, "yolo11s.pt", os.path.join(prep, "IR"), None)
        elif v == "ir_m":
            setups[v] = (os.path.join(prep, "IR", "data.yaml"), 640, "yolo11m.pt", os.path.join(prep, "IR"), None)
        elif v == "rgb_1280":
            d = os.path.join(a.work, "abl_rgb_1280")
            if not os.path.exists(os.path.join(d, "data.yaml")):
                def big(im_small, k):
                    full = cv2.imread(os.path.join(prep, "RGB", "images", k + ".jpg"))
                    H, W = full.shape[:2]
                    return cv2.resize(full, (1280, int(round(H * 1280 / W))), interpolation=cv2.INTER_AREA)
                derived_set(prep, d, "RGBs", big)
            setups[v] = (os.path.join(d, "data.yaml"), 1280, "yolo11s.pt", d, {k: s * 960 / 1280 for k, s in meta["rgb_scale"].items()})
        elif v == "rgb_m":
            setups[v] = (os.path.join(prep, "RGBs", "data.yaml"), 960, "yolo11m.pt", os.path.join(prep, "RGBs"), meta["rgb_scale"])
        else:
            raise SystemExit("unknown variant " + v)
        extra = setups[v][5] if len(setups[v]) > 5 else {}
        setups[v] = setups[v][:5]
        if a.select_sessions:  # session-disjoint selection split for every variant
            yml, ntr, nsel = write_session_split(setups[v][3], select, extra.pop("_extra_train", []))
            setups[v] = (yml,) + setups[v][1:]
            split_info[v] = {"train_frames": ntr, "selection_frames": nsel}
        else:  # official split: train (+ extra data), official validation for selection
            xt = extra.pop("_extra_train", [])
            d = setups[v][3]
            tr = [l.strip() for l in open(os.path.join(d, "train.txt")) if l.strip()] + list(xt)
            with open(os.path.join(d, "train_x.txt"), "w") as f:
                f.write("\n".join(tr) + "\n")
            if not os.path.exists(os.path.join(d, "val_sub.txt")):
                vl = [l.strip() for l in open(os.path.join(d, "val.txt")) if l.strip()]
                with open(os.path.join(d, "val_sub.txt"), "w") as f:
                    f.write("\n".join(vl[::2]) + "\n")
            with open(os.path.join(d, "data_x.yaml"), "w") as f:
                f.write(f"path: {os.path.abspath(d)}\ntrain: train_x.txt\nval: val_sub.txt\ntest: test.txt\nnames:\n  0: drone\n")
            setups[v] = (os.path.join(d, "data_x.yaml"),) + setups[v][1:]
            split_info[v] = {"train_frames": len(tr), "protocol": "official"}
        for sd in (a.seeds or [a.seed]):
            job = {"name": f"abl_{v}_s{sd}", "data": setups[v][0], "imgsz": setups[v][1], "model": setups[v][2], "seed": sd}
            job.update(extra)
            jobs.append(job)
    print("[prep] derived sets ready", round(time.time() - t0), "s", flush=True)
    jobs.sort(key=lambda j: -(j["imgsz"] ** 2) * (3 if "11m" in j["model"] else 1) * j.get("epochs", a.epochs))  # longest first
    runs = os.path.abspath(os.path.join(a.work, "runs_abl"))
    common = [sys.executable, os.path.join(HERE, "train_queue.py"), "--project", runs, "--seed", str(a.seed),
              "--epochs", str(a.epochs), "--batch", str(a.batch), "--workers", str(a.workers), "--jobs", json.dumps(jobs),
              "--claim_dir", os.path.join(runs, "claims")]
    procs = [subprocess.Popen(common + ["--device", d]) for d in a.devices.split(",")]
    print("train exit codes", [p.wait() for p in procs], flush=True)
    from ultralytics import YOLO
    dev = 0 if a.devices != "cpu" else "cpu"
    for sd in (a.seeds or [a.seed]):
        a.seed = sd
        evaluate_seed(a, setups, runs, split_info, dev, YOLO, t0)
    print("done", round((time.time() - t0) / 60, 1), "min")


def evaluate_seed(a, setups, runs, split_info, dev, YOLO, t0):
    res = {"config": vars(a), "split": split_info, "variants": {}}
    for v in a.variants:
        if v not in setups:
            continue
        data, imgsz, model, d, scale = setups[v]
        w = os.path.join(runs, f"abl_{v}_s{a.seed}", "weights")
        if not os.path.exists(os.path.join(w, "best.pt")) and not os.path.exists(os.path.join(w, "last.pt")):
            res["variants"][v] = {"error": "no weights"}; continue
        m = YOLO(os.path.join(w, "best.pt") if os.path.exists(os.path.join(w, "best.pt")) else os.path.join(w, "last.pt"))
        dets = {}
        for sp in ("val", "test"):
            paths = [l.strip() for l in open(os.path.join(d, f"{sp}.txt")) if l.strip()]
            dets[sp] = P.infer(m, paths, imgsz, dev, scale=scale)
        with open(os.path.join(a.out, f"ablation_{v}_s{a.seed}.pkl"), "wb") as f:
            pickle.dump(dets, f, protocol=4)
        shutil.copyfile(os.path.join(w, "best.pt") if os.path.exists(os.path.join(w, "best.pt")) else os.path.join(w, "last.pt"),
                        os.path.join(a.out, f"weights_{v}_s{a.seed}.pt"))
        if a.select_sessions:  # validate on the FULL session-disjoint selection set
            full = data.replace("data_sd.yaml", "data_sel_full.yaml")
            with open(full, "w") as f:
                f.write(open(data).read().replace("val: sel_sub.txt", "val: sel.txt"))
            data = full
        vm = m.val(data=data, split="val", imgsz=imgsz, batch=a.batch, device=dev, plots=False, verbose=False)
        rc = os.path.join(runs, f"abl_{v}_s{a.seed}", "results.csv")
        res["variants"][v] = {"val_map50": float(vm.box.map50), "val_map50_95": float(vm.box.map), "imgsz": imgsz, "model": model,
                              "train_seconds": float(open(os.path.join(runs, f"abl_{v}_s{a.seed}", "DONE")).read() or 0),
                              "results_csv": open(rc).read() if os.path.exists(rc) else None}
        t = time.time(); [m.predict(paths[0], imgsz=imgsz, device=dev, verbose=False) for _ in range(30)]
        res["variants"][v]["latency_ms"] = 1000 * (time.time() - t) / 30
        json.dump(res, open(os.path.join(a.out, f"ablation_s{a.seed}.json"), "w"), indent=1)
        print(v, a.seed, {k: x for k, x in res["variants"][v].items() if k != "results_csv"}, flush=True)


if __name__ == "__main__":
    main()
