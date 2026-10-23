#!/usr/bin/env python3
"""
GPU stage of the Anti-UAV300 fusion-level study for ONE training seed.

Expects the output of antiuav_prep.py in --work (prep/, crops/, pixfuse/). Trains the IR, RGB, crop-verifier
and pixel-fusion detectors (two parallel queues when two GPUs are present), runs every inference the analysis
needs, saves all raw detections (detections_s<seed>.pkl) and evaluates single sensors and all fusion levels.
Results are written incrementally to results_antiuav_s<seed>.json; each analysis block is isolated so that
a failure in one block does not lose the others (the raw detections allow re-running the analysis offline).

python run_antiuav.py --work /tmp/skyfuse_au --out OUT --seed 0 [--robustness] [--devices 0,1]
"""
import argparse, json, os, pickle, shutil, subprocess, sys, time, traceback

import numpy as np

import antiuav_pipeline as P
from fusion_core import AxisAffine, fuse_many

RULES = ["or", "and", "max", "wbf", "proben", "learned"]
CONDS = ["rgb_dark", "rgb_fog", "rgb_blur", "ir_noise", "ir_lowcontrast", "rgb_missing", "ir_missing"]
HERE = os.path.dirname(os.path.abspath(__file__))


class Results:
    def __init__(self, path):
        self.path = path
        self.d = json.load(open(path)) if os.path.exists(path) else {}

    def set(self, keys, value):
        d = self.d
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.d, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        os.replace(tmp, self.path)


def lst(path):
    return [l.strip() for l in open(path) if l.strip()] if os.path.exists(path) else []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", required=True); p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--model", default="yolo11s.pt")
    p.add_argument("--epochs", type=int, default=30); p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch", type=int, default=16); p.add_argument("--workers", type=int, default=3)
    p.add_argument("--imgsz_ir", type=int, default=640); p.add_argument("--imgsz_rgb", type=int, default=960)
    p.add_argument("--devices", default="0,1"); p.add_argument("--robustness", action="store_true")
    p.add_argument("--pix_rgb", action="store_true", help="also run pixel-level fusion on the RGB grid (IR up-sampled)")
    a = p.parse_args()
    T0 = time.time(); seed = a.seed
    os.makedirs(a.out, exist_ok=True)
    R = Results(os.path.join(a.out, f"results_antiuav_s{seed}.json"))
    R.set(["config"], vars(a))
    errors = []

    def guard(name, fn):
        try:
            t = time.time(); fn(); print(f"[ok] {name} {time.time() - t:.0f}s", flush=True)
        except Exception:
            tb = traceback.format_exc(); print(tb, flush=True); errors.append({"block": name, "traceback": tb})
            R.set(["errors"], errors)

    prep = os.path.join(a.work, "prep"); crops = os.path.join(a.work, "crops"); pf = os.path.join(a.work, "pixfuse")
    gt, meta = P.load_prepared(prep)
    scene = meta["scene"]; rgb_scale = meta["rgb_scale"]
    split_keys = {sp: sorted(k for k, v in meta["split_of"].items() if v == sp) for sp in ("train", "val", "test")}
    # IR->RGB maps from train/val ground truth, refitted with the scale-bounded estimator (the prep stage used an
    # unbounded fit that was degenerate for a few gimbal-tracked sequences); pixel-fusion train/val images are rebuilt
    gm = P.gt_mappings(gt, meta, ("train", "val"))
    gmap = P.global_mapping(gm)
    vmaps = dict(gm)
    for s_, ks in meta["frames_of"].items():
        if meta["split_of"][ks[0]] in ("train", "val"):
            vmaps.setdefault(s_, gmap)
    mtv = {"global": gmap.to_dict(), "per_seq": {k: m.to_dict() for k, m in vmaps.items()}, "fitted": sorted(gm)}
    with open(os.path.join(a.out, "maps_trainval_refit.json"), "w") as f:
        json.dump(mtv, f)
    if not os.path.exists(os.path.join(pf, "REFIT")):
        t = time.time(); shutil.rmtree(pf, ignore_errors=True)
        P.build_pixel_fusion(prep, pf, gt, meta, vmaps, ("train", "val"))
        open(os.path.join(pf, "REFIT"), "w").close()
        print(f"[prep] pixel fusion (IR grid) rebuilt with refitted maps {time.time() - t:.0f}s", flush=True)
    oracle = P.gt_mappings(gt, meta, ("test",))
    R.set(["mapping", "global"], gmap.to_dict())
    fitted = [vmaps[s] for s in mtv["fitted"]]
    R.set(["mapping", "train_val_scale_stats"], {
        "ax_median": float(np.median([m.ax for m in fitted])), "ay_median": float(np.median([m.ay for m in fitted])),
        "bx_iqr": [float(np.percentile([m.bx for m in fitted], q)) for q in (25, 75)],
        "by_iqr": [float(np.percentile([m.by for m in fitted], q)) for q in (25, 75)], "n": len(fitted)})

    # ------------------------------------------------------------------ training (parallel queues)
    runs = os.path.abspath(os.path.join(a.work, "runs"))
    pfr = os.path.join(a.work, "pixfuse_rgb")
    if a.pix_rgb and not os.path.exists(os.path.join(pfr, "data.yaml")):
        t = time.time()
        P.build_pixel_fusion_rgbgrid(prep, pfr, gt, meta, vmaps, ("train", "val"))
        print(f"[prep] pixel fusion on RGB grid {time.time() - t:.0f}s", flush=True)
    # one job list, ordered by expected cost; every GPU worker pulls the next unclaimed job
    jobs = [{"name": f"rgb_s{seed}", "data": os.path.join(prep, "RGBs", "data.yaml"), "imgsz": a.imgsz_rgb}]
    if a.pix_rgb:
        jobs.append({"name": f"pixrgb_s{seed}", "data": os.path.join(pfr, "data.yaml"), "imgsz": a.imgsz_rgb})
    jobs += [{"name": f"ir_s{seed}", "data": os.path.join(prep, "IR", "data.yaml"), "imgsz": a.imgsz_ir},
             {"name": f"pix_s{seed}", "data": os.path.join(pf, "data.yaml"), "imgsz": a.imgsz_ir},
             {"name": f"crop_s{seed}", "data": os.path.join(crops, "data.yaml"), "imgsz": 320}]
    names = [j["name"].rsplit("_s", 1)[0] for j in jobs]
    devs = a.devices.split(",")
    common = [sys.executable, os.path.join(HERE, "train_queue.py"), "--project", runs, "--seed", str(seed),
              "--model", a.model, "--epochs", str(a.epochs), "--patience", str(a.patience), "--batch", str(a.batch),
              "--workers", str(a.workers), "--jobs", json.dumps(jobs), "--claim_dir", os.path.join(runs, f"claims_s{seed}")]
    t_train = time.time()
    procs = [subprocess.Popen(common + ["--device", d]) for d in devs]
    codes = [pr.wait() for pr in procs]
    if any(codes):
        raise SystemExit(f"training failed: {codes}")
    from ultralytics import YOLO

    def load(name):
        w = os.path.join(runs, name, "weights")
        return YOLO(os.path.join(w, "best.pt") if os.path.exists(os.path.join(w, "best.pt")) else os.path.join(w, "last.pt"))

    m_ir, m_rgb, m_crop, m_pf = (load(f"{n}_s{seed}") for n in ("ir", "rgb", "crop", "pix"))
    m_pfr = load(f"pixrgb_s{seed}") if a.pix_rgb else None
    tsec = {n: float(open(os.path.join(runs, f"{n}_s{seed}", "DONE")).read() or 0) for n in names}
    R.set(["train_seconds"], tsec); R.set(["train_wall_minutes"], (time.time() - t_train) / 60)
    R.set(["training_curves"], {n: open(os.path.join(runs, f"{n}_s{seed}", "results.csv")).read()
                                for n in names if os.path.exists(os.path.join(runs, f"{n}_s{seed}", "results.csv"))})
    dev = 0 if devs[0] != "cpu" else "cpu"

    # ------------------------------------------------------------------ inference (raw detections are saved)
    D = {}
    det_path = os.path.join(a.out, f"detections_s{seed}.pkl")

    def save_dets():
        with open(det_path + ".tmp", "wb") as f:
            pickle.dump(D, f, protocol=4)
        os.replace(det_path + ".tmp", det_path)

    ir_p = {sp: lst(os.path.join(prep, "IR", f"{sp}.txt")) for sp in ("val", "test")}
    rgb_p = {sp: lst(os.path.join(prep, "RGBs", f"{sp}.txt")) for sp in ("val", "test")}
    for sp in ("val", "test"):
        D[f"ir_{sp}"] = P.infer(m_ir, ir_p[sp], a.imgsz_ir, dev)
        D[f"rgb_{sp}"] = P.infer(m_rgb, rgb_p[sp], a.imgsz_rgb, dev, scale=rgb_scale)
    D["pix_val"] = P.infer(m_pf, lst(os.path.join(pf, "val.txt")), a.imgsz_ir, dev)
    if m_pfr is not None:
        D["pixrgb_val"] = P.infer(m_pfr, lst(os.path.join(pfr, "val.txt")), a.imgsz_rgb, dev, scale=rgb_scale)
    save_dets()

    cal_ir, prior_ir = P.calibrate(D["ir_val"], gt["IR"], split_keys["val"])
    cal_rgb, prior_rgb = P.calibrate(D["rgb_val"], gt["RGB"], split_keys["val"])
    cal_pf, _ = P.calibrate(D["pix_val"], gt["IR"], split_keys["val"])
    cal_pfr = P.calibrate(D["pixrgb_val"], gt["RGB"], split_keys["val"])[0] if m_pfr is not None else None
    prior = float(np.clip((prior_ir + prior_rgb) / 2, 0.05, 0.95))
    R.set(["calibration"], {"ir": cal_ir.to_dict(), "rgb": cal_rgb.to_dict(), "pix": cal_pf.to_dict(), "prior": prior,
                            "pixrgb": cal_pfr.to_dict() if cal_pfr is not None else None,
                            "prior_ir": prior_ir, "prior_rgb": prior_rgb})
    C = {k: P.apply_cal(D[k], cal_ir if k.startswith("ir") else cal_rgb) for k in ("ir_val", "ir_test", "rgb_val", "rgb_test")}

    # test mappings are self-calibrated from detections (no ground truth)
    sc_all = [v for m in fitted for v in (m.ax, m.ay)]
    scale_range = (0.85 * float(np.percentile(sc_all, 2)), 1.15 * float(np.percentile(sc_all, 98)))
    tmaps, tinfo = P.test_mappings(D["ir_test"], D["rgb_test"], meta, gmap, scale_range=scale_range)
    err_self = P.mapping_error(tmaps, gt, meta); err_global = P.mapping_error({s: gmap for s in tmaps}, gt, meta)
    err_oracle = P.mapping_error({s: m for s, m in oracle.items() if s in tmaps}, gt, meta)
    R.set(["mapping_test"], {"scale_range": scale_range, "self_calibrated_share": float(np.mean([v["self_calibrated"] for v in tinfo.values()])),
                             "per_seq": tinfo, "err_px_self": err_self, "err_px_global": err_global, "err_px_oracle": err_oracle,
                             "median_err_px_self": float(np.median(list(err_self.values()))) if err_self else None,
                             "median_err_px_global": float(np.median(list(err_global.values()))) if err_global else None,
                             "median_err_px_oracle": float(np.median(list(err_oracle.values()))) if err_oracle else None})
    D["tmaps"] = {s: m.to_dict() for s, m in tmaps.items()}

    # pixel-level fusion test images use the self-calibrated maps
    pf_test = os.path.join(a.work, f"pixfuse_test_s{seed}")
    P.build_pixel_fusion(prep, pf_test, gt, meta, tmaps, ("test",))
    D["pix_test"] = P.infer(m_pf, lst(os.path.join(pf_test, "test.txt")), a.imgsz_ir, dev)
    d_pf = P.apply_cal(D["pix_test"], cal_pf)
    pfr_test = os.path.join(a.work, f"pixfuse_rgb_test_s{seed}")
    d_pfr = None
    if m_pfr is not None:
        P.build_pixel_fusion_rgbgrid(prep, pfr_test, gt, meta, tmaps, ("test",))
        D["pixrgb_test"] = P.infer(m_pfr, lst(os.path.join(pfr_test, "test.txt")), a.imgsz_rgb, dev, scale=rgb_scale)
        d_pfr = P.apply_cal(D["pixrgb_test"], cal_pfr)  # full-resolution RGB coordinates

    # cascade: IR proposals verified on native-resolution RGB crops
    pv = P.cascade_parts(m_crop, prep, split_keys["val"], C["ir_val"], vmaps, device=dev)
    crop_cal, q_sil = P.cascade_calibration(pv, gt["RGB"])
    pt = P.cascade_parts(m_crop, prep, split_keys["test"], C["ir_test"], tmaps, device=dev)
    D["cascade_parts_val"] = pv; D["cascade_parts_test"] = pt
    save_dets()

    # ------------------------------------------------------------------ evaluation
    keys_te = split_keys["test"]
    seq_frames = {}
    for k in keys_te:
        seq_frames.setdefault(P.seq_of(k), []).append(k)

    def refs(ir_d, rgb_d, maps):
        return {"IR": (ir_d, P.to_frame(rgb_d, maps, "rgb2ir")), "RGB": (P.to_frame(ir_d, maps, "ir2rgb"), rgb_d)}

    test_ref = refs(C["ir_test"], C["rgb_test"], tmaps)
    val_ref = refs(C["ir_val"], C["rgb_val"], vmaps)
    oracle_ref = refs(C["ir_test"], C["rgb_test"], {**tmaps, **oracle})
    global_ref = refs(C["ir_test"], C["rgb_test"], {s: gmap for s in tmaps})
    lrms = {}

    def late_block(ref):
        g = gt[ref]
        R.set(["single", ref, "IR-only"], P.eval_block(test_ref[ref][0], g, keys_te, scene))
        R.set(["single", ref, "RGB-only"], P.eval_block(test_ref[ref][1], g, keys_te, scene))
        lrm = P.fit_learned(val_ref[ref][0], val_ref[ref][1], g, split_keys["val"], scene)
        lrms[ref] = lrm
        if lrm is not None:
            R.set(["learned_fusion_weights", ref], lrm.w.tolist())
        for rule in RULES:
            if rule == "learned" and lrm is None:
                continue
            f = P.fuse_all(test_ref[ref][0], test_ref[ref][1], keys_te, rule, prior, scene, lrm)
            R.set(["late", ref, rule], P.eval_block(f, g, keys_te, scene))
            if rule in ("proben", "learned", "max"):
                ft = {}
                for s_, ks in seq_frames.items():
                    ft.update(P.temporal_smooth(ks, f))
                R.set(["late_temporal", ref, rule], P.eval_block(ft, g, keys_te, scene))
                fo = P.fuse_all(oracle_ref[ref][0], oracle_ref[ref][1], keys_te, rule, prior, scene, lrm)
                R.set(["late_oracle_mapping", ref, rule], P.eval_block(fo, g, keys_te, scene))
                fg = P.fuse_all(global_ref[ref][0], global_ref[ref][1], keys_te, rule, prior, scene, lrm)
                R.set(["late_global_mapping", ref, rule], P.eval_block(fg, g, keys_te, scene))

    for ref in ("IR", "RGB"):
        guard(f"late_{ref}", lambda ref=ref: late_block(ref))

    def cascade_block():
        casc = P.cascade_combine(pt, crop_cal, q_sil, prior)
        casc_rgb = {k: P.fuse_frame(casc.get(k, np.zeros((0, 5))), C["rgb_test"].get(k, np.zeros((0, 5))), "proben", prior)
                    for k in keys_te}
        R.set(["cascade", "calibration"], {"crop": crop_cal.to_dict(), "q_silent": q_sil})
        R.set(["cascade", "RGB", "cascade"], P.eval_block(casc, gt["RGB"], keys_te, scene))
        R.set(["cascade", "RGB", "cascade+rgb_fullframe"], P.eval_block(casc_rgb, gt["RGB"], keys_te, scene))
        R.set(["cascade", "IR", "cascade"], P.eval_block(P.to_frame(casc, tmaps, "rgb2ir"), gt["IR"], keys_te, scene))
        R.set(["cascade", "IR", "cascade+rgb_fullframe"], P.eval_block(P.to_frame(casc_rgb, tmaps, "rgb2ir"), gt["IR"], keys_te, scene))

    def pixel_block():
        R.set(["pixel", "IR", "pixel_fusion"], P.eval_block(d_pf, gt["IR"], keys_te, scene))
        R.set(["pixel", "RGB", "pixel_fusion"], P.eval_block(P.to_frame(d_pf, tmaps, "ir2rgb"), gt["RGB"], keys_te, scene))
        hyb = {k: fuse_many([test_ref["IR"][0].get(k, np.zeros((0, 5))), test_ref["IR"][1].get(k, np.zeros((0, 5))),
                             d_pf.get(k, np.zeros((0, 5)))], prior) for k in keys_te}
        R.set(["hybrid", "IR", "ir+rgb+pixel_proben"], P.eval_block(hyb, gt["IR"], keys_te, scene))
        R.set(["hybrid", "RGB", "ir+rgb+pixel_proben"], P.eval_block(P.to_frame(hyb, tmaps, "ir2rgb"), gt["RGB"], keys_te, scene))

    def pixel_rgb_block():
        R.set(["pixel", "RGB", "pixel_fusion_rgbgrid"], P.eval_block(d_pfr, gt["RGB"], keys_te, scene))
        R.set(["pixel", "IR", "pixel_fusion_rgbgrid"], P.eval_block(P.to_frame(d_pfr, tmaps, "rgb2ir"), gt["IR"], keys_te, scene))

    guard("cascade", cascade_block)
    guard("pixel_hybrid", pixel_block)
    if d_pfr is not None:
        guard("pixel_rgbgrid", pixel_rgb_block)

    # ------------------------------------------------------------------ latency (batch 1, after warm-up)
    def latency():
        def lat(fn, n=50):
            fn(); fn(); t = time.time()
            for _ in range(n):
                fn()
            return 1000 * (time.time() - t) / n
        R.set(["latency_ms"], {
            "ir_detector": lat(lambda: m_ir.predict(ir_p["test"][0], imgsz=a.imgsz_ir, device=dev, verbose=False)),
            "rgb_detector": lat(lambda: m_rgb.predict(rgb_p["test"][0], imgsz=a.imgsz_rgb, device=dev, verbose=False)),
            "crop_verifier_per_candidate": lat(lambda: m_crop.predict(np.zeros((320, 320, 3), np.uint8), imgsz=320, device=dev, verbose=False)),
            "pixel_fusion_detector": lat(lambda: m_pf.predict(lst(os.path.join(pf_test, "test.txt"))[0], imgsz=a.imgsz_ir, device=dev, verbose=False)),
            "pixel_fusion_rgbgrid_detector": lat(lambda: m_pfr.predict(lst(os.path.join(pfr_test, "test.txt"))[0], imgsz=a.imgsz_rgb, device=dev, verbose=False)) if m_pfr is not None else None,
            "late_fusion_proben_per_frame_cpu": lat(lambda: P.fuse_frame(test_ref["IR"][0][keys_te[0]], test_ref["IR"][1][keys_te[0]], "proben", prior), 200),
            "device": str(devs[0])})
    guard("latency", latency)

    # ------------------------------------------------------------------ robustness (degraded inputs, test time only)
    if a.robustness:
        D["degraded"] = {}

        def robust(cond):
            deg = P.make_degrade(cond)
            if cond == "rgb_missing":
                di_raw, dr_raw = D["ir_test"], {k: np.zeros((0, 5)) for k in keys_te}
            elif cond == "ir_missing":
                di_raw, dr_raw = {k: np.zeros((0, 5)) for k in keys_te}, D["rgb_test"]
            else:
                di_raw = P.infer(m_ir, ir_p["test"], a.imgsz_ir, dev, degrade=deg, mod="IR") if cond.startswith("ir") else D["ir_test"]
                dr_raw = P.infer(m_rgb, rgb_p["test"], a.imgsz_rgb, dev, degrade=deg, mod="RGB", scale=rgb_scale) if cond.startswith("rgb") else D["rgb_test"]
            di, dr = P.apply_cal(di_raw, cal_ir), P.apply_cal(dr_raw, cal_rgb)
            dr_ir = P.to_frame(dr, tmaps, "rgb2ir")
            if cond in ("rgb_missing", "ir_missing"):
                sc = {k: dict(scene[k], **({"rgb_brightness": 0.0} if cond == "rgb_missing" else {"ir_contrast": 0.0})) for k in keys_te}
            else:
                sc = P.scene_cues(prep, keys_te, deg)
            # pixel-level fusion on degraded inputs (a missing sensor becomes a zero channel)
            pdir = os.path.join(a.work, f"pix_deg_{cond}")
            P.build_pixel_fusion(prep, pdir, gt, meta, tmaps, ("test",), degrade=deg)
            dp_raw = P.infer(m_pf, lst(os.path.join(pdir, "test.txt")), a.imgsz_ir, dev)
            shutil.rmtree(pdir, ignore_errors=True)
            dp = P.apply_cal(dp_raw, cal_pf)
            dpr_raw = None
            if m_pfr is not None:
                pdir2 = os.path.join(a.work, f"pixrgb_deg_{cond}")
                P.build_pixel_fusion_rgbgrid(prep, pdir2, gt, meta, tmaps, ("test",), degrade=deg)
                dpr_raw = P.infer(m_pfr, lst(os.path.join(pdir2, "test.txt")), a.imgsz_rgb, dev, scale=rgb_scale)
                shutil.rmtree(pdir2, ignore_errors=True)
            # cascade on degraded inputs
            if cond == "ir_missing":
                parts = {k: [] for k in keys_te}
            elif cond == "rgb_missing":
                parts = {k: [(r[0], r[1], None, None) for r in v] for k, v in pt.items()}
            elif cond.startswith("rgb"):
                parts = P.cascade_parts(m_crop, prep, keys_te, C["ir_test"], tmaps, device=dev, degrade=deg)
            else:
                parts = P.cascade_parts(m_crop, prep, keys_te, di, tmaps, device=dev)
            casc_ir = P.to_frame(P.cascade_combine(parts, crop_cal, q_sil, prior), tmaps, "rgb2ir")
            D["degraded"][cond] = {"ir": di_raw if cond.startswith("ir") and cond != "ir_missing" else None,
                                   "rgb": dr_raw if cond.startswith("rgb") and cond != "rgb_missing" else None,
                                   "pix": dp_raw, "pixrgb": dpr_raw, "cascade_parts": parts}
            save_dets()
            g = gt["IR"]
            out = {"IR-only": P.eval_block(di, g, keys_te, scene), "RGB-only": P.eval_block(dr_ir, g, keys_te, scene),
                   "pixel_fusion": P.eval_block(dp, g, keys_te, scene), "cascade": P.eval_block(casc_ir, g, keys_te, scene)}
            if dpr_raw is not None:
                out["pixel_fusion_rgbgrid"] = P.eval_block(P.to_frame(P.apply_cal(dpr_raw, cal_pfr), tmaps, "rgb2ir"), g, keys_te, scene)
            for rule in ("or", "and", "max", "wbf", "proben", "learned"):
                if rule == "learned" and lrms.get("IR") is None:
                    continue
                out[rule] = P.eval_block(P.fuse_all(di, dr_ir, keys_te, rule, prior, sc, lrms.get("IR")), g, keys_te, scene)
            R.set(["robustness_IRref", cond], out)

        for cond in CONDS:
            guard(f"robust_{cond}", lambda cond=cond: robust(cond))

    shutil.rmtree(pf_test, ignore_errors=True); shutil.rmtree(pfr_test, ignore_errors=True)
    R.set(["elapsed_hours"], (time.time() - T0) / 3600)
    print("done", R.path, "errors:", len(errors), flush=True)


if __name__ == "__main__":
    main()
