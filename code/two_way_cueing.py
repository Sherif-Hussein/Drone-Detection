#!/usr/bin/env python3
"""
Two-way cueing with re-detection (thermal reference frame): the cameras tell each other where to look, and each
camera's own detector places the box.

For each frame, RGB detections (main-study YOLO11s RGB, seed 0) are mapped into the thermal image with the
self-calibrated per-sequence map. A thermal detection matched to an RGB cue (size-normalised centre distance) has its
score raised by noisy-OR, s' = 1 - (1 - p_t)(1 - gamma p_r). An RGB cue with no thermal match and p_r >= theta triggers a
re-detection: the thermal detector is run again on a thermal crop centred on the cue (window 160 px upscaled 2x, or
320 px native), and its best box, kept in thermal coordinates, enters with s = 1 - (1 - p_c)(1 - gamma p_r).

Arms, all calibrated (Platt) and tuned by leave-one-session-out cross-validation over the 10 test sessions:
  thermal alone | boost only | boost + mapped RGB box (anchored late fusion with unmatched RGB detections, as in the
  paper) | boost + re-detection (two-way cueing) | swap control (cues from another sequence's RGB detections).
Differences: session-cluster bootstrap.

python two_way_cueing.py PREP RUNS IR_TRIAL_DIR IR_IMAGES_DIR OUT.json [--models ir_s640,ir_n640] [--weights DIR]
"""
import argparse, itertools, json, os, pickle, sys, time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import antiuav_pipeline as P
from fusion_core import AxisAffine, match_by_centre, iou_matrix
from analyze_antiuav import frame_records, ap_from, seq_pack, session_of_seq, cluster_bootstrap

E = np.zeros((0, 5))
WINDOWS = {2: 160, 1: 320}  # zoom -> thermal crop side in pixels (both give a 320-px network input)


def crop_detections(model, img_dir, cues_ir, keys, zoom, max_cues=5):
    """cues_ir: frame -> (N,5) RGB cues in thermal coordinates. Returns frame -> list (per cue) of (M,5) thermal boxes."""
    out, win = {}, WINDOWS[zoom]
    for k in keys:
        c = cues_ir.get(k, E)
        c = c[np.argsort(-c[:, 4])][:max_cues] if len(c) else c
        out[k] = []
        if not len(c):
            continue
        im = cv2.imread(os.path.join(img_dir, k + ".jpg"))
        H, W = im.shape[:2]
        crops, offs = [], []
        for b in c:
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            x0 = int(np.clip(round(cx - win / 2), 0, max(W - win, 0))); y0 = int(np.clip(round(cy - win / 2), 0, max(H - win, 0)))
            cr = im[y0:y0 + win, x0:x0 + win]
            if zoom != 1:
                cr = cv2.resize(cr, (cr.shape[1] * zoom, cr.shape[0] * zoom), interpolation=cv2.INTER_CUBIC)
            crops.append(cr); offs.append((x0, y0))
        res = model.predict(crops, imgsz=320, conf=0.01, device="cpu", verbose=False, max_det=3)
        for r, (x0, y0) in zip(res, offs):
            bb = r.boxes
            d = np.c_[bb.xyxy.cpu().numpy() / zoom, bb.conf.cpu().numpy()] if len(bb) else E.copy()
            if len(d):
                d[:, [0, 2]] += x0; d[:, [1, 3]] += y0
            out[k].append(d)
    return out, c_order(cues_ir, keys, max_cues)


def c_order(cues_ir, keys, max_cues):
    return {k: (cues_ir.get(k, E)[np.argsort(-cues_ir.get(k, E)[:, 4])][:max_cues] if len(cues_ir.get(k, E)) else E) for k in keys}


def fuse_frame(t, c, crops, gamma, theta, mode, tau=1.0):
    """t: calibrated thermal dets; c: calibrated cues (thermal coords, sorted as crops); crops: calibrated crop dets per cue."""
    t = t.copy()
    pairs, _, uc = match_by_centre(t[:, :4], c[:, :4], tau) if len(t) and len(c) else ([], [], list(range(len(c))))
    for i, j in pairs:
        t[i, 4] = 1 - (1 - t[i, 4]) * (1 - gamma * c[j, 4])
    add = []
    if mode in ("insert", "redetect"):
        for j in uc:
            if c[j, 4] < theta:
                continue
            if mode == "insert":
                add.append(np.r_[c[j, :4], gamma * c[j, 4]])
            else:
                d = crops[j] if j < len(crops) else E
                if not len(d):
                    continue
                if len(t):
                    d = d[iou_matrix(d[:, :4], t[:, :4]).max(1) < 0.5] if len(d) else d
                if not len(d):
                    continue
                b = d[np.argmax(d[:, 4])]
                add.append(np.r_[b[:4], 1 - (1 - b[4]) * (1 - gamma * c[j, 4])])
    return np.vstack([t] + [a[None] for a in add]) if add else t


def main():
    ap_ = argparse.ArgumentParser()
    for n in ("prep", "runs", "trial", "imgs", "out"):
        ap_.add_argument(n)
    ap_.add_argument("--models", default="ir_s640,ir_n640,ir_nP2_640")
    ap_.add_argument("--weights", default=None, help="default: TRIAL/weights")
    ap_.add_argument("--nboot", type=int, default=2000)
    a = ap_.parse_args()
    from ultralytics import YOLO
    gt, meta = P.load_prepared(a.prep)
    avail = {os.path.splitext(f)[0] for f in os.listdir(a.imgs)}
    keys = sorted(k for k, v in meta["split_of"].items() if v == "test" and k in avail)
    fo = {}
    for k in keys:
        fo.setdefault(P.seq_of(k), []).append(k)
    seqs = sorted(fo)
    sess = sorted({session_of_seq(s) for s in seqs})
    keys_of = {h: [k for k in keys if session_of_seq(P.seq_of(k)) == h] for h in sess}
    D0 = pickle.load(open(os.path.join(a.runs, "detections_s0.pkl"), "rb"))
    tmaps = {q: AxisAffine.from_dict(d) for q, d in D0["tmaps"].items()}
    rgb_raw = D0["rgb_test"]
    cues_raw = P.to_frame(rgb_raw, tmaps, "rgb2ir")
    # swap control: each sequence receives the RGB detections of the next sequence (frame by frame, cyclic)
    nxt = {s: seqs[(i + 1) % len(seqs)] for i, s in enumerate(seqs)}
    swap_raw, swap_rgb = {}, {}
    for s in seqs:
        ka, kb = fo[s], fo[nxt[s]]
        for i, k in enumerate(ka):
            src = kb[i % len(kb)]
            d = rgb_raw.get(src, E)
            swap_rgb[k] = d  # mapped with the receiving sequence's own map, exactly like the real cues
            swap_raw[k] = tmaps[s].inverse().boxes(d) if (s in tmaps and len(d)) else d
    trial = pickle.load(open(os.path.join(a.trial, "ir_small_trial_test_dets.pkl"), "rb"))
    wdir = a.weights or os.path.join(a.trial, "weights")
    GRID = list(itertools.product((0.25, 0.5, 1.0), (0.3, 0.5), (2, 1)))  # gamma, theta, zoom
    res = {"n_frames": len(keys), "sessions": sess, "grid": GRID, "models": {}}

    def ap(d, ks):
        r = frame_records(d, gt["IR"], ks)
        return ap_from(np.concatenate([r[k][0] for k in ks]), np.concatenate([r[k][1] for k in ks]), sum(r[k][2] for k in ks))

    for name in a.models.split(","):
        t0 = time.time()
        model = YOLO(os.path.join(wdir, name + ".pt"))
        ir_raw = trial[name]
        crops = {}
        for cname, craw in (("real", cues_raw), ("swap", swap_raw)):
            for z in (2, 1):
                crops[cname, z], order = crop_detections(model, a.imgs, craw, keys, z)
        crop_s = (time.time() - t0) / max(sum(len(v) for c in crops.values() for v in c.values()), 1) * 1000
        orders = {"real": c_order(cues_raw, keys, 5), "swap": c_order(swap_raw, keys, 5)}
        rgbsrc = {"real": rgb_raw, "swap": swap_rgb}
        arms = {n: {} for n in ("thermal alone", "boost only", "boost + mapped RGB box", "two-way cueing (re-detect)",
                                "swap control (re-detect)")}
        chosen = {n: {} for n in arms}
        for h in sess:
            tr = [k for k in keys if k not in set(keys_of[h])]
            cir, _ = P.calibrate(ir_raw, gt["IR"], tr)
            C = {}
            for src in ("real", "swap"):
                crgb, _ = P.calibrate(rgb_raw, gt["RGB"], tr)  # RGB scores calibrated in the RGB camera's own frame
                cue_cal = {k: (np.c_[orders[src][k][:, :4], crgb(orders[src][k][:, 4])] if len(orders[src][k]) else E) for k in keys}
                for z in (2, 1):
                    flat = {k: (np.vstack(crops[src, z][k]) if crops[src, z][k] else E) for k in tr}
                    ccal, _ = P.calibrate(flat, gt["IR"], tr)
                    C[src, z] = {k: [np.c_[d[:, :4], ccal(d[:, 4])] if len(d) else d for d in crops[src, z][k]] for k in keys}
                C[src] = cue_cal
            tcal = {k: (np.c_[ir_raw.get(k, E)[:, :4], cir(ir_raw.get(k, E)[:, 4])] if len(ir_raw.get(k, E)) else E) for k in keys}

            def run(src, mode, g, th, z, ks):
                return {k: fuse_frame(tcal[k], C[src][k], C[src, z][k], g, th, mode) for k in ks}

            for arm, (src, mode) in {"boost only": ("real", "boost"), "boost + mapped RGB box": ("real", "insert"),
                                     "two-way cueing (re-detect)": ("real", "redetect"),
                                     "swap control (re-detect)": ("swap", "redetect")}.items():
                grid = [g for g in GRID if g[2] == 2] if mode != "redetect" else GRID  # zoom only matters for re-detection
                best = max(grid, key=lambda g: ap(run(src, mode, *g, tr), tr))
                chosen[arm][h] = list(best)
                arms[arm].update(run(src, mode, *best, keys_of[h]))
            arms["thermal alone"].update({k: tcal[k] for k in keys_of[h]})
        pk = {n: seq_pack(frame_records(d, gt["IR"], keys), fo) for n, d in arms.items()}
        base = pk["thermal alone"]
        m = {"AP50": {n: float(ap(d, keys)) for n, d in arms.items()},
             "crop_ms_per_cue_cpu": crop_s, "chosen": chosen,
             "vs_thermal": {n: cluster_bootstrap([pk[n]], [base], seqs, a.nboot) for n in arms if n != "thermal alone"},
             "redetect_vs_insert": cluster_bootstrap([pk["two-way cueing (re-detect)"]], [pk["boost + mapped RGB box"]], seqs, a.nboot)}
        res["models"][name] = m
        print(name, {n: round(v, 4) for n, v in m["AP50"].items()},
              {n: (round(v["diff"], 4), [round(x, 4) for x in v["ci95"]], round(v["p_two_sided"], 3)) for n, v in m["vs_thermal"].items()},
              flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
