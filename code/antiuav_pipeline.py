"""
Anti-UAV300 (paired RGB + IR) pipeline for the SkyFuse fusion-level study.

Stages (called from the Kaggle notebook):
  discover()        find sequences, modality files and labels
  prepare()         sample frames, write YOLO datasets (IR, RGB), GT dicts, scene cues, data stats
  gt_mappings()     IR->RGB axis-affine per train/val sequence from paired GT boxes
  build_pixel_fusion()  3-channel IR-frame images (IR, warped RGB gray, mean) for early fusion
  build_crops()     native-resolution RGB crops around GT (+ negatives) for the cascade verifier
  infer()           run a YOLO model over frames (optionally with a degradation)
  test_mappings()   self-calibrated IR->RGB mapping per test sequence from detections only
  evaluate_all()    single, late (6 rules), cascade, pixel, hybrid; size/brightness bins; bootstrap
"""
import glob, json, math, os, random, re, time
from collections import defaultdict

import cv2
import numpy as np

from fusion_core import (AxisAffine, LearnedFusion, Platt, ap_suite, average_precision, bootstrap_diff,
                         centers, fit_axis_affine, fuse_frame, fuse_many, iou_matrix, label_detections,
                         match_by_centre, proben_pair, recall_at_fppi, temporal_smooth, wh)

IR_TOK = re.compile(r"(^|[^a-z])(ir|infrared|thermal|tir)([^a-z]|$)", re.I)
RGB_TOK = re.compile(r"(^|[^a-z])(rgb|visible|vis|color)([^a-z]|$)", re.I)
SIZE_BINS = {"<16px": (0, 16), "16-32px": (16, 32), ">=32px": (32, 1e9)}
LIGHT_BINS = {"night(<50)": (0, 50), "dim(50-100)": (50, 100), "day(>=100)": (100, 256)}


def key(seq, idx):
    return f"{seq}__{idx:05d}"


def seq_of(k):
    return k.rsplit("__", 1)[0]


# ----------------------------------------------------------------------------- discovery
def _modality(name):
    n = os.path.basename(name).lower()
    if IR_TOK.search(n.replace("_", " ").replace(".", " ")):
        return "IR"
    if RGB_TOK.search(n.replace("_", " ").replace(".", " ")):
        return "RGB"
    return None


def discover(root):
    """Return list of dicts: seq, split, files{IR:{video|frames, json}, RGB:{...}}."""
    seqs = []
    for dirpath, dirnames, filenames in os.walk(root):
        jsons = [f for f in filenames if f.lower().endswith(".json")]
        vids = [f for f in filenames if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
        if not jsons:
            continue
        entry = {"IR": {}, "RGB": {}}
        for f in jsons:
            m = _modality(f)
            if m:
                entry[m]["json"] = os.path.join(dirpath, f)
        for f in vids:
            m = _modality(f)
            if m:
                entry[m]["video"] = os.path.join(dirpath, f)
        for d in dirnames:  # frame folders
            m = _modality(d)
            if m and glob.glob(os.path.join(dirpath, d, "*.jpg")):
                entry[m]["frames"] = os.path.join(dirpath, d)
        ok = all(("json" in entry[m]) and ("video" in entry[m] or "frames" in entry[m]) for m in ("IR", "RGB"))
        if not ok:
            continue
        parts = [p.lower() for p in os.path.relpath(dirpath, root).split(os.sep)]
        split = next((s for s in ("train", "val", "test") if any(s in p for p in parts[:-1])), None)
        seqs.append({"seq": re.sub(r"[^A-Za-z0-9_-]", "_", os.path.relpath(dirpath, root)), "split": split,
                     "dir": dirpath, "files": entry})
    if seqs and any(s["split"] is None for s in seqs):
        rng = random.Random(0)
        names = sorted(s["seq"] for s in seqs); rng.shuffle(names)
        n = len(names); cut = {nm: ("train" if i < 0.6 * n else "val" if i < 0.75 * n else "test") for i, nm in enumerate(names)}
        for s in seqs:
            s["split"] = s["split"] or cut[s["seq"]]
    return sorted(seqs, key=lambda s: s["seq"])


def read_labels(path):
    with open(path) as f:
        d = json.load(f)
    exist = d.get("exist") or d.get("exists") or []
    rects = d.get("gt_rect") or d.get("gt") or d.get("rect") or []
    out = []
    for i in range(max(len(exist), len(rects))):
        e = exist[i] if i < len(exist) else 1
        r = rects[i] if i < len(rects) else []
        if e and isinstance(r, (list, tuple)) and len(r) == 4 and r[2] > 0 and r[3] > 0:
            x, y, w, h = map(float, r)
            out.append(np.array([x, y, x + w, y + h]))
        else:
            out.append(None)
    return out


class FrameReader:
    def __init__(self, spec):
        self.video = spec.get("video"); self.frames = spec.get("frames")
        self.files = sorted(glob.glob(os.path.join(self.frames, "*.jpg"))) if self.frames else None

    def iterate(self, step, n_max):
        if self.files is not None:
            for i in range(0, min(len(self.files), n_max), step):
                yield i, cv2.imread(self.files[i])
            return
        cap = cv2.VideoCapture(self.video); i = 0
        while i < n_max:
            if i % step == 0:
                ok, im = cap.read()
                if not ok:
                    break
                yield i, im
            elif not cap.grab():  # skip without decoding to an image
                break
            i += 1
        cap.release()


# ----------------------------------------------------------------------------- preparation
def _write_yolo(lbl_path, box, W, H, cls=0):
    with open(lbl_path, "w") as f:
        if box is not None:
            x1, y1, x2, y2 = np.clip(box, [0, 0, 0, 0], [W, H, W, H])
            if x2 - x1 > 1 and y2 - y1 > 1:
                f.write(f"{cls} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")


def _yaml(path, root, tr, va, te, names=("drone",)):
    with open(path, "w") as f:
        f.write(f"path: {root}\ntrain: {tr}\nval: {va}\ntest: {te}\nnames:\n")
        for i, n in enumerate(names):
            f.write(f"  {i}: {n}\n")


def prepare(seqs, out, step=15, rgb_quality=90, max_seqs=None, rgb_det_width=960):
    """Sample frames from each sequence into out/{IR,RGB,RGBs}/images, YOLO labels, GT json, scene cues.
    RGB keeps full resolution (cascade crops); RGBs is a copy resized to `rgb_det_width` that the
    full-frame RGB detector trains on (identical to letterboxing at that size, but cheaper to load)."""
    os.makedirs(out, exist_ok=True)
    gt = {"IR": {}, "RGB": {}}; scene = {}; sizes = {}; split_of = {}; frames_of = defaultdict(list)
    rgb_scale = {}; seq_len = {}
    lists = {m: defaultdict(list) for m in ("IR", "RGB", "RGBs")}
    for si, s in enumerate(seqs[:max_seqs] if max_seqs else seqs):
        lab = {m: read_labels(s["files"][m]["json"]) for m in ("IR", "RGB")}
        n = min(len(lab["IR"]), len(lab["RGB"]))
        seq_len[s["seq"]] = {"IR": len(lab["IR"]), "RGB": len(lab["RGB"])}
        readers = {m: FrameReader(s["files"][m]) for m in ("IR", "RGB")}
        its = {m: readers[m].iterate(step, n) for m in ("IR", "RGB")}
        for (i_ir, im_ir), (i_rgb, im_rgb) in zip(its["IR"], its["RGB"]):
            assert i_ir == i_rgb
            k = key(s["seq"], i_ir)
            H, W = im_rgb.shape[:2]
            sc = W / float(rgb_det_width)
            im_small = cv2.resize(im_rgb, (rgb_det_width, int(round(H / sc))), interpolation=cv2.INTER_AREA)
            rgb_scale[k] = sc
            for m, im in (("IR", im_ir), ("RGB", im_rgb), ("RGBs", im_small)):
                d = os.path.join(out, m); os.makedirs(os.path.join(d, "images"), exist_ok=True)
                os.makedirs(os.path.join(d, "labels"), exist_ok=True)
                ip = os.path.join(d, "images", k + ".jpg")
                if not os.path.exists(ip):
                    cv2.imwrite(ip, im, [cv2.IMWRITE_JPEG_QUALITY, rgb_quality if m != "IR" else 95])
                h, w = im.shape[:2]; sizes[m] = (w, h)
                b = lab["RGB" if m == "RGBs" else m][i_ir]
                if m == "RGBs" and b is not None:
                    b = b / sc
                _write_yolo(os.path.join(d, "labels", k + ".txt"), b, w, h)
                if m != "RGBs":
                    gt[m][k] = None if b is None else b.tolist()
                lists[m][s["split"]].append(ip)
            g_ir = cv2.cvtColor(im_ir, cv2.COLOR_BGR2GRAY) if im_ir.ndim == 3 else im_ir
            scene[k] = {"rgb_brightness": float(cv2.cvtColor(im_rgb, cv2.COLOR_BGR2GRAY).mean()),
                        "ir_contrast": float(g_ir.std())}
            split_of[k] = s["split"]; frames_of[s["seq"]].append(k)
        print(f"[prep] {si + 1}/{len(seqs)} {s['seq']} ({s['split']}) {len(frames_of[s['seq']])} frames", flush=True)
    for m in ("IR", "RGB", "RGBs"):
        d = os.path.join(out, m)
        for sp, v in lists[m].items():
            with open(os.path.join(d, f"{sp}.txt"), "w") as f:
                f.write("\n".join(sorted(v)) + "\n")
        # per-epoch validation during training uses every second validation frame (time budget);
        # calibration and fusion fitting use the full validation split
        with open(os.path.join(d, "val_sub.txt"), "w") as f:
            f.write("\n".join(sorted(lists[m]["val"])[::2]) + "\n")
        _yaml(os.path.join(d, "data.yaml"), os.path.abspath(d), "train.txt", "val_sub.txt", "test.txt")
    meta = {"step": step, "sizes": sizes, "split_of": split_of, "frames_of": dict(frames_of), "scene": scene,
            "rgb_scale": rgb_scale, "seq_len": seq_len}
    with open(os.path.join(out, "gt.json"), "w") as f:
        json.dump(gt, f)
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f)
    return load_prepared(out)


def load_prepared(out):
    with open(os.path.join(out, "gt.json")) as f:
        g = json.load(f)
    with open(os.path.join(out, "meta.json")) as f:
        meta = json.load(f)
    gt = {m: {k: (None if v is None else np.array(v)) for k, v in g[m].items()} for m in g}
    return gt, meta


def data_stats(gt, meta):
    """Distributions for the paper's dataset figure."""
    st = {"sequences": {}, "frames": {}, "exist_rate": {}, "sqrt_area_hist": {}, "brightness_hist": None}
    for sp in ("train", "val", "test"):
        st["sequences"][sp] = sum(1 for s, ks in meta["frames_of"].items() if meta["split_of"][ks[0]] == sp)
        st["frames"][sp] = sum(1 for k, v in meta["split_of"].items() if v == sp)
    edges = [0, 8, 16, 24, 32, 48, 64, 96, 128, 256, 10000]
    for m in ("IR", "RGB"):
        s = [math.sqrt((b[2] - b[0]) * (b[3] - b[1])) for b in gt[m].values() if b is not None]
        st["exist_rate"][m] = len(s) / max(len(gt[m]), 1)
        st["sqrt_area_hist"][m] = {"edges": edges, "counts": np.histogram(s, edges)[0].tolist(),
                                   "median": float(np.median(s)) if s else None}
    br = [v["rgb_brightness"] for v in meta["scene"].values()]
    st["brightness_hist"] = {"edges": list(range(0, 257, 16)), "counts": np.histogram(br, range(0, 257, 16))[0].tolist()}
    st["light_bins"] = {n: int(sum(lo <= b < hi for b in br)) for n, (lo, hi) in LIGHT_BINS.items()}
    st["image_size"] = meta["sizes"]
    return st


# ----------------------------------------------------------------------------- mappings
def gt_mappings(gt, meta, splits=("train", "val")):
    """IR->RGB AxisAffine per sequence from frames where both GT boxes exist."""
    maps = {}
    for seq, ks in meta["frames_of"].items():
        if meta["split_of"][ks[0]] not in splits:
            continue
        a = [gt["IR"][k] for k in ks if gt["IR"][k] is not None and gt["RGB"][k] is not None]
        b = [gt["RGB"][k] for k in ks if gt["IR"][k] is not None and gt["RGB"][k] is not None]
        if len(a) < 3:
            continue
        a, b = np.array(a), np.array(b)
        m, _ = fit_axis_affine(centers(a), centers(b), wh(a), wh(b))
        if m is not None:
            maps[seq] = m
    return maps


def global_mapping(maps):
    if not maps:
        return AxisAffine()
    arr = np.array([[m.ax, m.bx, m.ay, m.by] for m in maps.values()])
    return AxisAffine(*np.median(arr, axis=0))


def test_mappings(dets_ir, dets_rgb, meta, fallback, split="test", min_score=0.5, scale_range=None,
                  min_pairs=8, min_inlier_ratio=0.6, gain=0.5):
    """Self-calibrated mapping per test sequence from confident top-1 detections only (no GT).
    A per-sequence map replaces the global rig map only if (i) its scales lie inside `scale_range`
    (the range seen on training/validation sequences), (ii) at least `min_inlier_ratio` of the detection
    pairs are RANSAC inliers, and (iii) on those inliers its median centre residual is below `gain` times
    that of the global map. All three tests use detections only."""
    maps, info = {}, {}
    for seq, ks in meta["frames_of"].items():
        if meta["split_of"][ks[0]] != split:
            continue
        a, b = [], []
        for k in ks:
            di, dr = dets_ir.get(k, np.zeros((0, 5))), dets_rgb.get(k, np.zeros((0, 5)))
            if len(di) and len(dr) and di[:, 4].max() >= min_score and dr[:, 4].max() >= min_score:
                a.append(di[di[:, 4].argmax(), :4]); b.append(dr[dr[:, 4].argmax(), :4])
        m, use_self, why = None, False, "too_few_pairs"
        res_s = res_g = ratio = None
        if len(a) >= min_pairs:
            a, b = np.array(a), np.array(b)
            m, inl = fit_axis_affine(centers(a), centers(b), wh(a), wh(b))
            why = "fit_failed"
            if m is not None and inl is not None:
                ratio = float(inl.mean())
                res = lambda mm: float(np.median(np.linalg.norm(centers(mm.boxes(a[inl])) - centers(b[inl]), axis=1)))
                res_s, res_g = res(m), res(fallback)
                ok_scale = scale_range is None or (scale_range[0] <= m.ax <= scale_range[1] and scale_range[0] <= m.ay <= scale_range[1])
                if not ok_scale:
                    why = "implausible_scale"
                elif ratio < min_inlier_ratio:
                    why = "few_inliers"
                elif res_s >= gain * res_g:
                    why = "no_clear_gain"
                else:
                    use_self, why = True, "self"
        maps[seq] = m if use_self else fallback
        info[seq] = {"pairs": int(len(a)), "self_calibrated": bool(use_self), "reason": why, "inlier_ratio": ratio,
                     "res_self_px": res_s, "res_global_px": res_g}
    return maps, info


def scene_cues(prep_dir, keys, degrade=None):
    """RGB brightness and IR contrast per frame, recomputed on (optionally degraded) images."""
    out = {}
    for k in keys:
        rgb = cv2.imread(os.path.join(prep_dir, "RGB", "images", k + ".jpg"))
        ir = cv2.imread(os.path.join(prep_dir, "IR", "images", k + ".jpg"))
        if degrade is not None:
            rgb, ir = degrade("RGB", rgb), degrade("IR", ir)
        out[k] = {"rgb_brightness": float(cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).mean()),
                  "ir_contrast": float(cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY).std())}
    return out


def mapping_error(maps, gt, meta, split="test"):
    """Mean centre error (RGB px) of a mapping vs paired GT (analysis only)."""
    errs = {}
    for seq, m in maps.items():
        ks = meta["frames_of"][seq]
        e = [np.linalg.norm(centers(m.boxes(gt["IR"][k][None]))[0] - centers(gt["RGB"][k][None])[0])
             for k in ks if gt["IR"][k] is not None and gt["RGB"][k] is not None]
        if e:
            errs[seq] = float(np.median(e))
    return errs


# ----------------------------------------------------------------------------- derived datasets
def warp_rgb_to_ir(im_rgb, m_ir2rgb, ir_size):
    W, H = ir_size
    M = np.float32([[m_ir2rgb.ax, 0, m_ir2rgb.bx], [0, m_ir2rgb.ay, m_ir2rgb.by]])
    g = cv2.cvtColor(im_rgb, cv2.COLOR_BGR2GRAY) if im_rgb.ndim == 3 else im_rgb
    # warpAffine has no area filter, so first area-downsample RGB close to the IR scale (anti-aliasing),
    # then warp with the correspondingly rescaled map
    f = max(1.0, min(m_ir2rgb.ax, m_ir2rgb.ay))
    if f > 1.2:
        g = cv2.resize(g, (max(1, int(round(g.shape[1] / f))), max(1, int(round(g.shape[0] / f)))), interpolation=cv2.INTER_AREA)
        M = M / np.float32(f)
    return cv2.warpAffine(g, M, (W, H), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR, borderValue=0)


def pixel_fusion_image(im_ir, im_rgb, m_ir2rgb):
    g_ir = cv2.cvtColor(im_ir, cv2.COLOR_BGR2GRAY) if im_ir.ndim == 3 else im_ir
    H, W = g_ir.shape
    g_rgb = warp_rgb_to_ir(im_rgb, m_ir2rgb, (W, H))
    mean = ((g_ir.astype(np.uint16) + g_rgb) // 2).astype(np.uint8)
    return cv2.merge([g_ir, g_rgb, mean])


def build_pixel_fusion(prep_dir, out, gt, meta, maps, splits, degrade=None):
    """Early (pixel-level) fusion in the IR frame. Train/val use GT mappings; test uses the
    self-calibrated mappings passed in `maps`."""
    os.makedirs(os.path.join(out, "images"), exist_ok=True); os.makedirs(os.path.join(out, "labels"), exist_ok=True)
    lists = defaultdict(list)
    for k, sp in meta["split_of"].items():
        if sp not in splits:
            continue
        m = maps.get(seq_of(k))
        if m is None:
            continue
        ir = cv2.imread(os.path.join(prep_dir, "IR", "images", k + ".jpg"))
        rgb = cv2.imread(os.path.join(prep_dir, "RGB", "images", k + ".jpg"))
        if degrade:
            ir, rgb = degrade("IR", ir), degrade("RGB", rgb)
        im = pixel_fusion_image(ir, rgb, m)
        ip = os.path.join(out, "images", k + ".jpg"); cv2.imwrite(ip, im, [cv2.IMWRITE_JPEG_QUALITY, 95])
        H, W = im.shape[:2]
        b = gt["IR"][k]; _write_yolo(os.path.join(out, "labels", k + ".txt"), b, W, H)
        lists[sp].append(ip)
    for sp, v in lists.items():
        with open(os.path.join(out, f"{sp}.txt"), "w") as f:
            f.write("\n".join(sorted(v)) + "\n")
    if "val" in lists:
        with open(os.path.join(out, "val_sub.txt"), "w") as f:
            f.write("\n".join(sorted(lists["val"])[::2]) + "\n")
    _yaml(os.path.join(out, "data.yaml"), os.path.abspath(out), "train.txt", "val_sub.txt" if "val" in lists else "test.txt",
          "test.txt" if "test" in lists else "val.txt")
    return lists


def pixel_fusion_image_rgbgrid(im_ir, im_rgbs, m_ir2rgb, s):
    """Early fusion on the RGB detector grid: the IR image is up-sampled and warped into the (960-wide) RGB frame.
    s = full-resolution RGB width / RGBs width, so IR->RGBs is x' = (a/s) x + b/s."""
    g_ir = cv2.cvtColor(im_ir, cv2.COLOR_BGR2GRAY) if im_ir.ndim == 3 else im_ir
    g_rgb = cv2.cvtColor(im_rgbs, cv2.COLOR_BGR2GRAY) if im_rgbs.ndim == 3 else im_rgbs
    H, W = g_rgb.shape
    M = np.float32([[m_ir2rgb.ax / s, 0, m_ir2rgb.bx / s], [0, m_ir2rgb.ay / s, m_ir2rgb.by / s]])
    ir_w = cv2.warpAffine(g_ir, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    mean = ((ir_w.astype(np.uint16) + g_rgb) // 2).astype(np.uint8)
    return cv2.merge([ir_w, g_rgb, mean])


def build_pixel_fusion_rgbgrid(prep_dir, out, gt, meta, maps, splits, degrade=None):
    """Early fusion in the RGB frame (labels = RGB ground truth). Train/val use GT maps; test the self-calibrated ones."""
    os.makedirs(os.path.join(out, "images"), exist_ok=True); os.makedirs(os.path.join(out, "labels"), exist_ok=True)
    lists = defaultdict(list)
    for k, sp in meta["split_of"].items():
        if sp not in splits:
            continue
        m = maps.get(seq_of(k))
        if m is None:
            continue
        ir = cv2.imread(os.path.join(prep_dir, "IR", "images", k + ".jpg"))
        rgb = cv2.imread(os.path.join(prep_dir, "RGBs", "images", k + ".jpg"))
        if degrade:
            ir, rgb = degrade("IR", ir), degrade("RGB", rgb)
        s = meta["rgb_scale"][k]
        im = pixel_fusion_image_rgbgrid(ir, rgb, m, s)
        ip = os.path.join(out, "images", k + ".jpg"); cv2.imwrite(ip, im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        H, W = im.shape[:2]
        b = gt["RGB"][k]
        _write_yolo(os.path.join(out, "labels", k + ".txt"), None if b is None else np.asarray(b) / s, W, H)
        lists[sp].append(ip)
    for sp, v in lists.items():
        with open(os.path.join(out, f"{sp}.txt"), "w") as f:
            f.write("\n".join(sorted(v)) + "\n")
    if "val" in lists:
        with open(os.path.join(out, "val_sub.txt"), "w") as f:
            f.write("\n".join(sorted(lists["val"])[::2]) + "\n")
    _yaml(os.path.join(out, "data.yaml"), os.path.abspath(out), "train.txt", "val_sub.txt" if "val" in lists else "test.txt",
          "test.txt" if "test" in lists else "val.txt")
    return lists


def build_crops(prep_dir, out, gt, meta, crop=320, jitter=100, neg_ratio=0.5, seed=0):
    """Native-resolution RGB crops for the cascade verifier (train and val splits)."""
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(out, "images"), exist_ok=True); os.makedirs(os.path.join(out, "labels"), exist_ok=True)
    lists = defaultdict(list)
    for k, sp in meta["split_of"].items():
        if sp not in ("train", "val"):
            continue
        im = cv2.imread(os.path.join(prep_dir, "RGB", "images", k + ".jpg"))
        H, W = im.shape[:2]
        b = gt["RGB"][k]
        todo = []
        if b is not None:
            cx, cy = (b[0] + b[2]) / 2 + rng.uniform(-jitter, jitter), (b[1] + b[3]) / 2 + rng.uniform(-jitter, jitter)
            todo.append(("p", cx, cy))
        if rng.random() < neg_ratio:
            todo.append(("n", rng.uniform(0, W), rng.uniform(0, H)))
        for tag, cx, cy in todo:
            x0 = int(np.clip(cx - crop / 2, 0, max(W - crop, 0))); y0 = int(np.clip(cy - crop / 2, 0, max(H - crop, 0)))
            c = im[y0:y0 + crop, x0:x0 + crop]
            box = None
            if b is not None:
                bb = np.array([b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0])
                inter = (min(bb[2], crop) - max(bb[0], 0)) * (min(bb[3], crop) - max(bb[1], 0))
                if inter > 0.5 * (bb[2] - bb[0]) * (bb[3] - bb[1]):
                    box = np.clip(bb, 0, crop)
            if tag == "n" and box is not None:
                continue
            name = f"{k}_{tag}"
            ip = os.path.join(out, "images", name + ".jpg"); cv2.imwrite(ip, c, [cv2.IMWRITE_JPEG_QUALITY, 95])
            _write_yolo(os.path.join(out, "labels", name + ".txt"), box, c.shape[1], c.shape[0])
            lists[sp].append(ip)
    for sp, v in lists.items():
        with open(os.path.join(out, f"{sp}.txt"), "w") as f:
            f.write("\n".join(sorted(v)) + "\n")
    _yaml(os.path.join(out, "data.yaml"), os.path.abspath(out), "train.txt", "val.txt", "val.txt")
    return lists


# ----------------------------------------------------------------------------- degradations
def make_degrade(name, seed=0):
    rng = np.random.default_rng(seed)

    def f(mod, im):
        if im is None:
            return im
        x = im.astype(np.float32)
        if name == "rgb_dark" and mod == "RGB":
            x = x * 0.2 + rng.normal(0, 6, x.shape)
        elif name == "rgb_fog" and mod == "RGB":
            x = 0.4 * x + 0.6 * 220
        elif name == "rgb_blur" and mod == "RGB":
            x = cv2.GaussianBlur(x, (0, 0), 3)
        elif name == "ir_noise" and mod == "IR":
            x = x + rng.normal(0, 20, x.shape)
        elif name == "ir_lowcontrast" and mod == "IR":
            x = 0.35 * (x - x.mean()) + x.mean()
        elif (name == "rgb_missing" and mod == "RGB") or (name == "ir_missing" and mod == "IR"):
            x = np.zeros_like(x)  # dead sensor: only meaningful for pixel-level fusion inputs
        return np.clip(x, 0, 255).astype(np.uint8)
    return f


# ----------------------------------------------------------------------------- inference
def infer(model, image_paths, imgsz, device=0, conf=0.01, batch=16, degrade=None, mod=None, max_det=10, scale=None):
    """Returns dict frame_key -> (N,5) detections in that image's own coordinates
    (multiplied by scale[key] when given, e.g. to map RGBs boxes back to full-resolution RGB)."""
    out = {}
    for i in range(0, len(image_paths), batch):
        chunk = image_paths[i:i + batch]
        ims = [cv2.imread(p) for p in chunk]
        if degrade is not None:
            ims = [degrade(mod, im) for im in ims]
        res = model.predict(ims, imgsz=imgsz, conf=conf, device=device, verbose=False, max_det=max_det)
        for p, r in zip(chunk, res):
            b = r.boxes
            d = np.c_[b.xyxy.cpu().numpy(), b.conf.cpu().numpy()] if len(b) else np.zeros((0, 5))
            k = os.path.splitext(os.path.basename(p))[0]
            if scale is not None and len(d):
                d[:, :4] *= scale[k]
            out[k] = d
    return out


def cascade_parts(model_crop, prep_dir, keys, dets_ir_cal, maps, crop=320, imgsz=320,
                  device=0, top_k=5, min_ir=0.05, degrade=None):
    """IR proposals -> native-resolution RGB crops -> crop verifier.
    Returns per frame a list of (ir_box_in_rgb(4), p_ir, crop_box_in_rgb(4) or None, raw_crop_score or None)."""
    parts = {}
    for k in keys:
        d = dets_ir_cal.get(k, np.zeros((0, 5)))
        d = d[d[:, 4] >= min_ir] if len(d) else d
        d = d[np.argsort(-d[:, 4])][:top_k] if len(d) else d
        parts[k] = []
        if len(d) == 0:
            continue
        m = maps[seq_of(k)]
        im = cv2.imread(os.path.join(prep_dir, "RGB", "images", k + ".jpg"))
        if degrade is not None:
            im = degrade("RGB", im)
        H, W = im.shape[:2]
        dr = m.boxes(d)
        crops, offs = [], []
        for row in dr:
            cx, cy = (row[0] + row[2]) / 2, (row[1] + row[3]) / 2
            x0 = int(np.clip(cx - crop / 2, 0, max(W - crop, 0))); y0 = int(np.clip(cy - crop / 2, 0, max(H - crop, 0)))
            crops.append(im[y0:y0 + crop, x0:x0 + crop]); offs.append((x0, y0))
        res = model_crop.predict(crops, imgsz=imgsz, conf=0.01, device=device, verbose=False, max_det=3)
        for row, r, (x0, y0) in zip(dr, res, offs):
            b = r.boxes
            if len(b):
                j = int(b.conf.argmax()); xy = b.xyxy[j].cpu().numpy()
                parts[k].append((row[:4], float(row[4]), np.array([xy[0] + x0, xy[1] + y0, xy[2] + x0, xy[3] + y0]),
                                 float(b.conf[j])))
            else:
                parts[k].append((row[:4], float(row[4]), None, None))
    return parts


def cascade_combine(parts, crop_cal, q_silent, prior):
    out = {}
    for k, lst in parts.items():
        rows = []
        for ir_box, p_ir, c_box, c_raw in lst:
            if c_box is not None:
                rows.append([*c_box, proben_pair(p_ir, float(crop_cal(c_raw)), prior)])
            else:
                rows.append([*ir_box, proben_pair(p_ir, q_silent, prior)])
        out[k] = np.array(rows, float).reshape(-1, 5)
    return out


def cascade_calibration(parts, gt_rgb):
    """Fit the crop-score calibration and the 'verifier silent' probability on validation parts."""
    s, y, silent_tp = [], [], []
    for k, lst in parts.items():
        g = gt_rgb.get(k)
        for ir_box, p_ir, c_box, c_raw in lst:
            box = c_box if c_box is not None else ir_box
            tp = int(g is not None and iou_matrix(np.asarray(box)[None], np.asarray(g)[None])[0, 0] >= 0.5)
            if c_box is not None:
                s.append(c_raw); y.append(tp)
            else:
                silent_tp.append(tp)
    cal = Platt().fit(np.array(s), np.array(y))
    q = float(np.clip(np.mean(silent_tp), 0.01, 0.99)) if silent_tp else 0.1
    return cal, q


# ----------------------------------------------------------------------------- evaluation helpers
def to_frame(dets, maps, direction):
    """Map detections between frames. direction: 'ir2rgb' or 'rgb2ir'."""
    out = {}
    for k, d in dets.items():
        m = maps.get(seq_of(k))
        if m is None or len(d) == 0:
            out[k] = d; continue
        out[k] = (m if direction == "ir2rgb" else m.inverse()).boxes(d)
    return out


def calibrate(dets, gt, keys):
    s, y = label_detections({k: dets.get(k, np.zeros((0, 5))) for k in keys}, {k: gt.get(k) for k in keys})
    return Platt().fit(s, y), (float(y.mean()) if len(y) else 0.5)


def apply_cal(dets, cal):
    return {k: (np.c_[d[:, :4], cal(d[:, 4])] if len(d) else d) for k, d in dets.items()}


def fuse_all(da, db, keys, rule, prior, scene=None, lr_model=None, tau=1.0):
    return {k: fuse_frame(da.get(k, np.zeros((0, 5))), db.get(k, np.zeros((0, 5))), rule, prior, tau,
                          lr_model, (scene or {}).get(k)) for k in keys}


def fit_learned(da, db, gt_ref, keys, scene, tau=1.0):
    """Train the condition-aware fusion model on validation frames (reference frame = gt_ref)."""
    X, y = [], []
    for k in keys:
        a, b = da.get(k, np.zeros((0, 5))), db.get(k, np.zeros((0, 5)))
        pairs, ua, ub = match_by_centre(a, b, tau)
        g = gt_ref.get(k)
        cand = [(a[i, 4], b[j, 4], True, True, 0.5 * (a[i, :4] + b[j, :4])) for i, j in pairs]
        cand += [(a[i, 4], None, True, False, a[i, :4]) for i in ua]
        cand += [(None, b[j, 4], False, True, b[j, :4]) for j in ub]
        for pa, pb, ha, hb, box in cand:
            X.append(LearnedFusion.feats(pa, pb, ha, hb, scene.get(k)))
            y.append(int(g is not None and iou_matrix(np.asarray(box)[None], np.asarray(g)[None])[0, 0] >= 0.5))
    return LearnedFusion().fit(X, y) if len(set(y)) > 1 else None


def eval_block(dets, gt_ref, keys, scene):
    r = ap_suite({k: dets.get(k, np.zeros((0, 5))) for k in keys}, {k: gt_ref.get(k) for k in keys},
                 frames=keys, size_bins=SIZE_BINS)
    r["recall_at_fppi_0.01"] = recall_at_fppi(dets, gt_ref, 0.01, keys)
    r["recall_at_fppi_0.1"] = recall_at_fppi(dets, gt_ref, 0.1, keys)
    r["ap50_by_light"] = {}
    for n, (lo, hi) in LIGHT_BINS.items():
        kk = [k for k in keys if lo <= scene[k]["rgb_brightness"] < hi]
        r["ap50_by_light"][n] = (average_precision(dets, gt_ref, 0.5, frames=kk) if kk else None)
        r["ap50_by_light"][n + "_frames"] = len(kk)
    return r
