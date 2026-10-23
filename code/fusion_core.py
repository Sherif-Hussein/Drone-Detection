"""
Core fusion utilities for RGB-IR drone detection (SkyFuse study).

Detections are numpy arrays of shape (N, 5): x1, y1, x2, y2, score (score in [0,1]).
Ground truth per frame: (4,) array x1,y1,x2,y2 or None.
Mappings are axis-aligned affine maps between camera frames:  X_dst = a * X_src + b  (per axis).
"""
import math
import numpy as np

EPS = 1e-6


# ----------------------------------------------------------------------------- geometry
class AxisAffine:
    """x' = ax*x + bx ; y' = ay*y + by. Boxes scale with (ax, ay)."""

    def __init__(self, ax=1.0, bx=0.0, ay=1.0, by=0.0):
        self.ax, self.bx, self.ay, self.by = float(ax), float(bx), float(ay), float(by)

    def to_dict(self):
        return {"ax": self.ax, "bx": self.bx, "ay": self.ay, "by": self.by}

    @staticmethod
    def from_dict(d):
        return AxisAffine(d["ax"], d["bx"], d["ay"], d["by"])

    def inverse(self):
        return AxisAffine(1 / self.ax, -self.bx / self.ax, 1 / self.ay, -self.by / self.ay)

    def boxes(self, b):
        if b is None or len(b) == 0:
            return np.zeros((0, b.shape[1] if b is not None and b.ndim == 2 else 5))
        out = b.copy().astype(float)
        out[:, 0] = self.ax * b[:, 0] + self.bx
        out[:, 2] = self.ax * b[:, 2] + self.bx
        out[:, 1] = self.ay * b[:, 1] + self.by
        out[:, 3] = self.ay * b[:, 3] + self.by
        return out

    def box(self, b):
        if b is None:
            return None
        return self.boxes(np.asarray(b, float)[None, :4])[0]


SCALE_BOUNDS = (1.5, 4.5)  # plausible RGB-px-per-IR-px scales for the Anti-UAV rig (measured median about 2.6)


def fit_axis_affine(src_pts, dst_pts, src_wh=None, dst_wh=None, ransac_iters=500, thr=None, seed=0,
                    scale_bounds=SCALE_BOUNDS):
    """Robust fit of per-axis affine from matched centres (and optionally box sizes for scale).
    RANSAC hypotheses with implausible scales are rejected; if the refined fit is still implausible (degenerate
    point sets, e.g. a gimbal that keeps the target near the image centre), the scale is taken from the median
    box-size ratio and the offset from the median residual. Returns (AxisAffine, inlier_mask) or (None, None)."""
    src = np.asarray(src_pts, float); dst = np.asarray(dst_pts, float)
    n = len(src)
    if n < 3:
        return None, None
    rng = np.random.default_rng(seed)
    if thr is None:
        thr = 0.02 * max(np.ptp(dst[:, 0]) + 1, np.ptp(dst[:, 1]) + 1, 50)
    ok = (lambda m: scale_bounds[0] <= m.ax <= scale_bounds[1] and scale_bounds[0] <= m.ay <= scale_bounds[1])         if scale_bounds else (lambda m: True)
    best, best_in = None, None

    def lsq(i):
        A = np.c_[src[i, 0], np.ones(len(i))]; ax, bx = np.linalg.lstsq(A, dst[i, 0], rcond=None)[0]
        A = np.c_[src[i, 1], np.ones(len(i))]; ay, by = np.linalg.lstsq(A, dst[i, 1], rcond=None)[0]
        return AxisAffine(ax, bx, ay, by)

    def size_fallback():
        if src_wh is None or dst_wh is None:
            return None, None
        sc = np.median(np.asarray(dst_wh, float) / np.maximum(np.asarray(src_wh, float), EPS), axis=0)
        if scale_bounds:
            sc = np.clip(sc, scale_bounds[0], scale_bounds[1])
        b = np.median(dst - src * sc, axis=0)
        m = AxisAffine(sc[0], b[0], sc[1], b[1])
        res = np.linalg.norm(np.c_[m.ax * src[:, 0] + m.bx, m.ay * src[:, 1] + m.by] - dst, axis=1)
        return m, res < max(thr, np.median(res) * 2)

    for _ in range(ransac_iters):
        i = rng.choice(n, size=2, replace=False)
        if np.ptp(src[i, 0]) < 1e-3 or np.ptp(src[i, 1]) < 1e-3:
            continue
        m = lsq(i)
        if not ok(m):
            continue
        pred = np.c_[m.ax * src[:, 0] + m.bx, m.ay * src[:, 1] + m.by]
        inl = np.linalg.norm(pred - dst, axis=1) < thr
        if best_in is None or inl.sum() > best_in.sum():
            best, best_in = m, inl
    if best is None or best_in.sum() < 3:
        return size_fallback()
    m = lsq(np.where(best_in)[0])
    if not ok(m):
        return size_fallback()
    return m, best_in


def centers(b):
    return np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2]


def wh(b):
    return np.c_[b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]]


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, EPS)


def nms(d, thr=0.5):
    if len(d) == 0:
        return d
    d = d[np.argsort(-d[:, 4])]
    keep = []
    rem = np.ones(len(d), bool)
    for i in range(len(d)):
        if not rem[i]:
            continue
        keep.append(i)
        if i + 1 < len(d):
            ious = iou_matrix(d[i:i + 1, :4], d[i + 1:, :4])[0]
            rem[i + 1:] &= ious < thr
    return d[keep]


def match_by_centre(a, b, tau=1.0):
    """Greedy cross-modal matching by centre distance normalised by box size (not IoU).
    Returns list of (i, j) pairs, unmatched_a, unmatched_b."""
    if len(a) == 0 or len(b) == 0:
        return [], list(range(len(a))), list(range(len(b)))
    ca, cb = centers(a), centers(b)
    sa = np.sqrt(np.prod(np.maximum(wh(a), 1), axis=1)); sb = np.sqrt(np.prod(np.maximum(wh(b), 1), axis=1))
    D = np.linalg.norm(ca[:, None] - cb[None], axis=2) / np.maximum(sa[:, None], sb[None, :])
    pairs, ua, ub = [], set(range(len(a))), set(range(len(b)))
    order = np.dstack(np.unravel_index(np.argsort(D, axis=None), D.shape))[0]
    for i, j in order:
        if D[i, j] > tau:
            break
        if i in ua and j in ub:
            pairs.append((int(i), int(j))); ua.discard(i); ub.discard(j)
    return pairs, sorted(ua), sorted(ub)


# ----------------------------------------------------------------------------- calibration
def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class Platt:
    """Platt scaling: p_cal = sigmoid(A * logit(score) + B), fitted by logistic regression."""

    def __init__(self, A=1.0, B=0.0):
        self.A, self.B = A, B

    def fit(self, scores, labels, iters=2000, lr=0.1, l2=1e-3):
        x = logit(np.asarray(scores, float)); y = np.asarray(labels, float)
        if len(x) == 0 or y.min() == y.max():
            # degenerate validation set (no correct or no wrong detections): constant probability equal to the
            # observed precision, instead of passing raw scores through uncalibrated
            prec = float(y.mean()) if len(y) else 0.0
            self.A, self.B = 0.0, float(logit(np.clip(prec, 1e-3, 1 - 1e-3)))
            return self
        if len(x) < 10:
            return self
        A, B = 1.0, 0.0
        for _ in range(iters):
            p = sigmoid(A * x + B)
            gA = np.mean((p - y) * x) + l2 * A; gB = np.mean(p - y)
            A -= lr * gA; B -= lr * gB
        self.A, self.B = float(A), float(B)
        return self

    def __call__(self, s):
        return sigmoid(self.A * logit(np.asarray(s, float)) + self.B)

    def to_dict(self):
        return {"A": self.A, "B": self.B}


def label_detections(dets_by_frame, gt_by_frame, iou_thr=0.5):
    """TP/FP labels for every detection (greedy by score), used for calibration."""
    scores, labels = [], []
    for k, d in dets_by_frame.items():
        g = gt_by_frame.get(k)
        if len(d) == 0:
            continue
        d = d[np.argsort(-d[:, 4])]
        used = False
        for row in d:
            tp = False
            if g is not None and not used:
                if iou_matrix(row[None, :4], np.asarray(g)[None, :4])[0, 0] >= iou_thr:
                    tp = True; used = True
            scores.append(row[4]); labels.append(int(tp))
    return np.array(scores), np.array(labels)


# ----------------------------------------------------------------------------- fusion rules
def proben_pair(p1, p2, prior):
    """ProbEn (Chen et al., ECCV 2022) for one class under conditional independence."""
    num = p1 * p2 * (1 - prior)
    den = num + (1 - p1) * (1 - p2) * prior
    return num / np.maximum(den, EPS)


def fuse_frame(da, db, rule, prior=0.5, tau=1.0, lr_model=None, scene=None):
    """Fuse two calibrated detection sets that are already in the SAME reference frame.
    da, db: (N,5) arrays. Returns fused (M,5)."""
    if rule == "A":
        return da.copy()
    if rule == "B":
        return db.copy()
    pairs, ua, ub = match_by_centre(da, db, tau)
    out = []
    for i, j in pairs:
        pa, pb = da[i, 4], db[j, 4]
        wa = pa / max(pa + pb, EPS)
        box = wa * da[i, :4] + (1 - wa) * db[j, :4]
        if rule == "or":
            s = max(pa, pb); box = da[i, :4] if pa >= pb else db[j, :4]
        elif rule == "and":
            s = min(pa, pb)
        elif rule == "max":
            s = max(pa, pb)
        elif rule == "wbf":
            s = (pa + pb) / 2
        elif rule == "proben":
            s = proben_pair(pa, pb, prior)
        elif rule == "learned":
            s = lr_model.predict(pa, pb, True, True, scene)
        else:
            raise ValueError(rule)
        out.append([*box, s])
    for i in ua:
        pa = da[i, 4]
        if rule == "and":
            continue
        s = {"or": pa, "max": pa, "wbf": pa / 2, "proben": pa}.get(rule)
        if rule == "learned":
            s = lr_model.predict(pa, None, True, False, scene)
        out.append([*da[i, :4], s])
    for j in ub:
        pb = db[j, 4]
        if rule == "and":
            continue
        s = {"or": pb, "max": pb, "wbf": pb / 2, "proben": pb}.get(rule)
        if rule == "learned":
            s = lr_model.predict(None, pb, False, True, scene)
        out.append([*db[j, :4], s])
    out = np.array(out, float).reshape(-1, 5)
    return nms(out, 0.5) if rule in ("or",) else out


def fuse_many(dets, prior=0.5, tau=1.0):
    """ProbEn over K>=2 detectors (hybrid). dets: list of (N_k,5) arrays in the same frame."""
    pool = [d.copy() for d in dets if len(d)]
    if not pool:
        return np.zeros((0, 5))
    fused = pool[0]
    for d in pool[1:]:
        fused = fuse_frame(fused, d, "proben", prior, tau)
    return fused


class LearnedFusion:
    """Condition-aware late fusion: logistic regression on
    [logit p_ir, logit p_rgb, present_ir, present_rgb, rgb_brightness, ir_contrast, interactions]."""

    def __init__(self):
        self.w = None

    @staticmethod
    def feats(pa, pb, ha, hb, scene):
        la = logit(pa) if ha else 0.0; lb = logit(pb) if hb else 0.0
        br = (scene or {}).get("rgb_brightness", 128) / 255.0
        ct = (scene or {}).get("ir_contrast", 30) / 100.0
        return np.array([1, la, lb, float(ha), float(hb), br, ct, la * ct, lb * br, float(ha and hb)])

    def fit(self, X, y, iters=3000, lr=0.05, l2=1e-3):
        X = np.asarray(X, float); y = np.asarray(y, float)
        w = np.zeros(X.shape[1])
        for _ in range(iters):
            p = sigmoid(X @ w)
            w -= lr * (X.T @ (p - y) / len(y) + l2 * w)
        self.w = w
        return self

    def predict(self, pa, pb, ha, hb, scene):
        return float(sigmoid(self.feats(pa, pb, ha, hb, scene) @ self.w))


def temporal_smooth(frames, dets_by_frame, alpha=0.6, tau=2.0):
    """Boost each detection by the best nearby detection in the previous sampled frame."""
    out, prev = {}, None
    for k in frames:
        d = dets_by_frame.get(k, np.zeros((0, 5))).copy()
        if prev is not None and len(prev) and len(d):
            pairs, _, _ = match_by_centre(d, prev, tau)
            for i, j in pairs:
                d[i, 4] = alpha * d[i, 4] + (1 - alpha) * max(d[i, 4], prev[j, 4])
        out[k] = d
        prev = d
    return out


# ----------------------------------------------------------------------------- evaluation
def average_precision(dets_by_frame, gt_by_frame, iou_thr=0.5, size_range=None, frames=None):
    """Single-class AP with 101-point interpolation (COCO style). Frames without GT contribute FPs.
    size_range=(lo,hi) on sqrt(area) of GT boxes; dets matched to ignored GT and unmatched dets
    outside the range are ignored (COCO area convention)."""
    frames = frames if frames is not None else sorted(set(gt_by_frame) | set(dets_by_frame))
    recs, n_gt = [], 0
    for k in frames:
        g = gt_by_frame.get(k)
        ign_g = False
        if g is not None and size_range is not None:
            s = math.sqrt(max(g[2] - g[0], 0) * max(g[3] - g[1], 0))
            ign_g = not (size_range[0] <= s < size_range[1])
        if g is not None and not ign_g:
            n_gt += 1
        d = dets_by_frame.get(k, np.zeros((0, 5)))
        if len(d) == 0:
            continue
        d = d[np.argsort(-d[:, 4])]
        used = False
        for row in d:
            tp, ignore = 0, False
            if g is not None and not used and iou_matrix(row[None, :4], np.asarray(g)[None, :4])[0, 0] >= iou_thr:
                used = True
                if ign_g:
                    ignore = True
                else:
                    tp = 1
            elif size_range is not None:
                s = math.sqrt(max(row[2] - row[0], 0) * max(row[3] - row[1], 0))
                if not (size_range[0] <= s < size_range[1]):
                    ignore = True
            if not ignore:
                recs.append((row[4], tp))
    if n_gt == 0:
        return float("nan")
    if not recs:
        return 0.0
    recs.sort(key=lambda r: -r[0])
    tps = np.cumsum([r[1] for r in recs]); fps = np.cumsum([1 - r[1] for r in recs])
    rec = tps / n_gt; prec = tps / np.maximum(tps + fps, EPS)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        idx = np.searchsorted(rec, t, side="left")
        ap += prec[idx] if idx < len(prec) else 0.0
    return ap / 101


def ap_suite(dets_by_frame, gt_by_frame, frames=None, size_bins=None):
    res = {"ap50": average_precision(dets_by_frame, gt_by_frame, 0.5, frames=frames)}
    res["ap50_95"] = float(np.nanmean([average_precision(dets_by_frame, gt_by_frame, t, frames=frames)
                                       for t in np.arange(0.5, 0.96, 0.05)]))
    if size_bins:
        res["ap50_by_size"] = {name: average_precision(dets_by_frame, gt_by_frame, 0.5, rng, frames)
                               for name, rng in size_bins.items()}
    return res


def recall_at_fppi(dets_by_frame, gt_by_frame, fppi=0.01, frames=None):
    """Recall at a fixed false-positive-per-image rate (operator-oriented metric)."""
    frames = frames if frames is not None else sorted(set(gt_by_frame) | set(dets_by_frame))
    recs, n_gt = [], 0
    for k in frames:
        g = gt_by_frame.get(k); n_gt += g is not None
        d = dets_by_frame.get(k, np.zeros((0, 5)))
        d = d[np.argsort(-d[:, 4])] if len(d) else d
        used = False
        for row in d:
            tp = 0
            if g is not None and not used and iou_matrix(row[None, :4], np.asarray(g)[None, :4])[0, 0] >= 0.5:
                tp = 1; used = True
            recs.append((row[4], tp))
    if n_gt == 0 or not recs:
        return float("nan")
    recs.sort(key=lambda r: -r[0])
    tps = np.cumsum([r[1] for r in recs]); fps = np.cumsum([1 - r[1] for r in recs])
    ok = fps <= fppi * len(frames)
    return float(tps[ok].max() / n_gt) if ok.any() else 0.0


def bootstrap_diff(seq_frames, fa, fb, gt, n=1000, seed=0):
    """95% CI of AP50(fa) - AP50(fb), resampling test sequences with replacement."""
    rng = np.random.default_rng(seed)
    seqs = sorted(seq_frames)
    diffs = []
    for _ in range(n):
        pick = rng.choice(len(seqs), size=len(seqs), replace=True)
        frames, ren_a, ren_b, ren_g = [], {}, {}, {}
        for t, si in enumerate(pick):
            for k in seq_frames[seqs[si]]:
                kk = (t, k); frames.append(kk)
                ren_a[kk] = fa.get(k, np.zeros((0, 5))); ren_b[kk] = fb.get(k, np.zeros((0, 5)))
                if gt.get(k) is not None:
                    ren_g[kk] = gt[k]
        diffs.append(average_precision(ren_a, ren_g, 0.5, frames=frames) -
                     average_precision(ren_b, ren_g, 0.5, frames=frames))
    diffs = np.array(diffs)
    return {"mean": float(np.mean(diffs)), "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "p_le_0": float(np.mean(diffs <= 0))}


def fuse_anchored(dref, doth, rule="proben", prior=0.5, tau=1.0, gamma=0.0, lr_model=None, scene=None):
    """Reference-anchored late fusion. Boxes always come from the reference camera, whose detector saw the
    image the output is drawn on; the other camera (whose boxes arrive through an imperfect cross-camera map)
    only changes the score of matched reference detections. Unmatched detections of the other camera are
    kept at gamma times their score (gamma = 0 discards them)."""
    pairs, ua, ub = match_by_centre(dref, doth, tau)
    out = []
    for i, j in pairs:
        pa, pb = dref[i, 4], doth[j, 4]
        if rule == "proben":
            s = proben_pair(pa, pb, prior)
        elif rule == "max":
            s = max(pa, pb)
        elif rule == "wbf":
            s = (pa + pb) / 2
        elif rule == "learned":
            s = lr_model.predict(pa, pb, True, True, scene)
        else:
            raise ValueError(rule)
        out.append([*dref[i, :4], s])
    for i in ua:
        s = dref[i, 4] if rule != "learned" else lr_model.predict(dref[i, 4], None, True, False, scene)
        out.append([*dref[i, :4], s])
    if gamma > 0:
        for j in ub:
            out.append([*doth[j, :4], gamma * doth[j, 4]])
    return np.array(out, float).reshape(-1, 5)
