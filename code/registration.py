"""Per-frame (temporal-window) refinement of the IR->RGB map for unregistered gimbal cameras.

The sequence-level self-calibrated map fixes scale and offset for a whole recording, but zoom and pointing drift
within a recording. Here the scale is kept from the sequence map and the offset is re-estimated for every frame as
the median residual of confident IR/RGB detection pairs in neighbouring sampled frames. The current frame is
excluded from its own window (leave-one-out), so a detection seen by only one camera is placed using its neighbours,
exactly as it would be online. No ground truth is used. `oracle_frame_maps` gives the ground-truth upper bound.
"""
import numpy as np

from fusion_core import AxisAffine, centers

E = np.zeros((0, 5))


def _pair(di, dr, min_score):
    if len(di) and len(dr) and di[:, 4].max() >= min_score and dr[:, 4].max() >= min_score:
        return di[di[:, 4].argmax(), :4], dr[dr[:, 4].argmax(), :4]
    return None


def windowed_maps(dets_ir, dets_rgb, frames_of, seq_maps, window=2, min_score=0.5, include_self=False):
    """Returns {frame_key: AxisAffine}. Frames without enough neighbouring pairs keep the sequence map."""
    out = {}
    for seq, ks in frames_of.items():
        if seq not in seq_maps:
            continue
        m = seq_maps[seq]
        pairs = {k: _pair(dets_ir.get(k, E), dets_rgb.get(k, E), min_score) for k in ks}
        for t, k in enumerate(ks):
            nb = [ks[j] for j in range(max(0, t - window), min(len(ks), t + window + 1)) if include_self or j != t]
            res = []
            for kk in nb:
                p = pairs[kk]
                if p is None:
                    continue
                ci, cr = centers(np.asarray(p[0])[None])[0], centers(np.asarray(p[1])[None])[0]
                res.append(cr - np.array([m.ax * ci[0] + m.bx, m.ay * ci[1] + m.by]))
            if res:
                dx, dy = np.median(np.array(res), axis=0)
                out[k] = AxisAffine(m.ax, m.bx + dx, m.ay, m.by + dy)
            else:
                out[k] = m
    return out


def oracle_frame_maps(gt, frames_of, seq_maps):
    """Upper bound: sequence scale, offset that puts the RGB ground-truth centre exactly on the IR ground-truth
    centre in every frame where both exist (other frames keep the sequence map)."""
    out = {}
    for seq, ks in frames_of.items():
        if seq not in seq_maps:
            continue
        m = seq_maps[seq]
        for k in ks:
            gi, gr = gt["IR"].get(k), gt["RGB"].get(k)
            if gi is None or gr is None:
                out[k] = m; continue
            ci, cr = centers(np.asarray(gi, float)[None])[0], centers(np.asarray(gr, float)[None])[0]
            out[k] = AxisAffine(m.ax, cr[0] - m.ax * ci[0], m.ay, cr[1] - m.ay * ci[1])
    return out


def to_frame_pf(dets, frame_maps, direction):
    """Map detections with per-frame maps. direction: 'ir2rgb' or 'rgb2ir'."""
    out = {}
    for k, d in dets.items():
        m = frame_maps.get(k)
        if m is None or len(d) == 0:
            out[k] = d; continue
        out[k] = (m if direction == "ir2rgb" else m.inverse()).boxes(d)
    return out


def frame_map_error(frame_maps, gt, keys):
    e = []
    for k in keys:
        gi, gr = gt["IR"].get(k), gt["RGB"].get(k)
        if gi is None or gr is None or k not in frame_maps:
            continue
        e.append(np.linalg.norm(centers(frame_maps[k].boxes(np.asarray(gi, float)[None]))[0] - centers(np.asarray(gr, float)[None])[0]))
    return float(np.median(e)) if e else float("nan")
