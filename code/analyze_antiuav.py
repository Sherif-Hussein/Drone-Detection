#!/usr/bin/env python3
"""
Offline analysis of the Anti-UAV300 fusion study from saved raw detections (no GPU needed).

Inputs : --prep  directory with gt.json, meta.json, maps_trainval.json (output of antiuav_prep.py)
         --runs  directory with results_antiuav_s<seed>.json and detections_s<seed>.pkl
Output : --out/analysis_antiuav.json with
   seed-pooled sequence bootstrap of AP50 differences, swap test, mapping stress test,
   calibration error before/after Platt scaling, and complementarity (who detects what) by size and light.

Fast evaluation: IoU matching is done once per frame, so bootstrap resamples only re-aggregate
per-frame (score, tp) lists.
"""
import argparse, glob, json, math, os, pickle, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import antiuav_pipeline as P  # noqa: E402
from fusion_core import AxisAffine, Platt, fuse_frame, fuse_many, iou_matrix  # noqa: E402

E = np.zeros((0, 5))


# ----------------------------------------------------------------------------- fast AP machinery
def frame_records(dets, gt, keys, iou_thr=0.5):
    """Per frame: (scores, tp flags, has_gt)."""
    rec = {}
    for k in keys:
        d = dets.get(k, E); g = gt.get(k)
        if len(d):
            d = d[np.argsort(-d[:, 4])]
            tp = np.zeros(len(d), bool)
            if g is not None:
                ious = iou_matrix(d[:, :4], np.asarray(g, float)[None, :4])[:, 0]
                hit = np.where(ious >= iou_thr)[0]
                if len(hit):
                    tp[hit[0]] = True
            rec[k] = (d[:, 4].copy(), tp)
        else:
            rec[k] = (np.zeros(0), np.zeros(0, bool))
        rec[k] = rec[k] + (g is not None,)
    return rec


COCO_BINS = {"S(<32^2)": (0, 32), "M(32^2-96^2)": (32, 96), "L(>96^2)": (96, 1e9)}
FINE_BINS = {"<24": (0, 24), "24-32": (24, 32), "32-48": (32, 48), ">=48": (48, 1e9)}


def _sq(b):
    return math.sqrt(max(b[2] - b[0], 0) * max(b[3] - b[1], 0))


def frame_records_sized(dets, gt, keys, rng, iou_thr=0.5):
    """COCO area convention: GT outside the size range is ignored (a detection matching it is ignored too);
    unmatched detections outside the range are ignored."""
    rec = {}
    for k in keys:
        d = dets.get(k, E); g = gt.get(k)
        ign_g = g is not None and not (rng[0] <= _sq(g) < rng[1])
        sc, tp = [], []
        if len(d):
            d = d[np.argsort(-d[:, 4])]
            used = False
            for row in d:
                hit = g is not None and not used and iou_matrix(row[None, :4], np.asarray(g, float)[None, :4])[0, 0] >= iou_thr
                if hit:
                    used = True
                    if not ign_g:
                        sc.append(row[4]); tp.append(True)
                elif rng[0] <= _sq(row) < rng[1]:
                    sc.append(row[4]); tp.append(False)
        rec[k] = (np.array(sc), np.array(tp, bool), g is not None and not ign_g)
    return rec


def suite(dets, gt, keys):
    """Metrics used in the literature: AP50, AP75, AP50:95, COCO S/M/L, finer size bins, best-F1 P/R/F1."""
    out = {}
    aps = []
    for t in np.arange(0.5, 0.96, 0.05):
        r = frame_records(dets, gt, keys, t)
        sc = np.concatenate([v[0] for v in r.values()]); tp = np.concatenate([v[1] for v in r.values()])
        ng = sum(v[2] for v in r.values())
        ap = ap_from(sc, tp, ng); aps.append(ap)
        if abs(t - 0.5) < 1e-6:
            out["AP50"] = ap
            o = np.argsort(-sc); tps = np.cumsum(tp[o]); fps = np.cumsum(~tp[o])
            if len(o) and ng:
                P_ = tps / np.maximum(tps + fps, 1); R_ = tps / ng; F_ = 2 * P_ * R_ / np.maximum(P_ + R_, 1e-9)
                i = int(np.argmax(F_)); out.update({"P": float(P_[i]), "R": float(R_[i]), "F1": float(F_[i]), "thr_bestF1": float(sc[o][i])})
        if abs(t - 0.75) < 1e-6:
            out["AP75"] = ap
    out["AP50_95"] = float(np.nanmean(aps))
    for nm, rg in {**COCO_BINS, **FINE_BINS}.items():
        r = frame_records_sized(dets, gt, keys, rg)
        sc = np.concatenate([v[0] for v in r.values()]); tp = np.concatenate([v[1] for v in r.values()])
        out["AP50_" + nm] = ap_from(sc, tp, sum(v[2] for v in r.values()))
        out["n_gt_" + nm] = int(sum(v[2] for v in r.values()))
    return out


def ap_from(scores, tps, n_gt):
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    o = np.argsort(-scores, kind="stable"); t = tps[o]
    ctp = np.cumsum(t); cfp = np.cumsum(~t)
    rec = ctp / n_gt; prec = ctp / np.maximum(ctp + cfp, 1e-9)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    idx = np.searchsorted(rec, np.linspace(0, 1, 101), side="left")
    return float(np.sum(np.where(idx < len(prec), prec[np.minimum(idx, len(prec) - 1)], 0.0)) / 101)


def seq_pack(rec, seq_frames):
    """Concatenate per-frame records per sequence -> (scores, tps, n_gt)."""
    out = {}
    for s, ks in seq_frames.items():
        sc = [rec[k][0] for k in ks]; tp = [rec[k][1] for k in ks]
        out[s] = (np.concatenate(sc) if sc else np.zeros(0), np.concatenate(tp) if tp else np.zeros(0, bool),
                  int(sum(rec[k][2] for k in ks)))
    return out


def ap_of_seqs(pack, seqs):
    sc = np.concatenate([pack[s][0] for s in seqs]); tp = np.concatenate([pack[s][1] for s in seqs])
    return ap_from(sc, tp, sum(pack[s][2] for s in seqs))


def bootstrap(packs_a, packs_b, seqs, n=1000, seed=0):
    """packs_*: list over training seeds of per-sequence packs. Statistic: mean over seeds of AP(a)-AP(b).
    Resamples test sequences (shared across seeds)."""
    rng = np.random.default_rng(seed)
    point = float(np.mean([ap_of_seqs(a, seqs) - ap_of_seqs(b, seqs) for a, b in zip(packs_a, packs_b)]))
    diffs = []
    for _ in range(n):
        pick = [seqs[i] for i in rng.integers(0, len(seqs), len(seqs))]
        diffs.append(np.mean([ap_of_seqs(a, pick) - ap_of_seqs(b, pick) for a, b in zip(packs_a, packs_b)]))
    diffs = np.array(diffs)
    return {"diff": point, "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "p_le_0": float(np.mean(diffs <= 0)), "n_seqs": len(seqs), "n_resamples": n}


def session_of_seq(s):
    """Recording session of a sequence name '<split>_<date>_<time>_<n>_<m>' -> '<date>_<time>'."""
    return "_".join(s.split("_", 1)[1].split("_")[:2])


def cluster_bootstrap(packs_a, packs_b, seqs, n=2000, seed=0, margin=0.01):
    """Two-level-safe bootstrap of the seed-mean AP difference that resamples whole recording sessions (clusters of
    test sequences), so that sequences filmed in the same session are not treated as independent.
    Returns the point difference, 95 % and 90 % percentile intervals, a two-sided bootstrap p-value and an
    equivalence (TOST) decision for the margin +-margin (90 % interval inside the margin)."""
    rng = np.random.default_rng(seed)
    cl = {}
    for s in seqs:
        cl.setdefault(session_of_seq(s), []).append(s)
    names = sorted(cl)
    point = float(np.mean([ap_of_seqs(a, seqs) - ap_of_seqs(b, seqs) for a, b in zip(packs_a, packs_b)]))
    diffs = []
    for _ in range(n):
        pick = [s for i in rng.integers(0, len(names), len(names)) for s in cl[names[i]]]
        diffs.append(np.mean([ap_of_seqs(a, pick) - ap_of_seqs(b, pick) for a, b in zip(packs_a, packs_b)]))
    diffs = np.array(diffs)
    p2 = float(min(1.0, 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))))
    lo90, hi90 = float(np.percentile(diffs, 5)), float(np.percentile(diffs, 95))
    return {"diff": point, "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "ci90": [lo90, hi90], "p_two_sided": p2, "equivalent": bool(-margin < lo90 and hi90 < margin),
            "margin": margin, "n_sessions": len(names), "n_seqs": len(seqs), "n_resamples": n}


def holm(pvals):
    """Holm step-down adjusted p-values for a dict name -> p."""
    items = sorted(pvals.items(), key=lambda t: t[1])
    m, run, out = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


# ----------------------------------------------------------------------------- helpers
def ece(probs, labels, bins=10):
    probs, labels = np.asarray(probs), np.asarray(labels)
    if len(probs) == 0:
        return float("nan")
    e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs >= lo) & (probs < hi if hi < 1 else probs <= hi)
        if m.any():
            e += m.mean() * abs(probs[m].mean() - labels[m].mean())
    return float(e)


def reliability(probs, labels, bins=10):
    probs, labels = np.asarray(probs), np.asarray(labels)
    edges = np.linspace(0, 1, bins + 1); out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs >= lo) & (probs < hi if hi < 1 else probs <= hi)
        out.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                    "conf": float(probs[m].mean()) if m.any() else None, "acc": float(labels[m].mean()) if m.any() else None})
    return out


def best_hit(d, g, thr):
    """Calibrated score of the best detection hitting GT g (IoU>=0.5), or 0."""
    if g is None or len(d) == 0:
        return 0.0
    ious = iou_matrix(d[:, :4], np.asarray(g, float)[None, :4])[:, 0]
    m = ious >= 0.5
    return float(d[m, 4].max()) if m.any() else 0.0


def shift_map(m, dx, dy):
    return AxisAffine(m.ax, m.bx + dx, m.ay, m.by + dy)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True); ap.add_argument("--runs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", type=int, default=1000)
    a = ap.parse_args()
    gt, meta = P.load_prepared(a.prep)
    scene = meta["scene"]
    mp = os.path.join(a.runs, "maps_trainval_refit.json")
    mtv = json.load(open(mp if os.path.exists(mp) else os.path.join(a.prep, "maps_trainval.json")))
    vmaps = {s: AxisAffine.from_dict(d) for s, d in mtv["per_seq"].items()}
    keys = {sp: sorted(k for k, v in meta["split_of"].items() if v == sp) for sp in ("val", "test")}
    kt = keys["test"]
    seq_frames = {}
    for k in kt:
        seq_frames.setdefault(P.seq_of(k), []).append(k)
    seqs = sorted(seq_frames)
    seeds = sorted(int(os.path.basename(f)[len("detections_s"):-4]) for f in glob.glob(os.path.join(a.runs, "detections_s*.pkl")))
    print("seeds", seeds, "test seqs", len(seqs), "test frames", len(kt))
    G = gt["IR"]
    methods = {}   # name -> list over seeds of fused detections in IR frame
    out = {"seeds": seeds, "n_test_seqs": len(seqs), "n_test_frames": len(kt)}
    calib = {}
    compl = []
    for seed in seeds:
        R = json.load(open(os.path.join(a.runs, f"results_antiuav_s{seed}.json")))
        D = pickle.load(open(os.path.join(a.runs, f"detections_s{seed}.pkl"), "rb"))
        cal = R["calibration"]; prior = cal["prior"]
        c_ir, c_rgb, c_pf = (Platt(cal[x]["A"], cal[x]["B"]) for x in ("ir", "rgb", "pix"))
        tmaps = {s: AxisAffine.from_dict(d) for s, d in D["tmaps"].items()}
        C = {k: P.apply_cal(D[k], c_ir if k.startswith("ir") else c_rgb) for k in ("ir_val", "ir_test", "rgb_val", "rgb_test")}
        ir = C["ir_test"]; rgb_ir = P.to_frame(C["rgb_test"], tmaps, "rgb2ir")
        lrm = P.fit_learned(C["ir_val"], P.to_frame(C["rgb_val"], vmaps, "rgb2ir"), G, keys["val"], scene)
        pf = P.apply_cal(D["pix_test"], c_pf)
        cc = R["cascade"]["calibration"]
        casc = P.cascade_combine(D["cascade_parts_test"], Platt(cc["crop"]["A"], cc["crop"]["B"]), cc["q_silent"], prior)
        M = {"IR-only": ir, "RGB-only": rgb_ir}
        for rule in ("or", "and", "max", "wbf", "proben"):
            M[rule] = P.fuse_all(ir, rgb_ir, kt, rule, prior)
        if lrm is not None:
            M["learned"] = P.fuse_all(ir, rgb_ir, kt, "learned", prior, scene, lrm)
        ft = {}
        for s, ks in seq_frames.items():
            ft.update(P.temporal_smooth(ks, M["proben"]))
        M["proben+temporal"] = ft
        M["cascade"] = P.to_frame(casc, tmaps, "rgb2ir")
        M["pixel"] = pf
        if "pixrgb_test" in D and cal.get("pixrgb"):
            M["pixel_rgbgrid"] = P.to_frame(P.apply_cal(D["pixrgb_test"], Platt(cal["pixrgb"]["A"], cal["pixrgb"]["B"])), tmaps, "rgb2ir")
        M["hybrid"] = {k: fuse_many([ir.get(k, E), rgb_ir.get(k, E), pf.get(k, E)], prior) for k in kt}
        # swap test: fuse each sequence's IR detections with RGB detections from another sequence
        sw = {}
        for i, s in enumerate(seqs):
            o = seqs[(i + len(seqs) // 2) % len(seqs)]
            ks, ko = seq_frames[s], seq_frames[o]
            for j, k in enumerate(ks):
                kk = ko[j % len(ko)]
                d = C["rgb_test"].get(kk, E)
                sw[k] = fuse_frame(ir.get(k, E), tmaps[s].inverse().boxes(d) if len(d) else E, "proben", prior)
        M["swap_test_proben"] = sw
        for name, d in M.items():
            methods.setdefault(name, []).append(d)
        # mapping stress (ProbEn and MAX), shift in RGB pixels
        for dpx in (0, 5, 10, 20, 40):
            sm = {s: shift_map(m, dpx, 0) for s, m in tmaps.items()}
            rg = P.to_frame(C["rgb_test"], sm, "rgb2ir")
            for rule in ("proben", "max"):
                methods.setdefault(f"stress_{rule}_{dpx}px", []).append(P.fuse_all(ir, rg, kt, rule, prior))
        # calibration error on test detections (raw vs Platt)
        for nm, raw, c, g in (("IR", D["ir_test"], c_ir, gt["IR"]), ("RGB", D["rgb_test"], c_rgb, gt["RGB"])):
            s_, y_ = [], []
            for k in kt:
                rr = frame_records({k: raw.get(k, E)}, {k: g.get(k)}, [k])[k]
                s_.extend(rr[0].tolist()); y_.extend(rr[1].astype(int).tolist())
            s_, y_ = np.array(s_), np.array(y_)
            keep = s_ >= 0.05
            calib.setdefault(nm, []).append({"ece_raw": ece(s_[keep], y_[keep]), "ece_platt": ece(c(s_[keep]), y_[keep]),
                                             "reliability_raw": reliability(s_[keep], y_[keep]),
                                             "reliability_platt": reliability(c(s_[keep]), y_[keep])})
        # complementarity: for each test frame with a visible target, does each sensor detect it (calibrated p>=0.5)?
        for k in kt:
            g = G.get(k)
            if g is None:
                continue
            pi, pr = best_hit(ir.get(k, E), g, 0.5), best_hit(rgb_ir.get(k, E), g, 0.5)
            pfz = best_hit(M["proben"].get(k, E), g, 0.5)
            compl.append({"seed": seed, "k": k, "ir": pi >= 0.5, "rgb": pr >= 0.5, "proben": pfz >= 0.5,
                          "size": math.sqrt(max(g[2] - g[0], 0) * max(g[3] - g[1], 0)),
                          "bright": scene[k]["rgb_brightness"], "rgb_gt": gt["RGB"].get(k) is not None})
    # ---------------------------------------------------------------- literature-standard metrics (IR frame)
    MAIN = ["IR-only", "RGB-only", "or", "and", "max", "wbf", "proben", "learned", "proben+temporal", "cascade",
            "pixel", "pixel_rgbgrid", "hybrid"]
    lit = {}
    for n in MAIN:
        if n in methods:
            per = [suite(d, G, kt) for d in methods[n]]
            lit[n] = {m: {"mean": float(np.nanmean([p.get(m, np.nan) for p in per])),
                          "sd": float(np.nanstd([p.get(m, np.nan) for p in per], ddof=1)) if len(per) > 1 else 0.0}
                      for m in per[0]}
            print("suite", n, {m: round(v["mean"], 3) for m, v in lit[n].items() if m.startswith("AP")}, flush=True)
    out["literature_metrics_IRref"] = lit

    # ---------------------------------------------------------------- per-method AP (mean, sd over seeds)
    recs = {n: [frame_records(d, G, kt) for d in ds] for n, ds in methods.items()}
    packs = {n: [seq_pack(r, seq_frames) for r in rs] for n, rs in recs.items()}
    summ = {}
    for n, ps in packs.items():
        v = [ap_of_seqs(p, seqs) for p in ps]
        summ[n] = {"ap50_mean": float(np.mean(v)), "ap50_sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "per_seed": v}
    out["ap50_IRref"] = summ
    # ---------------------------------------------------------------- bootstrap of differences
    best_single = "IR-only" if summ["IR-only"]["ap50_mean"] >= summ["RGB-only"]["ap50_mean"] else "RGB-only"
    out["best_single"] = best_single
    comps = [(m, "IR-only") for m in ("proben", "learned", "max", "wbf", "or", "and", "proben+temporal", "cascade", "pixel",
                                      "pixel_rgbgrid", "hybrid", "swap_test_proben")]
    comps += [("proben", "RGB-only"), ("hybrid", "proben"), ("pixel", "proben"), ("pixel_rgbgrid", "proben"),
              ("pixel_rgbgrid", "pixel"), ("cascade", "proben"), ("learned", "proben")]
    bs = {}
    for x, y in comps:
        if x in packs and y in packs:
            bs[f"{x}_minus_{y}"] = bootstrap(packs[x], packs[y], seqs, a.bootstrap)
            print(x, "-", y, bs[f"{x}_minus_{y}"], flush=True)
    out["bootstrap_ap50_IRref"] = bs
    out["calibration"] = calib
    # ---------------------------------------------------------------- complementarity tables
    def tab(rows):
        n = len(rows)
        if n == 0:
            return None
        f = lambda c: float(np.mean([c(r) for r in rows]))
        return {"n": n, "both": f(lambda r: r["ir"] and r["rgb"]), "ir_only": f(lambda r: r["ir"] and not r["rgb"]),
                "rgb_only": f(lambda r: r["rgb"] and not r["ir"]), "neither": f(lambda r: not r["ir"] and not r["rgb"]),
                "proben": f(lambda r: r["proben"]), "ir": f(lambda r: r["ir"]), "rgb": f(lambda r: r["rgb"])}
    cm = {"all": tab(compl)}
    for nm, (lo, hi) in P.SIZE_BINS.items():
        cm["size " + nm] = tab([r for r in compl if lo <= r["size"] < hi])
    for nm, (lo, hi) in P.LIGHT_BINS.items():
        cm["light " + nm] = tab([r for r in compl if lo <= r["bright"] < hi])
    cm["rgb_target_not_visible"] = tab([r for r in compl if not r["rgb_gt"]])
    out["complementarity_IRgt"] = cm
    json.dump(out, open(a.out, "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
