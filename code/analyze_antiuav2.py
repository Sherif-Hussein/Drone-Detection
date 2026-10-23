#!/usr/bin/env python3
"""
Final offline analysis of the Anti-UAV300 fusion study (both reference frames), from saved raw detections.

python analyze_antiuav2.py --prep PREP --runs RUNS --out OUT.json [--bootstrap 1000]

Per seed and reference frame (IR, RGB) it builds every method's detections, then reports literature-standard
metrics (AP50, AP75, AP50:95, COCO S/M/L, finer size bins, illumination bins, best-F1 P/R/F1, recall at fixed
FPPI), seed mean/sd, seed-pooled sequence bootstrap of AP50 differences, mapping-free complementarity of the
two detectors, robustness under degraded/missing sensors (seed 0), a swap test and a mapping stress test.
"""
import argparse, glob, json, math, os, pickle, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import antiuav_pipeline as P  # noqa: E402
from fusion_core import AxisAffine, Platt, fuse_anchored, fuse_frame, fuse_many  # noqa: E402
from analyze_antiuav import (E, ap_from, ap_of_seqs, best_hit, bootstrap, ece, frame_records, reliability,  # noqa: E402
                             seq_pack, suite, cluster_bootstrap, holm, session_of_seq)

LIGHT = {"night": (0, 50), "dim": (50, 100), "day": (100, 256)}


def light_ap(dets, gt, keys, scene):
    out = {}
    for n, (lo, hi) in LIGHT.items():
        kk = [k for k in keys if lo <= scene[k]["rgb_brightness"] < hi]
        r = frame_records(dets, gt, kk)
        out[n] = ap_from(np.concatenate([v[0] for v in r.values()]) if r else np.zeros(0),
                         np.concatenate([v[1] for v in r.values()]) if r else np.zeros(0, bool), sum(v[2] for v in r.values()))
        out[n + "_frames"] = len(kk)
    return out


def recall_fppi(dets, gt, keys, fppi):
    r = frame_records(dets, gt, keys)
    sc = np.concatenate([v[0] for v in r.values()]); tp = np.concatenate([v[1] for v in r.values()])
    ng = sum(v[2] for v in r.values())
    if not len(sc) or not ng:
        return float("nan")
    o = np.argsort(-sc); ctp = np.cumsum(tp[o]); cfp = np.cumsum(~tp[o])
    ok = cfp <= fppi * len(keys)
    return float(ctp[ok].max() / ng) if ok.any() else 0.0


def full_metrics(dets, gt, keys, scene):
    m = suite(dets, gt, keys)
    for t in (0.1, 0.3):  # localisation tolerance
        r = frame_records(dets, gt, keys, t)
        m[f"AP{int(t * 100)}"] = ap_from(np.concatenate([v[0] for v in r.values()]), np.concatenate([v[1] for v in r.values()]),
                                         sum(v[2] for v in r.values()))
    m.update({"AP50_" + k: v for k, v in light_ap(dets, gt, keys, scene).items()})
    m["R@0.01FPPI"] = recall_fppi(dets, gt, keys, 0.01); m["R@0.1FPPI"] = recall_fppi(dets, gt, keys, 0.1)
    return m


def agg(per):
    keys = per[0].keys()
    return {k: {"mean": float(np.nanmean([p[k] for p in per])),
                "sd": float(np.nanstd([p[k] for p in per], ddof=1)) if len(per) > 1 else 0.0,
                "per_seed": [p[k] for p in per]} for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True); ap.add_argument("--runs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--cache", default=None, help="pickle of all methods' detections (written if absent, reused if present)")
    a = ap.parse_args()
    gt, meta = P.load_prepared(a.prep)
    scene = meta["scene"]
    mtv = json.load(open(os.path.join(a.runs, "maps_trainval_refit.json")))
    vmaps = {s: AxisAffine.from_dict(d) for s, d in mtv["per_seq"].items()}
    kt = sorted(k for k, v in meta["split_of"].items() if v == "test")
    kv = sorted(k for k, v in meta["split_of"].items() if v == "val")
    seq_frames = {}
    for k in kt:
        seq_frames.setdefault(P.seq_of(k), []).append(k)
    seqs = sorted(seq_frames)
    seeds = sorted(int(os.path.basename(f)[len("detections_s"):-4]) for f in glob.glob(os.path.join(a.runs, "detections_s*.pkl")))
    print("seeds", seeds, "test seqs", len(seqs), "frames", len(kt), flush=True)

    methods = {"IR": {}, "RGB": {}}
    cached = None
    if a.cache and os.path.exists(a.cache):
        cached = pickle.load(open(a.cache, "rb"))
        print("using cache", a.cache, flush=True)
    out = {"seeds": seeds, "n_test_seqs": len(seqs), "n_test_frames": len(kt)}
    robust, compl, calib = {}, [], {}
    for seed in ([] if cached else seeds):
        R = json.load(open(os.path.join(a.runs, f"results_antiuav_s{seed}.json")))
        D = pickle.load(open(os.path.join(a.runs, f"detections_s{seed}.pkl"), "rb"))
        cal = R["calibration"]; prior = cal["prior"]
        cir, crgb, cpf = (Platt(cal[x]["A"], cal[x]["B"]) for x in ("ir", "rgb", "pix"))
        cpfr = Platt(cal["pixrgb"]["A"], cal["pixrgb"]["B"]) if cal.get("pixrgb") else None
        tmaps = {s: AxisAffine.from_dict(d) for s, d in D["tmaps"].items()}
        oracle = P.gt_mappings(gt, meta, ("test",))
        omaps = {**tmaps, **oracle}
        C = {k: P.apply_cal(D[k], cir if k.startswith("ir") else crgb) for k in ("ir_val", "ir_test", "rgb_val", "rgb_test")}
        ir, rgb = C["ir_test"], C["rgb_test"]
        rgb_ir, ir_rgb = P.to_frame(rgb, tmaps, "rgb2ir"), P.to_frame(ir, tmaps, "ir2rgb")
        lrm = P.fit_learned(C["ir_val"], P.to_frame(C["rgb_val"], vmaps, "rgb2ir"), gt["IR"], kv, scene)
        cc = R["cascade"]["calibration"]; ccal = Platt(cc["crop"]["A"], cc["crop"]["B"])
        casc_rgb = P.cascade_combine(D["cascade_parts_test"], ccal, cc["q_silent"], prior)
        pf = P.apply_cal(D["pix_test"], cpf)
        pfr = P.apply_cal(D["pixrgb_test"], cpfr) if cpfr and "pixrgb_test" in D else None
        fa = lambda A, B, rule="proben", **kw: {k: fuse_anchored(A.get(k, E), B.get(k, E), rule, prior, **kw) for k in kt}
        # ------------------------------------------------ IR reference frame
        M = {"IR only": ir, "RGB only (mapped)": rgb_ir,
             "Late OR (box avg.)": P.fuse_all(ir, rgb_ir, kt, "or", prior),
             "Late AND (box avg.)": P.fuse_all(ir, rgb_ir, kt, "and", prior),
             "Late MAX (box avg.)": P.fuse_all(ir, rgb_ir, kt, "max", prior),
             "Late WBF (box avg.)": P.fuse_all(ir, rgb_ir, kt, "wbf", prior),
             "Late ProbEn (box avg.)": P.fuse_all(ir, rgb_ir, kt, "proben", prior),
             "Anchored ProbEn": fa(ir, rgb_ir), "Anchored MAX": fa(ir, rgb_ir, "max"),
             "Anchored ProbEn + unmatched RGB": fa(ir, rgb_ir, gamma=1.0),
             "Anchored ProbEn, GT-fitted map": fa(ir, P.to_frame(rgb, omaps, "rgb2ir")),
             "Late ProbEn (box avg.), GT-fitted map": P.fuse_all(ir, P.to_frame(rgb, omaps, "rgb2ir"), kt, "proben", prior),
             "Cascade (mapped)": P.to_frame(casc_rgb, tmaps, "rgb2ir"),
             "Pixel-level, IR grid": pf,
             "Hybrid (IR+RGB+pixel)": {k: fuse_many([ir.get(k, E), rgb_ir.get(k, E), pf.get(k, E)], prior) for k in kt}}
        # anchored hybrid: thermal boxes kept, scores updated first by the mapped RGB detector, then by the pixel-level one
        M["Anchored hybrid (IR+RGB+pixel)"] = fa(M["Anchored ProbEn"], pf)
        if lrm is not None:
            M["Late condition-aware (box avg.)"] = P.fuse_all(ir, rgb_ir, kt, "learned", prior, scene, lrm)
            M["Anchored condition-aware"] = {k: fuse_anchored(ir.get(k, E), rgb_ir.get(k, E), "learned", prior, lr_model=lrm, scene=scene.get(k)) for k in kt}
        if pfr is not None:
            M["Pixel-level, RGB grid (mapped)"] = P.to_frame(pfr, tmaps, "rgb2ir")
        ft = {}
        for s, ks in seq_frames.items():
            ft.update(P.temporal_smooth(ks, M["Anchored ProbEn"]))
        M["Anchored ProbEn + temporal"] = ft
        sw = {}
        for i, s in enumerate(seqs):  # swap test: RGB detections from another sequence
            o = seqs[(i + len(seqs) // 2) % len(seqs)]
            for j, k in enumerate(seq_frames[s]):
                kk = seq_frames[o][j % len(seq_frames[o])]
                d = rgb.get(kk, E)
                sw[k] = fuse_anchored(ir.get(k, E), tmaps[s].inverse().boxes(d) if len(d) else E, "proben", prior)
        M["Swap test (anchored ProbEn)"] = sw
        for dpx in (0, 10, 20, 40, 80):
            sm = {s: AxisAffine(m.ax, m.bx + dpx, m.ay, m.by) for s, m in tmaps.items()}
            rr = P.to_frame(rgb, sm, "rgb2ir")
            M[f"stress anchored {dpx}px"] = fa(ir, rr)
            M[f"stress box-avg {dpx}px"] = P.fuse_all(ir, rr, kt, "proben", prior)
        for n, d in M.items():
            methods["IR"].setdefault(n, []).append(d)
        # ------------------------------------------------ RGB reference frame
        N = {"RGB only": rgb, "IR only (mapped)": ir_rgb,
             "Late ProbEn (box avg.)": P.fuse_all(rgb, ir_rgb, kt, "proben", prior),
             "Anchored ProbEn": fa(rgb, ir_rgb), "Anchored MAX": fa(rgb, ir_rgb, "max"),
             "Cascade": casc_rgb,
             "Cascade + RGB detector": {k: fuse_frame(casc_rgb.get(k, E), rgb.get(k, E), "proben", prior) for k in kt},
             "Anchored: RGB + cascade": fa(rgb, casc_rgb),
             "Pixel-level, IR grid (mapped)": P.to_frame(pf, tmaps, "ir2rgb")}
        if pfr is not None:
            N["Pixel-level, RGB grid"] = pfr
        for n, d in N.items():
            methods["RGB"].setdefault(n, []).append(d)
        # ------------------------------------------------ mapping-free complementarity (each detector in its own frame)
        for k in kt:
            gi, gr = gt["IR"].get(k), gt["RGB"].get(k)
            if gi is None and gr is None:
                continue
            compl.append({"seed": seed, "k": k, "gi": gi is not None, "gr": gr is not None,
                          "ir": best_hit(ir.get(k, E), gi, 0.5) >= 0.5 if gi is not None else False,
                          "rgb": best_hit(rgb.get(k, E), gr, 0.5) >= 0.5 if gr is not None else False,
                          "ir_s": best_hit(ir.get(k, E), gi, 0.5) if gi is not None else 0.0,
                          "rgb_s": best_hit(rgb.get(k, E), gr, 0.5) if gr is not None else 0.0,
                          "size_ir": math.sqrt(max(gi[2] - gi[0], 0) * max(gi[3] - gi[1], 0)) if gi is not None else None,
                          "bright": scene[k]["rgb_brightness"]})
        # ------------------------------------------------ calibration
        for nm, raw, c, g in (("IR", D["ir_test"], cir, gt["IR"]), ("RGB", D["rgb_test"], crgb, gt["RGB"])):
            r = frame_records(raw, g, kt)
            s_ = np.concatenate([v[0] for v in r.values()]); y_ = np.concatenate([v[1] for v in r.values()]).astype(int)
            keep = s_ >= 0.05
            calib.setdefault(nm, []).append({"ece_raw": ece(s_[keep], y_[keep]), "ece_platt": ece(c(s_[keep]), y_[keep]),
                                             "reliability_raw": reliability(s_[keep], y_[keep]),
                                             "reliability_platt": reliability(c(s_[keep]), y_[keep])})
        # ------------------------------------------------ robustness (seed with degraded detections)
        if "degraded" in D:
            for cond, dd in D["degraded"].items():
                di = P.apply_cal(dd["ir"], cir) if dd.get("ir") is not None else (ir if cond != "ir_missing" else {k: E for k in kt})
                dr = P.apply_cal(dd["rgb"], crgb) if dd.get("rgb") is not None else (rgb if cond != "rgb_missing" else {k: E for k in kt})
                dri, dir_ = P.to_frame(dr, tmaps, "rgb2ir"), P.to_frame(di, tmaps, "ir2rgb")
                parts = dd["cascade_parts"]
                cr = P.cascade_combine(parts, ccal, cc["q_silent"], prior)
                res = {"IR frame": {"IR only": di, "RGB only (mapped)": dri, "Late ProbEn (box avg.)": P.fuse_all(di, dri, kt, "proben", prior),
                                    "Late OR (box avg.)": P.fuse_all(di, dri, kt, "or", prior),
                                    "Anchored ProbEn": {k: fuse_anchored(di.get(k, E), dri.get(k, E), "proben", prior) for k in kt},
                                    "Pixel-level, IR grid": P.apply_cal(dd["pix"], cpf)},
                       "RGB frame": {"RGB only": dr, "IR only (mapped)": dir_,
                                     "Anchored ProbEn": {k: fuse_anchored(dr.get(k, E), dir_.get(k, E), "proben", prior) for k in kt},
                                     "Cascade + RGB detector": {k: fuse_frame(cr.get(k, E), dr.get(k, E), "proben", prior) for k in kt}}}
                if dd.get("pixrgb") is not None and cpfr is not None:
                    res["RGB frame"]["Pixel-level, RGB grid"] = P.apply_cal(dd["pixrgb"], cpfr)
                # best single camera in either frame, with the other camera mapped in as a fallback
                robust[cond] = {fr: {n: frame_ap for n, frame_ap in
                                     ((n, suite(d, gt["IR" if fr == "IR frame" else "RGB"], kt)["AP50"]) for n, d in v.items())}
                                for fr, v in res.items()}
                print("robust", cond, robust[cond], flush=True)
            robust["clean"] = {"IR frame": {n: suite(methods["IR"][n][-1], gt["IR"], kt)["AP50"] for n in
                                            ("IR only", "RGB only (mapped)", "Late ProbEn (box avg.)", "Late OR (box avg.)", "Anchored ProbEn", "Pixel-level, IR grid")},
                               "RGB frame": {n: suite(methods["RGB"][n][-1], gt["RGB"], kt)["AP50"] for n in
                                             ("RGB only", "IR only (mapped)", "Anchored ProbEn", "Cascade + RGB detector")
                                             + (("Pixel-level, RGB grid",) if pfr is not None else ())}}
            out["robustness_seed"] = seed
    if cached:
        methods, robust, compl, calib, rseed = cached["methods"], cached["robust"], cached["compl"], cached["calib"], cached["rseed"]
        if rseed is not None:
            out["robustness_seed"] = rseed
    elif a.cache:
        pickle.dump({"methods": methods, "robust": robust, "compl": compl, "calib": calib, "rseed": out.get("robustness_seed")},
                    open(a.cache, "wb"))
    out["robustness_AP50"] = robust
    out["calibration"] = calib

    # ---------------------------------------------------- metrics per method (seed mean/sd)
    out["metrics"] = {}
    for fr, G in (("IR", gt["IR"]), ("RGB", gt["RGB"])):
        out["metrics"][fr] = {}
        for n, ds in methods[fr].items():
            if n.startswith("stress"):
                out["metrics"][fr][n] = agg([{"AP50": suite(d, G, kt)["AP50"]} for d in ds])
            else:
                out["metrics"][fr][n] = agg([full_metrics(d, G, kt, scene) for d in ds])
            print(fr, n, round(out["metrics"][fr][n]["AP50"]["mean"], 4), flush=True)

    # ---------------------------------------------------- bootstrap of AP50 differences (seed-pooled, sequences resampled)
    def packs(fr, n):
        G = gt["IR"] if fr == "IR" else gt["RGB"]
        return [seq_pack(frame_records(d, G, kt), seq_frames) for d in methods[fr][n]]
    comps = {"IR": [(m, "IR only") for m in ("Anchored ProbEn", "Anchored MAX", "Anchored condition-aware", "Anchored ProbEn + temporal",
                                             "Late ProbEn (box avg.)", "Late OR (box avg.)", "Late condition-aware (box avg.)",
                                             "Pixel-level, IR grid", "Hybrid (IR+RGB+pixel)", "Anchored hybrid (IR+RGB+pixel)", "Cascade (mapped)", "Swap test (anchored ProbEn)",
                                             "Anchored ProbEn, GT-fitted map")]
                   + [("Anchored ProbEn", "Late ProbEn (box avg.)")],
             "RGB": [(m, "RGB only") for m in ("Anchored ProbEn", "Cascade + RGB detector", "Anchored: RGB + cascade",
                                              "Pixel-level, RGB grid", "Late ProbEn (box avg.)")]}
    out["bootstrap_AP50"] = {}
    for fr, cl in comps.items():
        out["bootstrap_AP50"][fr] = {}
        for x, y in cl:
            if x in methods[fr] and y in methods[fr]:
                b = bootstrap(packs(fr, x), packs(fr, y), seqs, a.bootstrap)
                out["bootstrap_AP50"][fr][f"{x} - {y}"] = b
                print("boot", fr, x, "-", y, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in b.items() if k != "n_resamples"}, flush=True)

    # ---------------------------------------------------- session-cluster bootstrap, Holm correction, equivalence (+-0.01)
    out["sessions_test"] = sorted({session_of_seq(s) for s in seqs})
    out["cluster_bootstrap_AP50"] = {}
    for fr, cl in comps.items():
        res = {}
        for x, y in cl:
            if x in methods[fr] and y in methods[fr] and not x.startswith("Swap"):
                res[f"{x} - {y}"] = cluster_bootstrap(packs(fr, x), packs(fr, y), seqs, 2 * a.bootstrap)
        adj = holm({k: v["p_two_sided"] for k, v in res.items()})
        for k, v in res.items():
            v["p_holm"] = adj[k]
            print("cluster", fr, k, round(v["diff"], 4), [round(c, 4) for c in v["ci95"]], "p_holm", round(v["p_holm"], 4),
                  "equiv" if v["equivalent"] else "", flush=True)
        out["cluster_bootstrap_AP50"][fr] = res

    # ---------------------------------------------------- complementarity tables (frames where the target is visible in both)
    both = [r for r in compl if r["gi"] and r["gr"]]

    def tab(rows):
        if not rows:
            return None
        f = lambda c: float(np.mean([c(r) for r in rows]))
        return {"n": len(rows), "both": f(lambda r: r["ir"] and r["rgb"]), "ir_only": f(lambda r: r["ir"] and not r["rgb"]),
                "rgb_only": f(lambda r: r["rgb"] and not r["ir"]), "neither": f(lambda r: not r["ir"] and not r["rgb"]),
                "ir": f(lambda r: r["ir"]), "rgb": f(lambda r: r["rgb"]), "either": f(lambda r: r["ir"] or r["rgb"])}
    cm = {"all": tab(both)}
    for nm, (lo, hi) in {"<24": (0, 24), "24-32": (24, 32), "32-48": (32, 48), ">=48": (48, 1e9)}.items():
        cm["size " + nm] = tab([r for r in both if lo <= r["size_ir"] < hi])
    for nm, (lo, hi) in LIGHT.items():
        cm["light " + nm] = tab([r for r in both if lo <= r["bright"] < hi])
    cm["visible in IR only"] = tab([r for r in compl if r["gi"] and not r["gr"]])
    cm["visible in RGB only"] = tab([r for r in compl if r["gr"] and not r["gi"]])
    cm["n_frames_per_seed"] = len(both) // max(len(seeds), 1)
    out["complementarity_own_frame"] = cm

    # null for complementarity: two thermal detectors (different training seeds) judged the same way; and the IR/RGB
    # disagreement across the score threshold that defines a hit
    by = {}
    for r in both:
        by.setdefault(r["seed"], {})[r["k"]] = r
    def disagree(h1, h2):
        a1, a2 = np.array(h1), np.array(h2)
        return {"one_only": float(np.mean(a1 != a2)), "either": float(np.mean(a1 | a2)),
                "gain_over_better": float(np.mean(a1 | a2) - max(a1.mean(), a2.mean()))}
    null, cross = {}, {}
    for i, s1 in enumerate(seeds):
        for s2 in seeds[i + 1:]:
            ks_ = sorted(set(by.get(s1, {})) & set(by.get(s2, {})))
            if ks_:
                null[f"IR s{s1} vs IR s{s2}"] = disagree([by[s1][k]["ir"] for k in ks_], [by[s2][k]["ir"] for k in ks_])
                null[f"RGB s{s1} vs RGB s{s2}"] = disagree([by[s1][k]["rgb"] for k in ks_], [by[s2][k]["rgb"] for k in ks_])
    for s1 in seeds:
        rs = list(by.get(s1, {}).values())
        cross[f"s{s1}"] = {f"{t:.1f}": disagree([r["ir_s"] >= t for r in rs], [r["rgb_s"] >= t for r in rs])
                           for t in (0.1, 0.3, 0.5, 0.7, 0.9)}
    out["complementarity_null"] = {"same_modality_seed_pairs": null, "ir_vs_rgb_by_threshold": cross}
    print("complementarity null", json.dumps(out["complementarity_null"], indent=0)[:1500], flush=True)
    json.dump(out, open(a.out, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("saved", a.out)


if __name__ == "__main__":
    main()
