"""Anti-UAV300 part of make_numbers.py (final analysis structure). Adds macros to M."""
import glob, json, os
import numpy as np


def add_antiuav(M, DATA, PEND):
    RUNS = os.path.join(DATA, "runs")
    R = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(RUNS, "results_antiuav_s*.json")))]
    ap_ = os.path.join(DATA, "analysis_antiuav_rev.json")
    if not os.path.exists(ap_):
        ap_ = os.path.join(DATA, "analysis_antiuav.json")
    an = json.load(open(ap_)) if os.path.exists(ap_) else None

    def ms_(d, k, nd=3, sd=True):
        if not d or k not in d or d[k]["mean"] is None or np.isnan(d[k]["mean"]):
            return PEND
        v = d[k]
        return f"{v['mean']:.{nd}f}" + (f"$\\pm${v['sd']:.{nd}f}" if sd and len(v["per_seed"]) > 1 else "")

    if an:
        MI, MR = an["metrics"]["IR"], an["metrics"]["RGB"]
        BI, BR = an["bootstrap_AP50"]["IR"], an["bootstrap_AP50"]["RGB"]
        CBI, CBR = an.get("cluster_bootstrap_AP50", {}).get("IR", {}), an.get("cluster_bootstrap_AP50", {}).get("RGB", {})
        M["ANumSeeds"] = str(len(an["seeds"]))
        ROWS = [("IR", "IR only", "IR only", None), ("IR", "RGB only (mapped)", "RGB only, mapped", None),
                ("IR", "Late OR (box avg.)", "Late OR (box averaging)", "Late OR (box avg.) - IR only"),
                ("IR", "Late ProbEn (box avg.)", "Late ProbEn (box averaging)", "Late ProbEn (box avg.) - IR only"),
                ("IR", "Late condition-aware (box avg.)", "Late cond.-aware (box averaging)", "Late condition-aware (box avg.) - IR only"),
                ("IR", "Anchored ProbEn", "Anchored ProbEn", "Anchored ProbEn - IR only"),
                ("IR", "Anchored ProbEn + unmatched RGB", "Anchored ProbEn + unmatched RGB", "Anchored ProbEn + unmatched RGB - IR only"),
                ("IR", "Anchored MAX", "Anchored MAX", "Anchored MAX - IR only"),
                ("IR", "Anchored condition-aware", "Anchored cond.-aware", "Anchored condition-aware - IR only"),
                ("IR", "Pixel-level, IR grid", "Pixel-level, IR grid", "Pixel-level, IR grid - IR only"),
                ("IR", "Hybrid (IR+RGB+pixel)", "Hybrid (box averaging)", "Hybrid (IR+RGB+pixel) - IR only"),
                ("IR", "Anchored hybrid (IR+RGB+pixel)", "Hybrid (anchored)", "Anchored hybrid (IR+RGB+pixel) - IR only"),
                ("RGB", "RGB only", "RGB only", None), ("RGB", "IR only (mapped)", "IR only, mapped", None),
                ("RGB", "Late ProbEn (box avg.)", "Late ProbEn (box averaging)", "Late ProbEn (box avg.) - RGB only"),
                ("RGB", "Anchored ProbEn", "Anchored ProbEn", "Anchored ProbEn - RGB only"),
                ("RGB", "Cascade", "Cascade", None),
                ("RGB", "Cascade + RGB detector", "Cascade + RGB detector", "Cascade + RGB detector - RGB only"),
                ("RGB", "Pixel-level, RGB grid", "Pixel-level, RGB grid\\textsuperscript{a}", "Pixel-level, RGB grid - RGB only")]
        COLS = ["AP50", "AP50_95", "AP10", "AP50_S(<32^2)", "AP50_night", "AP50_day", "R@0.01FPPI"]
        lines, cur = [], None
        for fr, key, label, bkey in ROWS:
            T = MI if fr == "IR" else MR
            d = T.get(key)
            if d is None:
                continue
            if fr != cur:
                lines.append(("\\midrule\n" if cur else "") +
                             f"\\multicolumn{{9}}{{@{{}}l}}{{\\textit{{{fr} image frame (scored against the {fr} camera's ground truth)}}}} \\\\")
                cur = fr
            best = {}
            for c in COLS:
                vals = [round(T[r[1]][c]["mean"], 3) for r in ROWS if r[0] == fr and r[1] in T and c in T[r[1]] and not np.isnan(T[r[1]][c]["mean"])]
                best[c] = max(vals) if vals else None
            cells = []
            for c in COLS:
                if c not in d or np.isnan(d[c]["mean"]):
                    cells.append("--"); continue
                v = d[c]
                t = f"{v['mean']:.3f}" + (f"\\,{{\\scriptsize$\\pm${v['sd']:.3f}}}" if len(v["per_seed"]) > 1 and c == "AP50" else "")
                cells.append(f"\\textbf{{{t}}}" if best[c] is not None and round(v["mean"], 3) == best[c] else t)
            b = (CBI if fr == "IR" else CBR).get(bkey) if bkey else None
            if b:
                mark = ("$^{*}$" if b["p_holm"] < 0.05 else "") + ("$^{\\dagger}$" if b["equivalent"] else "")
                cells.append(f"{b['diff']:+.3f} [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}]{mark}")
            else:
                cells.append("--")
            lines.append(f"\\quad {label} & " + " & ".join(cells) + " \\\\")
        M["TabMainRows"] = "\n".join(lines)
        for fr, n, mac in (("IR", "IR only", "AIRonly"), ("IR", "RGB only (mapped)", "ARGBmap"), ("RGB", "RGB only", "ARGBonly"),
                           ("RGB", "IR only (mapped)", "AIRmap"), ("IR", "Late ProbEn (box avg.)", "AProbenAvg"),
                           ("IR", "Late OR (box avg.)", "AOrAvg"), ("IR", "Late WBF (box avg.)", "AWbfAvg"),
                           ("IR", "Anchored hybrid (IR+RGB+pixel)", "AHybridAnch"), ("IR", "Late condition-aware (box avg.)", "ALearnedAvg"),
                           ("IR", "Anchored ProbEn", "AAnch"), ("IR", "Anchored ProbEn + unmatched RGB", "AAnchU"),
                           ("IR", "Anchored condition-aware", "AAnchL"), ("IR", "Anchored MAX", "AAnchMax"),
                           ("IR", "Anchored ProbEn + temporal", "AAnchT"), ("IR", "Pixel-level, IR grid", "APixIR"),
                           ("IR", "Hybrid (IR+RGB+pixel)", "AHybrid"), ("IR", "Cascade (mapped)", "ACascMap"),
                           ("RGB", "Cascade", "ACasc"), ("RGB", "Cascade + RGB detector", "ACascRGB"), ("RGB", "Pixel-level, RGB grid", "APixRGB"),
                           ("RGB", "Anchored ProbEn", "AAnchRGB"), ("RGB", "Late ProbEn (box avg.)", "AProbenAvgRGB"),
                           ("IR", "Swap test (anchored ProbEn)", "ASwap"), ("IR", "Anchored ProbEn, GT-fitted map", "AAnchOracle"),
                           ("IR", "Late ProbEn (box avg.), GT-fitted map", "AProbenAvgOracle")):
            T = MI if fr == "IR" else MR
            for c, suf in (("AP50", "AP"), ("AP50_95", "APS"), ("AP10", "APten"), ("AP30", "APthirty"), ("AP75", "APsevfive"),
                           ("AP50_S(<32^2)", "Small"), ("AP50_M(32^2-96^2)", "Medium"), ("AP50_L(>96^2)", "Large"),
                           ("AP50_night", "Night"), ("AP50_day", "Day"), ("R@0.01FPPI", "Rlo"), ("P", "Prec"), ("R", "Rec"), ("F1", "Fone")):
                M[mac + suf] = ms_(T.get(n), c, sd=(c == "AP50"))
        for fr, B in (("IR", BI), ("RGB", BR)):
            for k, b in B.items():
                mac = "Boot" + "".join(ch for ch in (fr + " " + k).title() if ch.isalpha())
                M[mac] = f"{b['diff']:+.3f} [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}]"
        for fr, B in (("IR", CBI), ("RGB", CBR)):
            for k, b in B.items():
                mac = "CBoot" + "".join(ch for ch in (fr + " " + k).title() if ch.isalpha())
                M[mac] = f"{b['diff']:+.3f} [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}]"
                M[mac + "Up"] = f"{b['ci95'][1]:+.3f}"
                M[mac + "P"] = f"{b['p_holm']:.2f}" if b["p_holm"] >= 0.01 else chr(92) + "ensuremath{<}0.01"
        sw = an.get("cluster_bootstrap_swap")
        if sw:
            M["CBootSwap"] = f"{sw['diff']:+.3f} [{sw['ci95'][0]:+.3f}, {sw['ci95'][1]:+.3f}]"
        if CBI:
            M["NSessTest"] = str(next(iter(CBI.values()))["n_sessions"])
            M["NCompHolm"] = str(len(CBI))
            M["NEquivIR"] = str(sum(b["equivalent"] for k, b in CBI.items() if not k.startswith("Swap")))
        cn = an.get("complementarity_null", {})
        if cn:
            nl = cn["same_modality_seed_pairs"]
            for mod in ("IR", "RGB"):
                v = [x for k, x in nl.items() if k.startswith(mod + " ")]
                if v:
                    M[f"CNull{mod}One"] = f"{100 * np.mean([x['one_only'] for x in v]):.1f}"
                    M[f"CNull{mod}Gain"] = f"{100 * np.mean([x['gain_over_better'] for x in v]):.1f}"
            th = cn["ir_vs_rgb_by_threshold"]
            for t, tw in (("0.1", "Lo"), ("0.5", "Mid"), ("0.9", "Hi")):
                M[f"CThr{tw}One"] = f"{100 * np.mean([th[s_][t]['one_only'] for s_ in th]):.1f}"
                M[f"CThr{tw}Gain"] = f"{100 * np.mean([th[s_][t]['gain_over_better'] for s_ in th]):.1f}"
        nps = an["complementarity_own_frame"].get("n_frames_per_seed")
        if nps:
            M["CAllNseed"] = f"{nps:,}"
        cvp = os.path.join(RUNS, "loso_cv.json")
        if os.path.exists(cvp):
            from collections import Counter
            cv = json.load(open(cvp))
            for k, b in cv["cluster_bootstrap_AP50"].items():
                mac = "CV" + "".join(ch for ch in k.title() if ch.isalpha())
                M[mac] = f"{b['diff']:+.3f} [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}]"
                M[mac + "P"] = f"{b['p_holm']:.2f}" if b["p_holm"] >= 0.01 else chr(92) + "ensuremath{<}0.01"
            ch = [tuple(v["cfg"]) for sd in cv["per_seed"].values() for v in sd["chosen"].values()]
            top = Counter(c[0] for c in ch).most_common(1)[0]
            M["CVNFolds"] = str(len(ch))
            M["CVTopRule"] = {"proben": "ProbEn", "max": "MAX", "wbf": "WBF"}[top[0]]
            M["CVTopRuleN"] = str(top[1])
            M["CVNDistinct"] = str(len(set(ch)))
            M["CVTopCfgN"] = str(Counter(ch).most_common(1)[0][1])
        for d, w in ((0, "Zero"), (10, "Ten"), (20, "Twenty"), (40, "Forty"), (80, "Eighty")):
            for kind, nm in (("anchored", "Anch"), ("box-avg", "Avg")):
                M[f"Stress{nm}{w}"] = ms_(MI.get(f"stress {kind} {d}px"), "AP50", sd=False)
        cm = an["complementarity_own_frame"]
        for k, mac in (("all", "CAll"), ("size <24", "CSa"), ("size 24-32", "CSb"), ("size 32-48", "CSc"), ("size >=48", "CSd"),
                       ("light night", "CNight"), ("light dim", "CDim"), ("light day", "CDay"), ("visible in IR only", "CIRvis")):
            t = cm.get(k)
            if t:
                for f_, suf in (("both", "Both"), ("ir_only", "IRo"), ("rgb_only", "RGBo"), ("neither", "None"), ("ir", "IR"),
                                ("rgb", "RGB"), ("either", "Either")):
                    M[mac + suf] = f"{100 * t[f_]:.1f}"
                M[mac + "Gain"] = f"{100 * (t['either'] - max(t['ir'], t['rgb'])):.1f}"
                M[mac + "One"] = f"{100 * (t['ir_only'] + t['rgb_only']):.1f}"
                M[mac + "N"] = f"{t['n']:,}"
        c = an["calibration"]
        for s in ("IR", "RGB"):
            M[f"ECE{s}raw"] = f"{np.mean([x['ece_raw'] for x in c[s]]):.3f}"
            M[f"ECE{s}cal"] = f"{np.mean([x['ece_platt'] for x in c[s]]):.3f}"
        rb = an.get("robustness_AP50", {})
        for cond, cw in (("rgb_dark", "RgbDark"), ("rgb_fog", "RgbFog"), ("rgb_blur", "RgbBlur"), ("ir_noise", "IrNoise"),
                         ("ir_lowcontrast", "IrLow"), ("rgb_missing", "RgbMiss"), ("ir_missing", "IrMiss"), ("clean", "Clean")):
            for fr, frw in (("IR frame", "I"), ("RGB frame", "R")):
                for n, nw in (("IR only", "IRonly"), ("RGB only", "RGBonly"), ("Anchored ProbEn", "Anch"), ("Late ProbEn (box avg.)", "Avg"),
                              ("Late OR (box avg.)", "Or"), ("Pixel-level, IR grid", "PixIR"), ("Pixel-level, RGB grid", "PixRGB"),
                              ("Cascade + RGB detector", "CascRGB"), ("RGB only (mapped)", "RGBmap"), ("IR only (mapped)", "IRmap")):
                    v = rb.get(cond, {}).get(fr, {}).get(n)
                    if v is not None:
                        M[f"Rob{cw}{frw}{nw}"] = f"{v:.3f}"
    RC = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(RUNS, "registration_curve_s*.json")))]
    if RC:
        xs = np.mean([[c["map_err_px"] for c in r["curve"]] for r in RC], axis=0)
        for key, nm in (("box-avg ProbEn", "Avg"), ("anchored ProbEn + unmatched RGB", "Anch")):
            ys = np.mean([[c[key] for c in r["curve"]] for r in RC], axis=0)
            M[f"RegGain{nm}Zero"] = f"{ys[0]:+.3f}"
            i12 = int(np.argmin(np.abs(xs - 12))); M[f"RegGain{nm}Twelve"] = f"{ys[i12]:+.3f}"; M["RegErrTwelve"] = f"{xs[i12]:.0f}"
            cross = next((xs[i - 1] + (xs[i] - xs[i - 1]) * ys[i - 1] / (ys[i - 1] - ys[i]) for i in range(1, len(ys)) if ys[i - 1] > 0 >= ys[i]), None)
            M[f"RegCross{nm}"] = f"{cross:.0f}" if cross is not None else PEND
            for pname, pw in (("sequence map", "Seq"), ("windowed map (w=3)", "Win")):
                M[f"RegPt{pw}{nm}"] = f"{np.mean([r['points'][pname][key] for r in RC]):+.3f}"
                M[f"RegErr{pw}"] = f"{np.mean([r['points'][pname]['map_err_px'] for r in RC]):.0f}"
    RC2 = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(RUNS, "registration_curve2_s*.json")))]
    if RC2:
        # crossing points pooled over seeds: mean point, and the envelope of the per-seed 95 % session-bootstrap intervals
        for model, mw in (("gauss", "Gauss"), ("empirical", "Emp"), ("coverage", "Cov")):
            for key, nm in (("box-avg ProbEn", "Avg"), ("anchored ProbEn + unmatched RGB", "Anch")):
                cs = [r["crossings"][model][key] for r in RC2]
                f = 100 if model == "coverage" else 1
                pts = [c["point"] for c in cs if c["point"] is not None]
                M[f"RegX{mw}{nm}"] = f"{f * np.mean(pts):.0f}" if pts else PEND
                lo = min(c["ci95"][0] for c in cs if c["ci95"]); hi = max(c["ci95"][1] for c in cs if c["ci95"])
                M[f"RegX{mw}{nm}CI"] = f"{f * lo:.0f}--{f * hi:.0f}"
        emp = [r["curves"]["empirical"] for r in RC2]
        i1 = int(np.argmin(np.abs(np.array([c["x"] for c in emp[0]]) - 1.0)))
        M["RegEmpAtWin"] = f"{np.mean([e[i1]['anchored ProbEn + unmatched RGB'] for e in emp]):+.3f}"
        M["RegEmpAtWinAvg"] = f"{np.mean([e[i1]['box-avg ProbEn'] for e in emp]):+.3f}"
    if R:
        m = [r["mapping_test"] for r in R if "mapping_test" in r]
        M["MapSelf"] = f"{100 * np.mean([x['self_calibrated_share'] for x in m]):.0f}"
        M["MapErrSelf"] = f"{np.mean([x['median_err_px_self'] for x in m]):.0f}"
        M["MapErrGlobal"] = f"{np.mean([x['median_err_px_global'] for x in m]):.0f}"
        M["MapErrOracle"] = f"{np.mean([x['median_err_px_oracle'] for x in m]):.0f}"
        lat = [r["latency_ms"] for r in R if "latency_ms" in r]
        for k, n in (("ir_detector", "LatIR"), ("rgb_detector", "LatRGB"), ("crop_verifier_per_candidate", "LatCas"),
                     ("pixel_fusion_detector", "LatPix"), ("pixel_fusion_rgbgrid_detector", "LatPixRGB"),
                     ("late_fusion_proben_per_frame_cpu", "LatFuse")):
            v = [x[k] for x in lat if x.get(k) is not None]
            if v:
                M[n] = f"{np.median(v):.1f}"
        M["TrainMinRGB"] = f"{np.mean([r['train_seconds']['rgb'] for r in R]) / 60:.0f}"
        M["TrainMinIR"] = f"{np.mean([r['train_seconds']['ir'] for r in R]) / 60:.0f}"
        M["CalPrior"] = f"{np.mean([r['calibration']['prior'] for r in R]):.2f}"
