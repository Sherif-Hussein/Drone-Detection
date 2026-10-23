"""Generate Supplementary Information tables (srep/si_tables.tex) from the result files."""
import glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); RUNS = os.path.join(DATA, "runs")
OUT = os.path.join(HERE, "..", "srep", "si_tables.tex")
an = json.load(open(os.path.join(DATA, "analysis_antiuav.json")))
L = []
f3 = lambda v: "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"

# S1: literature-standard metrics, both frames
cols = [("AP50", "AP\\textsubscript{50}"), ("AP75", "AP\\textsubscript{75}"), ("AP50_95", "AP\\textsubscript{50:95}"),
        ("AP50_S(<32^2)", "AP\\textsubscript{S}"), ("AP50_M(32^2-96^2)", "AP\\textsubscript{M}"), ("AP50_L(>96^2)", "AP\\textsubscript{L}"),
        ("P", "P"), ("R", "R"), ("F1", "F1"), ("AP10", "AP\\textsubscript{10}")]
L.append("\\begin{table}[h]\\centering\\scriptsize\\caption{\\textbf{Literature-standard metrics on the Anti-UAV300 test set} "
         "(mean over seeds). AP\\textsubscript{S/M/L}: AP\\textsubscript{50} for COCO small ($<32^2$), medium and large ($>96^2$ pixels) targets "
         "in the frame's own camera. P, R, F1 at the score threshold maximising F1. --: no targets in that size class.}\\label{tab:s1}")
L.append("\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{@{}l" + "c" * len(cols) + "@{}}\\toprule")
L.append("Method & " + " & ".join(c for _, c in cols) + " \\\\ \\midrule")
for fr in ("IR", "RGB"):
    L.append(f"\\multicolumn{{{len(cols) + 1}}}{{@{{}}l}}{{\\textit{{{fr} frame}}}} \\\\")
    for n, m in an["metrics"][fr].items():
        if n.startswith("stress"):
            continue
        L.append(n.replace("&", "\\&") + " & " + " & ".join(f3(m[k]["mean"]) if k in m else "--" for k, _ in cols) + " \\\\")
L.append("\\bottomrule\\end{tabular}\\end{table}")

# S2: anchored-fusion validation grid (seed 0)
gp = os.path.join(RUNS, "anchored_grid_s0.json")
if os.path.exists(gp):
    g = json.load(open(gp))
    L.append("\\begin{table}[h]\\centering\\scriptsize\\caption{\\textbf{Validation grid for anchored fusion} (seed 0, self-calibrated "
             "validation maps). Validation AP\\textsubscript{50} is saturated, so no setting can be preferred; defaults were fixed a priori "
             "(ProbEn, $\\tau=1$, $\\gamma=0$). Test values are shown for transparency only.}\\label{tab:s2}")
    L.append("\\begin{tabular}{@{}l l c c c c@{}}\\toprule Frame & Rule & $\\tau$ & $\\gamma$ & Val AP\\textsubscript{50} & Test AP\\textsubscript{50} \\\\ \\midrule")
    for fr in ("IR", "RGB"):
        L.append(f"{fr} & single camera & -- & -- & {g[f'val|{fr}|single']:.4f} & {g[f'test|{fr}|single']:.4f} \\\\")
        for rule in ("proben", "max", "wbf"):
            for tau in ("1.0", "2.0"):
                for gam in ("0.0", "1.0"):
                    k = f"{fr}|{rule}|{tau}|{gam}"
                    if f"val|{k}" in g:
                        L.append(f" & {rule} & {tau} & {gam} & {g['val|' + k]:.4f} & {g['test|' + k]:.4f} \\\\")
    L.append("\\bottomrule\\end{tabular}\\end{table}")

# S3: per-seed single-camera and key fusion results
L.append("\\begin{table}[h]\\centering\\scriptsize\\caption{\\textbf{Per-seed test AP\\textsubscript{50}} for the single cameras and the main fusion variants.}\\label{tab:s3}")
seeds = an["seeds"]
L.append("\\begin{tabular}{@{}l l " + "c" * len(seeds) + "@{}}\\toprule Frame & Method & " + " & ".join(f"Seed {s}" for s in seeds) + " \\\\ \\midrule")
for fr, names in (("IR", ["IR only", "Anchored ProbEn", "Late ProbEn (box avg.)", "Pixel-level, IR grid", "Hybrid (IR+RGB+pixel)"]),
                  ("RGB", ["RGB only", "Anchored ProbEn", "Cascade + RGB detector", "Pixel-level, RGB grid"])):
    for n in names:
        m = an["metrics"][fr].get(n)
        if m:
            ps = m["AP50"]["per_seed"]
            L.append(f"{fr} & {n} & " + " & ".join(f3(v) for v in ps) + " & " * (len(seeds) - len(ps)) + " \\\\")
L.append("\\bottomrule\\end{tabular}\\end{table}")

# S4: complementarity
cm = an["complementarity_own_frame"]
L.append("\\begin{table}[h]\\centering\\scriptsize\\caption{\\textbf{Mapping-free complementarity} (all seeds pooled): share of targets "
         "detected by each camera on its own image (calibrated score $\\ge$0.5, IoU $\\ge$0.5), for frames in which both cameras see the target "
         "unless stated.}\\label{tab:s4}")
L.append("\\begin{tabular}{@{}l r c c c c c c c@{}}\\toprule Subset & $n$ & Both & IR only & RGB only & Neither & IR & RGB & Either \\\\ \\midrule")
for k, v in cm.items():
    if v:
        L.append(f"{k.replace('>=', '$\\ge$').replace('<', '$<$')} & {v['n']} & " + " & ".join(f"{100 * v[x]:.1f}" for x in
                 ("both", "ir_only", "rgb_only", "neither", "ir", "rgb", "either")) + " \\\\")
L.append("\\bottomrule\\end{tabular}\\end{table}")

# S5: comparison with published results
L.append("\\begin{table}[h]\\centering\\scriptsize\\caption{\\textbf{Comparison with published Anti-UAV RGB--IR detectors.} Values as reported "
         "by the original authors; protocols differ (see notes), so the comparison is indicative.}\\label{tab:s5}")
L.append("\\begin{tabular}{@{}l l l c c l@{}}\\toprule Method & Venue & Fusion & AP\\textsubscript{50} & AP\\textsubscript{50:95} & Protocol notes \\\\ \\midrule")
ir = an["metrics"]["IR"]["IR only"]; rgb = an["metrics"]["RGB"]["RGB only"]; anc = an["metrics"]["IR"]["Anchored ProbEn"]
L += ["MBUDet\\cite{ye2026mbudet} & Information Fusion 2026 & feature & 0.901 & -- & MBU (every 40th frame, official split, IR reference) \\\\",
      "LER-YOLO\\cite{hou2026leryolo} & arXiv 2026 & feature & 0.897 & -- & MBU, 3 seeds, 320-px input \\\\",
      "\\quad its YOLOv5s IR-only baseline & & none & 0.812 & -- & as reported in ref.\\cite{hou2026leryolo}, 3 seeds, 320-px input \\\\",
      "\\quad its YOLOv5s RGB-only baseline & & none & 0.807 & -- & as above, RGB resampled to the IR reference \\\\",
      "DCAM-DETR\\cite{qin2026dcamdetr} & Information 2026 & feature & 0.947 & 0.783 & split not stated \\\\",
      "RTM-UAVDet\\cite{wang2025rtmuavdet} & IEEE TAES 2025 & feature & -- & 0.661 & validation split \\\\",
      f"Ours, IR only (YOLO11s) & -- & none & {ir['AP50']['mean']:.3f} & {ir['AP50_95']['mean']:.3f} & every 40th frame, official split, IR frame, 3 seeds \\\\",
      f"Ours, RGB only (YOLO11s) & -- & none & {rgb['AP50']['mean']:.3f} & {rgb['AP50_95']['mean']:.3f} & as above, RGB frame \\\\",
      f"Ours, anchored late fusion & -- & late & {anc['AP50']['mean']:.3f} & {anc['AP50_95']['mean']:.3f} & as above, IR frame \\\\",
      "\\bottomrule\\end{tabular}\\end{table}"]
open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("wrote", OUT, len(L), "lines")
