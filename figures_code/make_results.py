"""Build Fig. 4 (Halmstad), Fig. 5 (Anti-UAV300 fusion), Table 3 and numbers.tex from result JSONs.

Sources, in order of preference:
  ../data/results_halmstad.json   (Kaggle GPU run: yolo11s, 40 epochs, 3 seeds)
  /home/claude/work/results_halm/*.json  (local CPU run: yolo11n, 12 epochs, seed 0)
  ../data/results_antiuav.json    (Kaggle GPU run)
Anything missing is drawn as a 'pending' panel and written as \\pend{--} so the manuscript always compiles.
Re-run this script after dropping new JSONs into ../data/ and recompile the paper.
"""
import glob, io, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
TEX = os.path.join(HERE, "..", "tex")
LOCAL = "/home/claude/work/results_halm"
IR_C, VIS_C, FUS_C, GREY, INK, INK2, GRID = "#eb6834", "#2a78d6", "#1baf7a", "#b9b8b3", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": "#8a8984", "axes.linewidth": 0.6,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False})
M = {}  # macro name -> text


def fmt(x, nd=3):
    return "\\pend{--}" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def msd(vals, nd=3):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not v:
        return "\\pend{--}", None
    if len(v) == 1:
        return f"{v[0]:.{nd}f}", v[0]
    return f"{np.mean(v):.{nd}f}$\\pm${np.std(v, ddof=1):.{nd}f}", float(np.mean(v))


def title(ax, t):
    ax.set_title(t, loc="left", fontsize=8.5, fontweight="bold")


def pending(ax, t, msg="Pending GPU run (Kaggle notebook)"):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linestyle((0, (3, 3))); s.set_edgecolor(GREY)
    ax.text(0.5, 0.5, msg, ha="center", va="center", color=INK2, fontsize=8, transform=ax.transAxes)
    title(ax, t)


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True); ax.tick_params(width=0.6, length=3)


# =============================================================== Halmstad
runs, source = {}, None
kp = os.path.join(DATA, "results_halmstad.json")
if os.path.exists(kp):
    runs = json.load(open(kp))["halmstad"]; source = "kaggle"
else:
    for f in sorted(glob.glob(f"{LOCAL}/*.json")):
        r = json.load(open(f)); runs[r["name"]] = r
    source = "local" if runs else None


def pick(sensor, proto, imgsz=320):
    return [r for r in runs.values() if r["sensor"] == sensor and r["protocol"] == proto and r["imgsz"] == imgsz]


any_r = next(iter(runs.values()), None)
M["HalmModel"] = (any_r["model"].replace(".pt", "").upper().replace("YOLO11", "YOLO11-") if any_r else "\\pend{--}")
M["HalmEpochs"] = str(any_r["epochs"]) if any_r else "\\pend{--}"
M["HalmSeeds"] = str(len({r["seed"] for r in runs.values()})) if runs else "\\pend{--}"
M["HalmSource"] = {"kaggle": "GPU", "local": "CPU", None: "\\pend{--}"}[source]
pm = (json.load(open(kp))["prep_meta"] if source == "kaggle" else
      (json.load(open("/home/claude/work/halm_s30/prep_meta.json")) if os.path.exists("/home/claude/work/halm_s30/prep_meta.json") else None))
M["HalmStep"] = str(pm["step"]) if pm else "\\pend{--}"
for s_ in ("IR", "V"):
    for sp in ("train", "val", "test"):
        M[f"N{s_}{sp}"] = f'{pm["sensors"][s_]["clip_" + sp]:,}' if pm else "\\pend{--}"
        M[f"N{s_}clips{sp}"] = str(pm["sensors"][s_]["clips_by_split"][sp]) if pm else "\\pend{--}"
key = {"IR": "IR", "V": "V", "clip": "clip", "random": "rand"}
H = {}
for s in ("IR", "V"):
    for p in ("clip", "random"):
        rr = pick(s, p)
        txt, mu = msd([r["test"]["map50"] for r in rr]); M[f"H{s}{key[p]}AP"] = txt; H[(s, p, "ap50")] = mu
        txt, mu = msd([r["test"]["map50_95"] for r in rr]); M[f"H{s}{key[p]}APs"] = txt; H[(s, p, "ap5095")] = mu
        if p == "clip":
            for d in ("close", "medium", "distant"):
                txt, mu = msd([r["test_by_distance"][d]["map50"] for r in rr]); M[f"H{s}{d}"] = txt; H[(s, d)] = mu
            for c in ("drone", "bird", "airplane", "helicopter"):
                txt, mu = msd([(r["test"]["per_class"].get(c) or {}).get("ap50") for r in rr]); M[f"H{s}{c}"] = txt; H[(s, c)] = mu
            M[f"H{s}lat"] = fmt(np.mean([r["latency_ms_per_frame"] for r in rr]) if rr else None, 1)
    a, b = H[(s, "random", "ap50")], H[(s, "clip", "ap50")]
    M[f"H{s}gap"] = fmt(None if a is None or b is None else a - b, 3)
    M[f"H{s}gapRel"] = fmt(None if a is None or b is None else 100 * (a - b) / b, 0)
    a, b = H[(s, "random", "ap5095")], H[(s, "clip", "ap5095")]
    M[f"H{s}gapS"] = fmt(None if a is None or b is None else a - b, 3)
for s, sz in (("V", 640), ("IR", 640)):
    txt, _ = msd([r["test"]["map50"] for r in pick(s, "clip", sz)]); M[f"H{s}clipAPhi"] = txt

fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.4))
classes = ["drone", "bird", "airplane", "helicopter"]
# (a) per class, clip split
a = ax[0, 0]
if pick("IR", "clip") or pick("V", "clip"):
    x = np.arange(4); w = 0.38
    for j, (s, col, lab) in enumerate((("IR", IR_C, "Thermal IR"), ("V", VIS_C, "Visible"))):
        vals = [[(r["test"]["per_class"].get(c) or {}).get("ap50", np.nan) for r in pick(s, "clip")] for c in classes]
        mu = [np.nanmean(v) if v else np.nan for v in vals]
        sd = [np.nanstd(v, ddof=1) if len(v) > 1 else 0 for v in vals]
        a.bar(x + (j - 0.5) * w, mu, w - 0.04, yerr=sd, color=col, label=lab, error_kw=dict(lw=0.7, capsize=2, ecolor=INK2))
    a.set_xticks(x); a.set_xticklabels([c.title() for c in classes]); a.set_ylim(0, 1); a.set_ylabel("Test AP$_{50}$")
    a.legend(fontsize=7, loc="upper right"); style(a); title(a, "a  Per-class AP$_{50}$, clip-level split")
else:
    pending(a, "a  Per-class AP$_{50}$, clip-level split", "Pending local CPU run")
# (b) by distance
b = ax[0, 1]
if pick("IR", "clip") or pick("V", "clip"):
    dd = ["close", "medium", "distant"]
    for s, col, lab in (("IR", IR_C, "Thermal IR"), ("V", VIS_C, "Visible")):
        rr = pick(s, "clip")
        if not rr:
            continue
        mu = [np.mean([r["test_by_distance"][d]["map50"] for r in rr]) for d in dd]
        b.plot(dd, mu, color=col, lw=2, marker="o", ms=4, label=lab)
        b.annotate(f"{mu[-1]:.2f}", (2, mu[-1]), xytext=(5, 0), textcoords="offset points", va="center", fontsize=7, color=INK2)
    b.set_ylim(0, 1); b.set_ylabel("Test mAP$_{50}$"); b.set_xlabel("Distance bin (dataset annotation)")
    b.legend(fontsize=7, loc="lower left"); style(b); title(b, "b  mAP$_{50}$ by distance bin")
else:
    pending(b, "b  mAP$_{50}$ by distance bin", "Pending local CPU run")
# (c) leakage: clip vs random
c = ax[1, 0]
if any(H[(s, p, "ap50")] is not None for s in ("IR", "V") for p in ("clip", "random")):
    cats = [("IR", "ap50", "IR\nAP$_{50}$"), ("IR", "ap5095", "IR\nAP$_{50:95}$"), ("V", "ap50", "Visible\nAP$_{50}$"), ("V", "ap5095", "Visible\nAP$_{50:95}$")]
    x = np.arange(len(cats)); w = 0.38
    for j, (p, col, lab) in enumerate((("random", GREY, "Random-frame split (leaky)"), ("clip", INK, "Clip-level split"))):
        vals = [H[(s, p, m)] if H[(s, p, m)] is not None else np.nan for s, m, _ in cats]
        c.bar(x + (j - 0.5) * w, vals, w - 0.04, color=col, label=lab)
        for xi, v in zip(x, vals):
            if not np.isnan(v):
                c.text(xi + (j - 0.5) * w, v + 0.015, f"{v:.2f}", ha="center", fontsize=6.5, color=INK2)
    c.set_xticks(x); c.set_xticklabels([t for *_, t in cats]); c.set_ylim(0, 1.08); c.set_ylabel("Test score")
    c.legend(fontsize=7, loc="upper right"); style(c); title(c, "c  Clip-level vs random-frame split")
else:
    pending(c, "c  Clip-level vs random-frame split", "Pending local CPU run")
# (d) validation curves
d = ax[1, 1]
curves = 0
for s, col in (("IR", IR_C), ("V", VIS_C)):
    for p, ls in (("clip", "-"), ("random", "--")):
        rr = [r for r in pick(s, p) if r.get("results_csv") and r["seed"] == 0]
        if not rr:
            continue
        df = pd.read_csv(io.StringIO(rr[0]["results_csv"])); df.columns = [k.strip() for k in df.columns]
        d.plot(df["epoch"], df["metrics/mAP50(B)"], color=col, ls=ls, lw=1.6,
               label=f"{'IR' if s == 'IR' else 'Visible'}, {'clip' if p == 'clip' else 'random'}")
        curves += 1
if curves:
    d.set_xlabel("Epoch"); d.set_ylabel("Validation mAP$_{50}$"); d.set_ylim(0, 1); d.legend(fontsize=7, loc="lower right", ncol=1)
    style(d); title(d, "d  Validation curves (seed 0)")
else:
    pending(d, "d  Validation curves (seed 0)", "Pending local CPU run")
fig.tight_layout(); fig.savefig(f"{HERE}/fig4_halmstad.pdf"); fig.savefig(f"{HERE}/fig4_halmstad.png", dpi=200); plt.close(fig)

# =============================================================== Anti-UAV300
A = None
ap = os.path.join(DATA, "results_antiuav.json")
if os.path.exists(ap):
    A = json.load(open(ap))
seeds = sorted([k for k in (A or {}) if k.startswith("seed")])
REF = "IR"  # report in the IR frame (both frames are in the JSON)
METHODS = [("IR-only", ("single", REF, "IR-only")), ("RGB-only", ("single", REF, "RGB-only")),
           ("OR", ("late", REF, "or")), ("AND", ("late", REF, "and")), ("MAX", ("late", REF, "max")),
           ("WBF", ("late", REF, "wbf")), ("ProbEn", ("late", REF, "proben")), ("Learned", ("late", REF, "learned")),
           ("ProbEn+temporal", ("late_temporal", REF, "proben")), ("Cascade", ("cascade", "IR", "cascade")),
           ("Pixel", ("pixel", "IR", "pixel_fusion")), ("Hybrid", ("hybrid", "IR", "ir+rgb+pixel_proben"))]


def get(d, path):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


def collect(path, metric):
    out = []
    for s in seeds:
        blk = get(A[s], path)
        v = get(blk, metric) if blk else None
        out.append(v)
    return [v for v in out if v is not None]


macro_name = {"IR-only": "IRonly", "RGB-only": "RGBonly", "OR": "Or", "AND": "And", "MAX": "Max", "WBF": "Wbf",
              "ProbEn": "Proben", "Learned": "Learned", "ProbEn+temporal": "ProbenT", "Cascade": "Cascade",
              "Pixel": "Pixel", "Hybrid": "Hybrid"}
rows = []
for lab, path in METHODS:
    mn = macro_name[lab]
    r = {"lab": lab}
    for met, mk, nd in (("ap50", "AP", 3), ("ap50_95", "APs", 3), ("recall_at_fppi_0.01", "Rlo", 3), ("recall_at_fppi_0.1", "Rhi", 3)):
        txt, mu = msd(collect(path, (met,)) if A else [], nd); M[f"A{mn}{mk}"] = txt; r[met] = (txt, mu)
    for sb, sk in (("<16px", "Ss"), ("16-32px", "Sm"), (">=32px", "Sl")):
        txt, mu = msd(collect(path, ("ap50_by_size", sb)) if A else []); M[f"A{mn}{sk}"] = txt; r[sb] = (txt, mu)
    for lb, lk in (("night(<50)", "Ln"), ("dim(50-100)", "Ld"), ("day(>=100)", "Ly")):
        txt, mu = msd(collect(path, ("ap50_by_light", lb)) if A else []); M[f"A{mn}{lk}"] = txt; r[lb] = (txt, mu)
    rows.append(r)
# bootstrap
for k, mk in (("proben_minus_IR", "BootPI"), ("proben_minus_RGB", "BootPR"), ("learned_minus_IR", "BootLI")):
    b = get(A, (seeds[0], "bootstrap", REF, k)) if A and seeds else None
    M[mk] = (f"{b['mean']:+.3f} [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}]" if isinstance(b, dict) and "ci95" in b else "\\pend{--}")
mp = get(A, ("mapping", "global")) if A else None
M["MapScale"] = f"{mp['ax']:.2f}/{mp['ay']:.2f}" if mp else "\\pend{--}"
mt = get(A, (seeds[0], "mapping_test")) if A and seeds else None
M["MapSelf"] = fmt(100 * mt["self_calibrated_share"], 0) if mt else "\\pend{--}"
M["NSeqTest"] = str(get(A, ("discovery", "by_split", "test"))) if A and get(A, ("discovery", "by_split", "test")) else "\\pend{--}"
M["NSeqTrain"] = str(get(A, ("discovery", "by_split", "train"))) if A and get(A, ("discovery", "by_split", "train")) else "\\pend{--}"
lat = get(A, (seeds[0], "latency_ms")) if A and seeds else None
for k, mk in (("ir_detector", "LatIR"), ("rgb_detector", "LatRGB"), ("crop_verifier_per_candidate", "LatCas"), ("pixel_fusion_detector", "LatPix")):
    M[mk] = fmt(lat.get(k) if isinstance(lat, dict) else None, 1)
M["LatDev"] = (lat.get("device", "") if isinstance(lat, dict) else "\\pend{--}")

# Table 3
tab_rows = io.StringIO()
with tab_rows as f:
    groups = {"IR-only": "Single", "OR": "Late", "Cascade": "Cascade", "Pixel": "Pixel", "Hybrid": "Hybrid"}
    for r in rows:
        if r["lab"] in groups and r["lab"] != "IR-only":
            f.write("\\midrule\n")
        f.write(" & ".join([r["lab"], r["ap50"][0], r["ap50_95"][0], r["<16px"][0], r["16-32px"][0], r[">=32px"][0],
                            r["night(<50)"][0], r["recall_at_fppi_0.01"][0]]) + " \\\\\n")
    M["TabMainRows"] = f.getvalue()

# Fig 5
fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.6))
cols = {"IR-only": IR_C, "RGB-only": VIS_C}
if A and seeds:
    a = ax[0, 0]
    labs = [r["lab"] for r in rows]
    mu = [r["ap50"][1] if r["ap50"][1] is not None else np.nan for r in rows]
    sd = [np.std(collect(p, ("ap50",)), ddof=1) if len(collect(p, ("ap50",))) > 1 else 0 for _, p in METHODS]
    best_single = np.nanmax(mu[:2])
    cc = [cols.get(l, FUS_C if m > best_single else GREY) for l, m in zip(labs, mu)]
    y = np.arange(len(labs))[::-1]
    a.barh(y, mu, 0.7, xerr=sd, color=cc, error_kw=dict(lw=0.7, capsize=2, ecolor=INK2))
    a.axvline(best_single, color=INK, lw=0.7, ls=":")
    for yi, m in zip(y, mu):
        a.text(m + 0.01, yi, f"{m:.3f}", va="center", fontsize=6.5, color=INK2)
    a.set_yticks(y); a.set_yticklabels(labs, fontsize=7); a.set_xlabel("Test AP$_{50}$ (IR frame)"); a.set_xlim(0, 1.08)
    a.grid(axis="x", color=GRID, lw=0.5); a.set_axisbelow(True); title(a, "a  Fusion level and rule")
    # b size
    b = ax[0, 1]; sb = ["<16px", "16-32px", ">=32px"]; show = ["IR-only", "RGB-only", "ProbEn", "Learned", "Cascade"]
    pal = {"IR-only": IR_C, "RGB-only": VIS_C, "ProbEn": FUS_C, "Learned": "#0e6e4c", "Cascade": INK2}
    w = 0.16; x = np.arange(3)
    for j, l in enumerate(show):
        r = next(rr for rr in rows if rr["lab"] == l)
        b.bar(x + (j - 2) * w, [r[s][1] if r[s][1] is not None else np.nan for s in sb], w - 0.02, color=pal[l], label=l)
    b.set_xticks(x); b.set_xticklabels(["<16 px", "16–32 px", "≥32 px"]); b.set_ylim(0, 1); b.set_ylabel("AP$_{50}$")
    b.set_xlabel("Target size in IR frame"); style(b)
    hb, lb_ = b.get_legend_handles_labels()
    title(b, "b  AP$_{50}$ by target size")
    # c light
    c = ax[1, 0]; lb = ["night(<50)", "dim(50-100)", "day(>=100)"]
    for j, l in enumerate(show):
        r = next(rr for rr in rows if rr["lab"] == l)
        c.bar(x + (j - 2) * w, [r[s][1] if r[s][1] is not None else np.nan for s in lb], w - 0.02, color=pal[l], label=l)
    c.set_xticks(x); c.set_xticklabels(["Night", "Dim", "Day"]); c.set_ylim(0, 1); c.set_ylabel("AP$_{50}$")
    c.set_xlabel("RGB scene brightness"); style(c); title(c, "c  AP$_{50}$ by illumination")
    # d robustness
    d = ax[1, 1]
    conds = [k for k in get(A[seeds[0]], ("robustness_ap50_IRref",)) or {}]
    if conds:
        meths = ["IR-only", "RGB-only", "max", "proben", "learned"]
        lab_m = {"max": "MAX", "proben": "ProbEn", "learned": "Learned"}
        pal2 = {"IR-only": IR_C, "RGB-only": VIS_C, "max": GREY, "proben": FUS_C, "learned": "#0e6e4c"}
        xx = np.arange(len(conds))
        for m in meths:
            v = [np.nanmean([(lambda z: np.nan if z is None else z)(get(A[s], ("robustness_ap50_IRref", cd, m))) for s in seeds]) for cd in conds]
            d.plot(xx, v, color=pal2[m], marker="o", ms=3.5, lw=1.4, label=lab_m.get(m, m))
        nice = {"rgb_dark": "RGB\ndark", "rgb_fog": "RGB\nfog", "rgb_blur": "RGB\nblur", "ir_noise": "IR\nnoise", "ir_lowcontrast": "IR low\ncontrast", "rgb_missing": "RGB\nmissing", "ir_missing": "IR\nmissing"}
        d.set_xticks(xx); d.set_xticklabels([nice.get(k, k) for k in conds], fontsize=6)
        d.set_ylim(0, 1); d.set_ylabel("AP$_{50}$"); d.legend(fontsize=6.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0)); d.set_ylim(0, 1.22); style(d)
        title(d, "d  Degraded or missing sensor")
    else:
        pending(d, "d  Degraded or missing sensor")
else:
    for a_, t in zip(ax.flat, ["a  Fusion level and rule", "b  AP$_{50}$ by target size",
                               "c  AP$_{50}$ by illumination", "d  Degraded or missing sensor"]):
        pending(a_, t)
if A and seeds:
    fig.legend(hb, lb_, loc="lower center", ncol=5, fontsize=7, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=[0, 0.04, 1, 1]); fig.savefig(f"{HERE}/fig5_antiuav.pdf"); fig.savefig(f"{HERE}/fig5_antiuav.png", dpi=200); plt.close(fig)

with open(f"{TEX}/numbers.tex", "w") as f:
    f.write("% generated by figs/make_results.py -- do not edit\n")
    for k, v in sorted(M.items()):
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print("halmstad source:", source, "| runs:", list(runs)); print("antiuav:", bool(A), seeds)
print({k: v for k, v in M.items() if k.startswith("H")})
