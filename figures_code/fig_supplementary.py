"""Supplementary figures: training curves, calibration reliability, mapping stress test and mapping error."""
import glob, io, json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from style import IR_C, VIS_C, FUS_C, GREY, GREY_D, INK, INK2, FULL_W, apply, panel, ygrid  # noqa: E402

apply()
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "data")
R = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(DATA, "runs", "results_antiuav_s*.json")))]
assert R, "no per-seed results found in " + os.path.join(DATA, "runs")
_rev = os.path.join(DATA, "analysis_antiuav_rev.json")
AN = json.load(open(_rev if os.path.exists(_rev) else os.path.join(DATA, "analysis_antiuav.json")))
HAL = os.path.join(DATA, "results_halmstad.json")


def curve(csv):
    df = pd.read_csv(io.StringIO(csv)); df.columns = [c.strip() for c in df.columns]
    return df["epoch"].values, df["metrics/mAP50(B)"].values


# ------------------------------------------------------------------ S1: training curves
fig, axs = plt.subplots(1, 3, figsize=(FULL_W, 2.2), gridspec_kw={"wspace": 0.35})
ax = axs[0]
names = {"ir": ("IR detector", IR_C), "rgb": ("RGB detector", VIS_C), "crop": ("Crop verifier", GREY_D), "pix": ("Pixel-level, IR grid", FUS_C),
         "pixrgb": ("Pixel-level, RGB grid", INK2)}
for r in R:
    for k, (lab, col) in names.items():
        c = r.get("training_curves", {}).get(k)
        if c:
            e, m = curve(c)
            ax.plot(e, m, color=col, lw=1.0, alpha=0.85, label=lab if r is R[0] else None)
ax.set_xlabel("Epoch"); ax.set_ylabel("Validation mAP$_{50}$"); ax.set_ylim(0, 1.0); ygrid(ax)
ax.legend(fontsize=5.0, loc="lower right", frameon=False, handlelength=1.2)
panel(ax, "a", "Anti-UAV300 (all seeds)")
if os.path.exists(HAL):
    runs = json.load(open(HAL))["halmstad"]
    for i, (s, col) in enumerate((("IR", IR_C), ("V", VIS_C))):
        ax = axs[1 + i]
        for r in runs.values():
            if r["sensor"] != s or r["imgsz"] != 320 or "results_csv" not in r:
                continue
            e, m = curve(r["results_csv"])
            ax.plot(e, m, color=col if r["protocol"] == "clip" else GREY, lw=1.0, ls="-" if r["protocol"] == "clip" else "--",
                    label=("clip-level split" if r["protocol"] == "clip" else "random split") if r["seed"] == 0 else None)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Validation mAP$_{50}$"); ax.set_ylim(0, 1.0); ygrid(ax)
        ax.legend(fontsize=5.6, loc="lower right")
        panel(ax, "b" if s == "IR" else "c", f"Halmstad, {'thermal' if s == 'IR' else 'visible'}")
fig.savefig(os.path.join(HERE, "figS1_training.pdf"), bbox_inches="tight", pad_inches=0.03)
fig.savefig(os.path.join(HERE, "figS1_training.png"), dpi=200, bbox_inches="tight", pad_inches=0.03)

# ------------------------------------------------------------------ S2: calibration, mapping stress, mapping error
fig, axs = plt.subplots(2, 2, figsize=(FULL_W, 4.8), gridspec_kw={"wspace": 0.38, "hspace": 0.55})
axs = axs.ravel()
ax = axs[0]
for s, col in (("IR", IR_C), ("RGB", VIS_C)):
    cal = AN["calibration"].get(s)
    if not cal:
        continue
    for kind, ls, mk in (("reliability_raw", ":", "o"), ("reliability_platt", "-", "s")):
        rel = [b for b in cal[0][kind] if b["n"] >= 20]
        ax.plot([b["conf"] for b in rel], [b["acc"] for b in rel], ls=ls, marker=mk, ms=2.5, color=col, lw=1.0,
                label=f"{s}, {'raw' if 'raw' in kind else 'Platt'} (ECE {np.mean([c['ece_raw' if 'raw' in kind else 'ece_platt'] for c in cal]):.3f})")
ax.plot([0, 1], [0, 1], color=INK, lw=0.6)
ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed precision"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(fontsize=5.6, loc="lower right"); ygrid(ax); panel(ax, "a", "Score calibration (test, 10 equal-width bins, n ≥ 20)")
ax = axs[1]
st = AN["metrics"]["IR"]
for kind, col, lab in (("anchored", FUS_C, "Anchored ProbEn"), ("box-avg", GREY_D, "Late ProbEn, box avg.")):
    d = [0, 10, 20, 40, 80]
    m = [st.get(f"stress {kind} {x}px", {}).get("AP50", {}).get("mean", np.nan) for x in d]
    s = [st.get(f"stress {kind} {x}px", {}).get("AP50", {}).get("sd", 0) for x in d]
    ax.errorbar(d, m, yerr=s, color=col, marker="o", ms=3, lw=1.2, capsize=1.5, label=lab)
ax.axhline(st["IR only"]["AP50"]["mean"], color=IR_C, lw=0.8, ls="--", label="IR only")
ax.set_xlabel("Added mapping offset (RGB px)"); ax.set_ylabel("AP$_{50}$"); ygrid(ax); ax.legend(fontsize=5.6)
panel(ax, "b", "Mapping stress test")
ax = axs[2]
errs = {k: [] for k in ("self", "global", "oracle")}
for r in R:
    mt = r.get("mapping_test", {})
    for k in errs:
        errs[k] += list(mt.get(f"err_px_{k}", {}).values())
data = [np.array(errs[k]) for k in ("global", "self", "oracle") if errs[k]]
if data:
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False, medianprops=dict(color=INK))
    for p_, c in zip(bp["boxes"], (GREY, FUS_C, GREY_D)):
        p_.set_facecolor(c); p_.set_alpha(0.6); p_.set_edgecolor(INK2)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["Global", "Self-cal.", "GT-fitted"])
ax.set_yscale("log"); ax.set_ylabel("Median centre error per sequence (RGB px)"); ygrid(ax)
panel(ax, "c", "Cross-camera mapping error")
# d: share of frames that need exact registration (the rest keep the windowed map)
ax = axs[3]
RC2 = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(DATA, "runs", "registration_curve2_s*.json")))]
if RC2:
    for key, col, lab in (("box-avg ProbEn", GREY_D, "Late ProbEn, box avg."), ("anchored ProbEn + unmatched RGB", FUS_C, "Anchored + unmatched RGB")):
        xs = np.array([c["x"] for c in RC2[0]["curves"]["coverage"]]) * 100
        ys = np.array([[c[key] for c in r["curves"]["coverage"]] for r in RC2])
        ax.plot(xs, ys.mean(0), color=col, marker="o", ms=2.6, lw=1.2, label=lab)
        ax.fill_between(xs, ys.min(0), ys.max(0), color=col, alpha=0.2, lw=0)
    ax.axhline(0, color=INK, lw=0.7)
ax.set_xlabel("Frames with exact registration (%)"); ax.set_ylabel("ΔAP$_{50}$ vs IR only"); ygrid(ax)
ax.legend(fontsize=5.6, loc="upper left")
panel(ax, "d", "Partial exact registration")
fig.savefig(os.path.join(HERE, "figS2_checks.pdf"), bbox_inches="tight", pad_inches=0.03)
fig.savefig(os.path.join(HERE, "figS2_checks.png"), dpi=200, bbox_inches="tight", pad_inches=0.03)

# ------------------------------------------------------------------ S3: target size and degraded or missing cameras (seed 0)
from matplotlib.colors import LinearSegmentedColormap
from style import SEQ
MI = AN["metrics"]["IR"]
fig = plt.figure(figsize=(FULL_W, 4.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1], hspace=0.75, wspace=0.42)
ax = fig.add_subplot(gs[0, 0])
cols = [("AP50_<24", "<24"), ("AP50_24-32", "24–32"), ("AP50_32-48", "32–48"), ("AP50_>=48", "≥48")]
for n, col, mk, lab in (("IR only", IR_C, "o", "IR only"), ("Pixel-level, IR grid", GREY_D, "v", "Pixel-level, IR grid"),
                        ("Anchored ProbEn", FUS_C, "D", "Anchored ProbEn")):
    if n in MI:
        ax.plot(range(4), [MI[n][c]["mean"] for c, _ in cols], color=col, marker=mk, ms=3.2, lw=1.3, label=lab)
ax.set_xticks(range(4)); ax.set_xticklabels([l for _, l in cols]); ax.set_xlabel("IR target size (px)")
ax.set_ylim(0.6, 1.0); ax.set_ylabel("AP$_{50}$ (IR frame)"); ygrid(ax); ax.legend(fontsize=5.8, loc="lower right")
panel(ax, "a", "By target size (axis from 0.6)")
ax = fig.add_subplot(gs[0, 1])
cn = AN.get("complementarity_null")
if cn:
    th = cn["ir_vs_rgb_by_threshold"]
    ts = sorted(next(iter(th.values())), key=float)
    for f_, col, mk, lab in (("one_only", INK2, "o", "Exactly one detector finds it"), ("gain_over_better", FUS_C, "D", "Ideal gain over better detector")):
        v = np.array([[100 * th[s_][t][f_] for t in ts] for s_ in th])
        ax.plot([float(t) for t in ts], v.mean(0), color=col, marker=mk, ms=3, lw=1.3, label=lab)
        ax.fill_between([float(t) for t in ts], v.min(0), v.max(0), color=col, alpha=0.18, lw=0)
    nl = cn["same_modality_seed_pairs"]
    ir_null = 100 * np.mean([x["one_only"] for k, x in nl.items() if k.startswith("IR ")])
    ax.plot([0.5], [ir_null], "s", color=IR_C, ms=4.5, mec=INK, mew=0.4, label="Null: two IR detectors (score 0.5)")
ax.set_xlabel("Hit threshold on calibrated score"); ax.set_ylabel("Frames (%)"); ax.set_ylim(0, 20); ygrid(ax)
ax.legend(fontsize=5.6, loc="upper left")
panel(ax, "b", "Complementarity vs threshold")
rb = AN.get("robustness_AP50", {})
conds = [("clean", "Clean"), ("rgb_dark", "RGB dark"), ("rgb_fog", "RGB fog"), ("rgb_blur", "RGB blur"), ("rgb_missing", "RGB missing"),
         ("ir_noise", "IR noise"), ("ir_lowcontrast", "IR low contrast"), ("ir_missing", "IR missing")]
cmap = LinearSegmentedColormap.from_list("seq", ["#f7f7f5"] + SEQ)
sub = gs[1, :].subgridspec(1, 2, wspace=0.34, width_ratios=[6, 5])
for si, (fr, meths) in enumerate((("IR frame", [("IR only", "IR only"), ("RGB only (mapped)", "RGB, mapped"), ("Late OR (box avg.)", "Late OR, avg."),
                                                ("Late ProbEn (box avg.)", "ProbEn, avg."), ("Anchored ProbEn", "Anchored ProbEn"),
                                                ("Pixel-level, IR grid", "Pixel, IR grid")]),
                                   ("RGB frame", [("RGB only", "RGB only"), ("IR only (mapped)", "IR, mapped"), ("Anchored ProbEn", "Anchored ProbEn"),
                                                  ("Cascade + RGB detector", "Cascade + RGB"), ("Pixel-level, RGB grid", "Pixel, RGB grid")]))):
    ax = fig.add_subplot(sub[si])
    Mx = np.full((len(meths), len(conds)), np.nan)
    for i, (n, _) in enumerate(meths):
        for j, (c, _) in enumerate(conds):
            v = rb.get(c, {}).get(fr, {}).get(n)
            if v is not None:
                Mx[i, j] = v
    ax.imshow(Mx, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(Mx.shape[0]):
        for j in range(Mx.shape[1]):
            if not np.isnan(Mx[i, j]):
                ax.text(j, i, f"{Mx[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if Mx[i, j] > 0.62 else INK)
    ax.set_xticks(range(len(conds))); ax.set_xticklabels([n for _, n in conds], rotation=35, ha="right", fontsize=6)
    ax.set_yticks(range(len(meths))); ax.set_yticklabels([n for _, n in meths], fontsize=6.2)
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.axvline(0.5, color="white", lw=2.5); ax.axvline(4.5, color="white", lw=2.5)
    panel(ax, "c" if si == 0 else "d", f"{fr}: degraded or missing camera")
fig.savefig(os.path.join(HERE, "figS3_size_robustness.pdf"), bbox_inches="tight", pad_inches=0.03)
fig.savefig(os.path.join(HERE, "figS3_size_robustness.png"), dpi=200, bbox_inches="tight", pad_inches=0.03)
print("saved S1, S2, S3")
