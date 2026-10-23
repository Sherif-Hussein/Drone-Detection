"""Fig. 5: where the cameras complement each other and what each fusion level makes of it (Anti-UAV300).
Panels: a complementarity, b fusion gain (session-cluster bootstrap), c localisation tolerance, d registration precision.
Size and robustness panels are in Supplementary Fig. S3 (fig_supplementary.py)."""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from style import IR_C, VIS_C, FUS_C, GREY, GREY_D, INK, INK2, FULL_W, apply, panel, ygrid  # noqa: E402

apply()
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "data")
AN = json.load(open(os.path.join(DATA, "analysis_antiuav_rev.json")))
CV = json.load(open(os.path.join(DATA, "runs", "loso_cv.json")))
PERM = json.load(open(os.path.join(DATA, "runs", "session_permutation.json")))["families"]
MI = AN["metrics"]["IR"]
LIGHT_C = "#dcdbd6"
MARGIN = 0.01

fig = plt.figure(figsize=(FULL_W, 5.9))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.62, wspace=0.62, width_ratios=[1.1, 1.0])

# a  complementarity (each detector against its own camera's ground truth; frames with the target visible in both)
ax = fig.add_subplot(gs[0, 0])
C = AN["complementarity_own_frame"]
groups = [("All", "all"), ("<24", "size <24"), ("24–32", "size 24-32"), ("32–48", "size 32-48"), ("≥48", "size >=48"),
          ("Night", "light night"), ("Dim", "light dim"), ("Day", "light day")]
groups = [(n, k) for n, k in groups if C.get(k)]
cats = [("both", "Both detect", INK2), ("ir_only", "IR only", IR_C), ("rgb_only", "RGB only", VIS_C), ("neither", "Neither", LIGHT_C)]
x = np.arange(len(groups)); bottom = np.zeros(len(groups))
for f_, lab, col in cats:
    v = np.array([C[k][f_] * 100 for _, k in groups])
    ax.bar(x, v, 0.64, bottom=bottom, color=col, label=lab, edgecolor="white", lw=0.6)
    bottom += v
for i, (_, k) in enumerate(groups):
    gain = 100 * (C[k]["either"] - max(C[k]["ir"], C[k]["rgb"]))
    ax.text(i, 101.5, f"+{gain:.1f}", ha="center", va="bottom", fontsize=6, color=FUS_C, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([n for n, _ in groups], fontsize=5.6)
for b_ in (0.5, 4.5):
    ax.axvline(b_, color="white", lw=3)
ax.text(2.5, -0.14, "IR target size (px)", ha="center", fontsize=6.5, color=INK2, transform=ax.get_xaxis_transform())
ax.text(6.0, -0.14, "Illumination", ha="center", fontsize=6.5, color=INK2, transform=ax.get_xaxis_transform())
ax.set_ylim(0, 112); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_ylabel("Targets (%)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.21), ncol=4, fontsize=6.2, handlelength=1.0, frameon=False)
panel(ax, "a", "Who finds the target")

# b  forest plot: difference to the single camera of the frame, session-cluster bootstrap, Holm-adjusted markers
ax = fig.add_subplot(gs[0, 1])
B = AN["cluster_bootstrap_AP50"]
CB = CV["cluster_bootstrap_AP50"]
rows = [(B, "IR", "Anchored ProbEn - IR only", "Anchored ProbEn"),
        (B, "IR", "Anchored MAX - IR only", "Anchored MAX"),
        (CB, None, "Anchored, CV-tuned - IR only", "Anchored, session-CV"),
        (B, "IR", "Anchored condition-aware - IR only", "Anchored cond.-aware"),
        (B, "IR", "Anchored hybrid (IR+RGB+pixel) - IR only", "Anchored hybrid"),
        (B, "IR", "Pixel-level, IR grid - IR only", "Pixel-level, IR grid"),
        (CB, None, "IR + IR (other seed), anchored ProbEn - IR only", "Null: IR + 2nd IR"),
        (B, "IR", "Late OR (box avg.) - IR only", "Late OR, box avg."),
        (B, "IR", "Hybrid (IR+RGB+pixel) - IR only", "Hybrid, box avg."),
        (B, "IR", "Late ProbEn (box avg.) - IR only", "Late ProbEn, box avg."),
        (B, "RGB", "Cascade + RGB detector - RGB only", "Cascade + RGB det."),
        (B, "RGB", "Anchored ProbEn - RGB only", "Anchored ProbEn"),
        (B, "RGB", "Pixel-level, RGB grid - RGB only", "Pixel-level, RGB grid¹"),
        (B, "RGB", "Late ProbEn (box avg.) - RGB only", "Late ProbEn, box avg.")]
rows = [r for r in rows if r[2] in (r[0][r[1]] if r[1] else r[0])]
XL = (-0.06, 0.03)
y = np.arange(len(rows))[::-1].astype(float)
is_rgb = np.array([r[1] == "RGB" for r in rows])
y[is_rgb] -= 0.8
ax.axvspan(-MARGIN, MARGIN, color=GREY, alpha=0.18, lw=0)
for yi, (src, fr, k, lab) in zip(y, rows):
    b = (src[fr] if fr else src)[k]; lo, hi = b["ci95"]; d = b["diff"]
    null = lab.startswith("Null")
    # colour = significance in the exact session permutation test (Holm within the image frame), not the bootstrap
    ph = PERM.get(fr, {}).get(k, {}).get("p_perm_holm", 1.0) if fr else 1.0
    col = INK2 if null else ((FUS_C if d > 0 else IR_C) if ph < 0.05 else GREY_D)
    if d < XL[0]:
        ax.annotate(f"{d:+.2f}", xy=(XL[0], yi), xytext=(XL[0] + 0.02, yi), fontsize=6, color=col, va="center",
                    arrowprops=dict(arrowstyle="->", color=col, lw=0.8))
        continue
    ax.plot([max(lo, XL[0]), min(hi, XL[1])], [yi, yi], color=col, lw=1.4, solid_capstyle="round")
    ax.plot(d, yi, "D" if null else "o", color=col, ms=3.4, mec="white", mew=0.4)
ax.axvline(0, color=INK, lw=0.7); ax.set_xlim(*XL); ax.set_ylim(y.min() - 0.8, y.max() + 0.6)
ax.set_yticks(y); ax.set_yticklabels([r[3] for r in rows], fontsize=6.2)
ax.axhline(y[is_rgb].max() + 0.9, color=GREY, lw=0.5, ls=":")
ax.text(1.03, y[~is_rgb].mean(), "IR frame", rotation=270, va="center", fontsize=6.5, color=IR_C, transform=ax.get_yaxis_transform())
ax.text(1.03, y[is_rgb].mean(), "RGB frame", rotation=270, va="center", fontsize=6.5, color=VIS_C, transform=ax.get_yaxis_transform())
ax.set_xlabel("ΔAP$_{50}$ vs single camera of the frame\n(95% session bootstrap; band: ±0.01 margin)")
ax.grid(axis="x", color="#e6e5e0", lw=0.5); ax.set_axisbelow(True)
panel(ax, "b", "Fusion gain")

# c  localisation tolerance
ax = fig.add_subplot(gs[1, 0])
thr = [("AP10", 0.1), ("AP30", 0.3), ("AP50", 0.5), ("AP75", 0.75)]
for n, col, mk, lab in (("IR only", IR_C, "o", "IR only"), ("RGB only (mapped)", VIS_C, "s", "RGB, mapped"),
                        ("Late ProbEn (box avg.)", GREY_D, "v", "Late ProbEn, box avg."),
                        ("Anchored ProbEn + unmatched RGB", FUS_C, "D", "Anchored + unmatched RGB")):
    if n in MI:
        ax.plot([t for _, t in thr], [MI[n][k]["mean"] for k, _ in thr], color=col, marker=mk, ms=3.4, lw=1.3, label=lab)
ax.set_xticks([0.1, 0.3, 0.5, 0.75]); ax.set_xlabel("IoU threshold"); ax.set_ylabel("AP (IR frame)"); ax.set_ylim(0, 1); ygrid(ax)
ax.legend(fontsize=6.2, loc="lower left", handlelength=1.4, frameon=False)
panel(ax, "c", "Localisation tolerance")

# d  registration precision: realistic (scaled real residuals) and Gaussian error models, seeds pooled
ax = fig.add_subplot(gs[1, 1])
RC = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(DATA, "runs", "registration_curve2_s*.json")))]
for key, col, lab in (("box-avg ProbEn", GREY_D, "Late ProbEn, box avg."), ("anchored ProbEn + unmatched RGB", FUS_C, "Anchored + unmatched RGB")):
    for model, ls, mk in (("empirical", "-", "o"), ("gauss", "--", None)):
        xs = np.mean([[c["map_err_px"] for c in r["curves"][model]] for r in RC], axis=0)
        ys = np.array([[c[key] for c in r["curves"][model]] for r in RC])
        ax.plot(xs, ys.mean(0), color=col, lw=1.3 if ls == "-" else 0.9, ls=ls, marker=mk, ms=2.4,
                label=lab if model == "empirical" else None)
        if model == "empirical":
            ax.fill_between(xs, ys.min(0), ys.max(0), color=col, alpha=0.22, lw=0)
    for pname, mk in (("sequence map", "s"), ("windowed map (w=3)", "^")):
        R1 = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(DATA, "runs", "registration_curve_s*.json")))]
        px = np.mean([r["points"][pname]["map_err_px"] for r in R1]); py = np.mean([r["points"][pname][key] for r in R1])
        ax.plot(px, py, mk, color=col, ms=5, mec=INK, mew=0.5, zorder=5)
ax.axhline(0, color=INK, lw=0.7)
ax.set_xlim(0, 40); ax.set_ylim(-0.2, 0.06)
ax.plot([], [], color=INK2, ls="-", lw=1.2, label="real residuals, scaled")
ax.plot([], [], color=INK2, ls="--", lw=0.9, label="Gaussian noise")
ax.plot([], [], "s", color="white", mec=INK, ms=4.5, label="sequence map")
ax.plot([], [], "^", color="white", mec=INK, ms=4.5, label="windowed map")
ax.set_xlabel("Median alignment error (RGB px)"); ax.set_ylabel("ΔAP$_{50}$ vs IR only"); ygrid(ax)
ax.legend(fontsize=5.8, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=3, frameon=False, handlelength=1.6, columnspacing=0.8)
panel(ax, "d", "Fusion gain vs alignment error")

out = os.path.join(HERE, "fig_antiuav_results")
fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.03); fig.savefig(out + ".png", dpi=300, bbox_inches="tight", pad_inches=0.03)
print("saved", out)
