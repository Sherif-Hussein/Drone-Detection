"""Fig. 4: Halmstad single-sensor results with leakage control, from results_halmstad.json."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from style import IR_C, VIS_C, GREY, GREY_D, INK, INK2, FULL_W, apply, panel, ygrid  # noqa: E402

apply()
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "data", "results_halmstad.json")
runs = json.load(open(SRC))["halmstad"]
SENS = (("IR", "Thermal IR", IR_C), ("V", "Visible", VIS_C))
CLASSES = ["drone", "bird", "airplane", "helicopter"]


SMALL, BIG = int(os.environ.get("HALM_SMALL", 320)), int(os.environ.get("HALM_BIG", 640))


def pick(sensor, proto, imgsz=None):
    imgsz = imgsz or SMALL
    return [r for r in runs.values() if r["sensor"] == sensor and r["protocol"] == proto and r["imgsz"] == imgsz]


def ms(vals):
    v = [x for x in vals if x is not None and not np.isnan(x)]
    return (np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0.0) if v else (np.nan, 0.0)


fig = plt.figure(figsize=(FULL_W, 4.3))
gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.3, 1.0], hspace=0.62, wspace=0.38)

# a  leakage
ax = fig.add_subplot(gs[:, 0])
prot = [("Clip-level split", lambda s: [r["test"]["map50"] for r in pick(s, "clip")], INK2, None),
        ("Random split, as reported", lambda s: [r["test"]["map50"] for r in pick(s, "random")], GREY, "////"),
        ("Same model, unseen clips", lambda s: [r["test_unseen_clips"]["map50"] for r in pick(s, "random") if "test_unseen_clips" in r], "#dcdbd6", None)]
w = 0.26
for j, (lab, f, col, hatch) in enumerate(prot):
    for i, (s, _, _) in enumerate(SENS):
        m, sd = ms(f(s))
        x = i + (j - 1) * w
        ax.bar(x, m, w - 0.03, color=col, edgecolor="white" if hatch is None else GREY_D, hatch=hatch, lw=0.4,
               label=lab if i == 0 else None)
        ax.errorbar(x, m, yerr=sd, color=INK, lw=0.7, capsize=1.5)
        ax.text(x, m + sd + 0.015, f"{m:.2f}", ha="center", va="bottom", fontsize=5.6, color=INK)
ax.set_xticks([0, 1]); ax.set_xticklabels([n for _, n, _ in SENS])
ax.set_ylim(0, 1.3); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_ylabel("Test mAP$_{50}$"); ygrid(ax)
ax.legend(loc="upper left", ncol=1, handlelength=1.4, fontsize=6)
panel(ax, "a", "Random splits inflate scores")

# b  per class (clip split)
ax = fig.add_subplot(gs[0, 1:])
w = 0.36
for j, (s, name, col) in enumerate(SENS):
    for i, c in enumerate(CLASSES):
        vals = [r["test"]["per_class"][c]["ap50"] for r in pick(s, "clip") if r["test"]["per_class"].get(c)]
        m, sd = ms(vals)
        x = i + (j - 0.5) * w
        ax.bar(x, m, w - 0.04, color=col, label=name if i == 0 else None)
        ax.errorbar(x, m, yerr=sd, color=INK, lw=0.7, capsize=1.5)
ax.set_xticks(range(4)); ax.set_xticklabels([c.capitalize() for c in CLASSES])
ax.set_ylim(0, 1.15); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_ylabel("AP$_{50}$ (clip split)"); ygrid(ax)
ax.legend(loc="upper left", ncol=2)
panel(ax, "b", "Per class")

# c  by distance
ax = fig.add_subplot(gs[1, 1])
dists = ["close", "medium", "distant"]
for j, (s, name, col) in enumerate(SENS):
    ms_ = [ms([r["test_by_distance"][d]["map50"] for r in pick(s, "clip") if d in r.get("test_by_distance", {})]) for d in dists]
    xs = np.arange(3) + (j - 0.5) * 0.08
    ax.errorbar(xs, [m for m, _ in ms_], yerr=[sd for _, sd in ms_], color=col, marker="o", ms=4, lw=1.6, capsize=1.5, label=name)
ax.set_xticks(range(3)); ax.set_xticklabels([d.capitalize() for d in dists]); ax.set_xlim(-0.3, 2.3)
ax.set_ylim(0, 1.0); ax.set_ylabel("mAP$_{50}$"); ygrid(ax)
panel(ax, "c", "By distance bin")

# d  input resolution (seed 0)
ax = fig.add_subplot(gs[1, 2])
for i, (s, name, col) in enumerate(SENS):
    lo = [r["test"]["map50"] for r in pick(s, "clip", SMALL) if r["seed"] == 0]
    hi = [r["test"]["map50"] for r in pick(s, "clip", BIG) if r["seed"] == 0]
    if lo and hi:
        x0 = i * 2.2
        ax.plot([x0, x0 + 1], [lo[0], hi[0]], color=col, lw=1.4, marker="o", ms=4)
        ax.text(x0 + 0.5, max(lo[0], hi[0]) + 0.02, f"Δ {hi[0] - lo[0]:+.3f}", ha="center", fontsize=6, color=INK2)
        ax.text(x0 + 0.5, 0.615, "IR" if s == "IR" else "Visible", ha="center", fontsize=6, color=col)
ax.set_xticks([0, 1, 2.2, 3.2]); ax.set_xticklabels(["320", "640", "320", "640"]); ax.set_xlim(-0.5, 3.7)
ax.set_xlabel("Input size (px), seed 0"); ax.set_ylim(0.6, 0.9); ax.set_ylabel("mAP$_{50}$"); ygrid(ax)
panel(ax, "d", "Input size")

out = os.path.join(HERE, "fig_halmstad_results")
fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.03); fig.savefig(out + ".png", dpi=300, bbox_inches="tight", pad_inches=0.03)
print("saved", out)
