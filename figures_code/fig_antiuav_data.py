"""Fig. 3: Anti-UAV300 data - sample synchronised pairs and distributions (from prep gt.json/meta.json/maps)."""
import json, math, os, sys, zipfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from style import IR_C, VIS_C, GREY, GREY_D, INK, INK2, FULL_W, apply, panel, ygrid  # noqa: E402

apply()
PREP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "data", "antiuav")
SAMPLES = os.path.join(PREP, "samples")
gt = json.load(open(os.path.join(PREP, "gt.json")))
meta = json.load(open(os.path.join(PREP, "meta.json")))
sys.path.insert(0, os.path.join(HERE, "..", "..", "code"))
import antiuav_pipeline as P  # noqa: E402
_gt, _meta = P.load_prepared(PREP)
_gm = P.gt_mappings(_gt, _meta, ("train", "val"))  # scale-bounded refit, as used in the experiments
maps = {"per_seq": {s: m.to_dict() for s, m in _gm.items()}, "fitted": sorted(_gm), "global": P.global_mapping(_gm).to_dict()}
sz = lambda b: math.sqrt(max(b[2] - b[0], 0) * max(b[3] - b[1], 0))
keys = sorted(meta["split_of"])
stats = {}

# ---------------------------------------------------------------- sample pairs (chosen automatically)
pairs = []
if os.path.isdir(SAMPLES):
    cand = [k for k in keys if os.path.exists(os.path.join(SAMPLES, k + "_IR.jpg")) and gt["IR"][k] and gt["RGB"][k]]
    br = {k: meta["scene"][k]["rgb_brightness"] for k in cand}
    day = sorted([k for k in cand if br[k] >= 100], key=lambda k: sz(gt["RGB"][k]))
    night = sorted([k for k in cand if br[k] < 50], key=lambda k: sz(gt["RGB"][k]))
    pick = []
    if day:
        pick += [day[len(day) * 3 // 4], day[len(day) // 8]]       # larger and small daytime targets
    if night:
        pick += [night[len(night) // 2]]
    pairs = pick[:3]

fig = plt.figure(figsize=(FULL_W, 3.9 if pairs else 3.0))
outer = fig.add_gridspec(2 if pairs else 1, 1, height_ratios=[0.6, 1.0] if pairs else [1], hspace=0.3)
if pairs:
    top = outer[0].subgridspec(1, 6, wspace=0.04, width_ratios=[1.25, 1.78] * 3)
    titles = ["Day, larger target", "Day, small target", "Night"]
    for i, k in enumerate(pairs):
        for r, (mod, fn, col) in enumerate((("IR", "_IR.jpg", IR_C), ("RGB", "_RGBs.jpg", VIS_C))):
            ax = fig.add_subplot(top[2 * i + r])
            im = plt.imread(os.path.join(SAMPLES, k + fn)); H, W = im.shape[:2]
            ax.imshow(im, cmap="gray" if im.ndim == 2 else None); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            b = np.array(gt[mod][k], float)
            if mod == "RGB":
                b = b / meta["rgb_scale"][k]
            ax.add_patch(Rectangle((b[0] - 4, b[1] - 4), b[2] - b[0] + 8, b[3] - b[1] + 8, fill=False, ec=col, lw=0.9))
            crop = os.path.join(SAMPLES, k + ("_IRcrop.jpg" if mod == "IR" else "_RGBcrop.jpg"))
            cx = (b[0] + b[2]) / 2 / W; cy = (b[1] + b[3]) / 2 / H
            if os.path.exists(crop):
                ins = ax.inset_axes([0.02 if cx > 0.5 else 0.6, 0.55 if cy > 0.5 else 0.02, 0.38, 0.43])
                ins.imshow(plt.imread(crop)); ins.set_xticks([]); ins.set_yticks([])
                for s in ins.spines.values():
                    s.set_visible(True); s.set_edgecolor("white"); s.set_linewidth(0.7)
            ax.text(0.5, -0.03, (f"IR · {sz(gt['IR'][k]):.0f} px" if mod == "IR" else f"RGB · {sz(gt['RGB'][k]):.0f} px"),
                    transform=ax.transAxes, ha="center", va="top", fontsize=5.8, color=col)
            if r == 0:
                ax.text(0.0, 1.04, titles[i], transform=ax.transAxes, ha="left", va="bottom", fontsize=6.2, color=INK2)
            if r == 0 and i == 0:
                ax.text(0.0, 1.3, r"$\bf{a}$  Synchronised but unregistered IR/RGB pairs (ground truth; insets at native resolution)",
                        transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5, color=INK)
    stats["samples"] = pairs

bot = outer[-1].subgridspec(1, 4, wspace=0.42, width_ratios=[1.15, 1, 0.85, 1.1])
# b  size distributions
ax = fig.add_subplot(bot[0])
s_ir = np.array([sz(b) for b in gt["IR"].values() if b]); s_rgb = np.array([sz(b) for b in gt["RGB"].values() if b])
bins = np.logspace(np.log10(3), np.log10(400), 32)
for s, col, lab in ((s_ir, IR_C, "IR"), (s_rgb, VIS_C, "RGB")):
    ax.hist(s, bins=bins, histtype="stepfilled", alpha=0.22, color=col, density=True)
    ax.hist(s, bins=bins, histtype="step", lw=1.3, color=col, density=True, label=f"{lab} (median {np.median(s):.0f} px)")
    ax.axvline(np.median(s), color=col, lw=0.8, ls="--")
ax.set_xscale("log"); ax.set_xticks([4, 16, 64, 256]); ax.set_xticklabels(["4", "16", "64", "256"])
ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
ax.set_xlabel("Target size √(wh) (px)"); ax.set_ylabel("Density"); ax.legend(loc="upper left", fontsize=5.4, handlelength=1.0, frameon=True, facecolor="white", edgecolor="none", framealpha=1)
panel(ax, "b" if pairs else "a", "Target size")
stats["size_median"] = {"IR": float(np.median(s_ir)), "RGB": float(np.median(s_rgb))}
stats["share_lt16"] = {"IR": float(np.mean(s_ir < 16)), "RGB": float(np.mean(s_rgb < 16))}

# c  visibility agreement per split
ax = fig.add_subplot(bot[1])
cats = [("Both", lambda i, r: i and r, INK2), ("IR only", lambda i, r: i and not r, IR_C),
        ("RGB only", lambda i, r: r and not i, VIS_C), ("Neither", lambda i, r: not i and not r, "#dcdbd6")]
splits = ["train", "val", "test"]
bottom = np.zeros(3); vis = {}
for name, f, col in cats:
    v = np.array([np.mean([f(gt["IR"][k] is not None, gt["RGB"][k] is not None) for k in keys if meta["split_of"][k] == sp]) for sp in splits])
    ax.bar(range(3), v * 100, 0.62, bottom=bottom * 100, color=col, label=name, edgecolor="white", lw=0.6)
    bottom += v; vis[name] = v.tolist()
ax.set_xticks(range(3)); ax.set_xticklabels([s.capitalize() for s in splits]); ax.set_ylim(0, 100)
ax.set_ylabel("Frames (%)"); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=5.8, handlelength=1)
panel(ax, "c" if pairs else "b", "Target visible in")
stats["visibility"] = vis

# d  brightness
ax = fig.add_subplot(bot[2])
br = np.array([meta["scene"][k]["rgb_brightness"] for k in keys])
ax.hist(br, bins=np.arange(0, 257, 8), color=GREY_D, edgecolor="white", lw=0.4)
for x in (50, 100):
    ax.axvline(x, color=INK, lw=0.6, ls=":")
yl = ax.get_ylim()[1]
ax.set_ylim(0, yl * 1.25)
for x, t, h in ((25, "night", 1.22), (178, "day", 1.22)):
    ax.text(x, yl * h, t, ha="center", va="top", fontsize=5.6, color=INK2)
ax.set_xlabel("Mean RGB grey level"); ax.set_ylabel("Frames")
panel(ax, "d" if pairs else "c", "Illumination")
stats["light_share"] = {"night": float(np.mean(br < 50)), "dim": float(np.mean((br >= 50) & (br < 100))), "day": float(np.mean(br >= 100))}

# e  per-sequence mapping
ax = fig.add_subplot(bot[3])
ps = [maps["per_seq"][s] for s in maps["fitted"]]
axs = np.array([p["ax"] for p in ps]); bxs = np.array([p["bx"] for p in ps])
ax.scatter(axs, bxs, s=7, color=GREY_D, edgecolor="white", lw=0.3, zorder=3)
g = maps["global"]; ax.scatter([g["ax"]], [g["bx"]], s=40, marker="*", color=INK, zorder=4, label="Global map")
ax.set_xlabel("Scale $a_x$ (RGB px per IR px)"); ax.set_ylabel("Offset $b_x$ (RGB px)"); ygrid(ax)
ax.legend(loc="best", fontsize=5.8)
panel(ax, "e" if pairs else "d", "IR→RGB map per sequence")
stats["map_ax"] = {"median": float(np.median(axs)), "p10": float(np.percentile(axs, 10)), "p90": float(np.percentile(axs, 90))}
stats["map_bx"] = {"median": float(np.median(bxs)), "p10": float(np.percentile(bxs, 10)), "p90": float(np.percentile(bxs, 90))}
paired = [(sz(gt["RGB"][k]) / max(sz(gt["IR"][k]), 1e-6)) for k in keys if gt["IR"][k] and gt["RGB"][k]]
stats["rgb_ir_size_ratio_median"] = float(np.median(paired))
stats["frames"] = {sp: sum(1 for k in keys if meta["split_of"][k] == sp) for sp in splits}
stats["sequences"] = {sp: len({k.rsplit('__', 1)[0] for k in keys if meta["split_of"][k] == sp}) for sp in splits}

out = os.path.join(HERE, "fig_antiuav_data")
fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.03); fig.savefig(out + ".png", dpi=300, bbox_inches="tight", pad_inches=0.03)
json.dump(stats, open(out + "_stats.json", "w"), indent=1)
print(json.dumps(stats, indent=1))
