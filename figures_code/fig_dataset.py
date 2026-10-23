"""Figure: Halmstad dataset audit (all panels computed from the 650 label files)."""
import json, os, glob
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mcos_decoder import load_groundtruth

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "samples", "halmstad_full", "Data")
OUT = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, OUT)
from style import IR_C, VIS_C, GREY, GREY_D, INK, INK2, FULL_W, apply, panel, ygrid
apply()
RES = {"IR": (320, 256), "V": (640, 512)}

d = pd.read_excel(f"{ROOT}/Video_dataset_description.xlsx"); d.columns = [c.strip() for c in d.columns]
dist_col = [c for c in d.columns if c.startswith("DISTANCE")][0]
gts = {}
for _, r in d.iterrows():
    s = "IR" if r.SENSOR == "IR" else "V"; n = f"{s}_{r.CLASS}_{int(r.NUMBER):03d}"
    gts[n] = (s, r.CLASS, str(r[dist_col]).strip(), load_groundtruth(f"{ROOT}/Video_{s}/{n}_LABELS.mat"))

stats = {}
cls = ["DRONE", "BIRD", "AIRPLANE", "HELICOPTER"]; dists = ["CLOSE", "MEDIUM", "DISTANT"]
tab = {s: {c: {dd: 0 for dd in dists} for c in cls} for s in ("IR", "V")}
for s, c, dd, _ in gts.values():
    tab[s][c][dd] += 1
stats["clips"] = tab
# (b) object size
sizes = {s: np.array([np.sqrt(b[2] * b[3]) for (ss, _, _, g) in gts.values() if ss == s for b in g if b is not None]) for s in ("IR", "V")}
stats["size_median"] = {s: float(np.median(v)) for s, v in sizes.items()}
stats["labelled_frames_by_class"] = {s: {c: int(sum(1 for (ss, cc, _, g) in gts.values() if ss == s and cc == c for b in g if b is not None)) for c in cls} for s in ("IR", "V")}
stats["frames_total"] = {s: int(sum(len(g) for (ss, _, _, g) in gts.values() if ss == s)) for s in ("IR", "V")}
stats["empty_frame_share"] = {s: float(sum(1 for (ss, _, _, g) in gts.values() if ss == s for b in g if b is None) / stats["frames_total"][s]) for s in ("IR", "V")}
stats["size_median_by_distance"] = {s: {dd: float(np.median([np.sqrt(b[2] * b[3]) for (ss, _, d2, g) in gts.values() if ss == s and d2 == dd for b in g if b is not None])) for dd in dists} for s in ("IR", "V")}
stats["size_share_lt16"] = {s: float(np.mean(v < 16)) for s, v in sizes.items()}
# (c) pairing audit: same-numbered pairs vs a shuffled-pair null (same class, different clip number)


def tracks(g, gi):
    k = min(len(g), len(gi))
    return np.array([((g[i][0] + g[i][2] / 2) / 640, (gi[i][0] + gi[i][2] / 2) / 320,
                      (g[i][1] + g[i][3] / 2) / 512, (gi[i][1] + gi[i][3] / 2) / 256,
                      np.sqrt(g[i][2] * g[i][3]), np.sqrt(gi[i][2] * gi[i][3])) for i in range(k) if g[i] and gi[i]])


def pair_stats(g, gi):
    x = tracks(g, gi)
    if len(x) < 30:
        return None
    r = lambda a, b: float(np.corrcoef(x[:, a], x[:, b])[0, 1]) if x[:, a].std() > 1e-3 and x[:, b].std() > 1e-3 else np.nan
    return r(0, 1), r(2, 3), r(4, 5)


real, agree = [], []
for n, (s, c, dd, g) in gts.items():
    if s != "V":
        continue
    m = n.replace("V_", "IR_", 1)
    if m not in gts:
        continue
    st = pair_stats(g, gts[m][3])
    if st is None:
        continue
    real.append(st); agree.append(dd == gts[m][2])
real = np.array(real)
rng = np.random.default_rng(0)
null, agree_null = [], []
irs = {c: [n for n, v in gts.items() if v[0] == "IR" and v[1] == c] for c in cls}
for n, (s, c, dd, g) in gts.items():
    if s != "V":
        continue
    own = n.replace("V_", "IR_", 1)
    cand = [m for m in irs[c] if m != own]
    for m in rng.choice(cand, min(10, len(cand)), replace=False):
        st = pair_stats(g, gts[m][3])
        if st is not None:
            null.append(st); agree_null.append(dd == gts[m][2])
null = np.array(null)
corr, corr_null = real[:, 0][~np.isnan(real[:, 0])], null[:, 0][~np.isnan(null[:, 0])]


def ks(a, b):
    a, b = np.sort(a), np.sort(b); z = np.concatenate([a, b])
    d = np.max(np.abs(np.searchsorted(a, z, "right") / len(a) - np.searchsorted(b, z, "right") / len(b)))
    ne = len(a) * len(b) / (len(a) + len(b)); lam = (np.sqrt(ne) + 0.12 + 0.11 / np.sqrt(ne)) * d
    p = 2 * sum((-1) ** (j - 1) * np.exp(-2 * j * j * lam * lam) for j in range(1, 101))
    return float(d), float(min(max(p, 0.0), 1.0))


nm = lambda v: float(np.nanmedian(v))
stats["pairing"] = {"pairs_tested": len(agree), "null_pairs": len(null),
                    "distance_bin_agreement": float(np.mean(agree)), "distance_bin_agreement_null": float(np.mean(agree_null)),
                    "corr_median": nm(real[:, 0]), "corr_median_null": nm(null[:, 0]),
                    "corr_y_median": nm(real[:, 1]), "corr_y_median_null": nm(null[:, 1]),
                    "corr_size_median": nm(real[:, 2]), "corr_size_median_null": nm(null[:, 2]),
                    "corr_share_gt_0.9": float(np.mean(corr > 0.9)), "corr_share_gt_0.9_null": float(np.mean(corr_null > 0.9)),
                    "ks_x_real_vs_null": ks(corr, corr_null)}
# (d) temporal redundancy: box IoU vs frame gap


def iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])); iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / (a[2] * a[3] + b[2] * b[3] - i)


gaps = [1, 2, 3, 5, 8, 10, 15, 20, 30, 45, 60, 90]
curve = {}
for s in ("IR", "V"):
    curve[s] = []
    for gp in gaps:
        v = [iou(g[i], g[i - gp]) for (ss, _, _, g) in gts.values() if ss == s for i in range(gp, len(g)) if g[i] and g[i - gp]]
        curve[s].append((float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75))))
stats["iou_vs_gap"] = {"gaps": gaps, **curve}
json.dump(stats, open(f"{OUT}/fig_dataset_stats.json", "w"), indent=1)

# ------------------------------------------------------------------ plot
fig, ax = plt.subplots(1, 4, figsize=(FULL_W, 2.35), gridspec_kw={"wspace": 0.42, "width_ratios": [1.3, 1, 1, 1]})
ax = np.array([[ax[0], ax[1]], [ax[2], ax[3]]])
# a
a = ax[0, 0]; w = 0.38; xs = np.arange(len(cls) * len(dists))
labels = [f"{c[:5].title()}\n{dd[:4].lower()}" for c in cls for dd in dists]
for j, (s, col, lab) in enumerate((("IR", IR_C, "Thermal IR (365 clips)"), ("V", VIS_C, "Visible (285 clips)"))):
    vals = [tab[s][c][dd] for c in cls for dd in dists]
    a.bar(xs + (j - 0.5) * w, vals, w - 0.04, color=col, label=lab.split(" (")[0] + " (" + lab.split(" (")[1])
a.set_xticks(xs); a.set_xticklabels([f"{dd[0]}" for c in cls for dd in dists], fontsize=5.8)
for i, c in enumerate(cls):
    a.text(i * 3 + 1, -19, c.title()[:4] + ("." if len(c) > 4 else ""), ha="center", fontsize=6, color=INK)
a.set_ylabel("Clips"); panel(a, "b", "Clips by class and range")
a.set_ylim(0, 100); ygrid(a)
# b
b = ax[0, 1]; bins = np.arange(0, 121, 4)
for s, col, lab in (("IR", IR_C, "Thermal IR, 320×256"), ("V", VIS_C, "Visible, 640×512")):
    b.hist(sizes[s], bins=bins, density=True, histtype="stepfilled", alpha=0.25, color=col)
    b.hist(sizes[s], bins=bins, density=True, histtype="step", lw=1.2, color=col, label=f"{lab.split(",")[0]} (median {np.median(sizes[s]):.0f} px)")
    b.axvline(np.median(sizes[s]), color=col, lw=1, ls="--")
b.set_xlabel("Target size √(wh) (px)"); b.set_ylabel("Density"); b.set_xlim(0, 100); b.set_ylim(0, b.get_ylim()[1] * 1.05)
for s_, col in (("IR", IR_C), ("V", VIS_C)):
    b.text(np.median(sizes[s_]) + 2, b.get_ylim()[1] * (0.92 if s_ == "IR" else 0.55), "median\n" + f"{np.median(sizes[s_]):.0f} px", fontsize=5.6, color=INK2, va="top")
panel(b, "c", "Target size")
# c
c = ax[1, 0]
bins = np.linspace(-1, 1, 21)
c.hist(corr, bins=bins, color=GREY, edgecolor="white", lw=0.8, label=f"Same-numbered\nmedian r = {np.median(corr):.2f}")
c.hist(corr_null, bins=bins, weights=np.full(len(corr_null), len(corr) / len(corr_null)), histtype="step", color=INK, lw=1.0,
       label=f"Shuffled (null)\nmedian r = {np.median(corr_null):.2f}")
c.legend(loc="upper left", fontsize=5.4, frameon=False, borderaxespad=0.1, handlelength=1.2)
c.axvline(0.9, color=INK, lw=0.8, ls=":")
c.set_xlabel("IR vs visible track correlation"); c.set_ylabel("Clip pairs")
panel(c, "d", "Clip pairing")
c.set_ylim(0, c.get_ylim()[1] * 1.45)
# d
e = ax[1, 1]
for s, col, lab in (("IR", IR_C, "Thermal IR"), ("V", VIS_C, "Visible")):
    med = [v[0] for v in curve[s]]; lo = [v[1] for v in curve[s]]; hi = [v[2] for v in curve[s]]
    e.fill_between(gaps, lo, hi, color=col, alpha=0.15, lw=0)
    e.plot(gaps, med, color=col, lw=1.4, marker="o", ms=2.5, label=lab)
e.axvline(30, color=INK, lw=0.6, ls=":"); e.text(33, 0.9, "1 s", fontsize=6, color=INK2)
e.set_xscale("log"); e.set_xticks([1, 2, 5, 10, 30, 90]); e.set_xticklabels(["1", "2", "5", "10", "30", "90"])
e.set_xlabel("Frame gap (30 fps)"); e.set_ylabel("IoU with earlier box"); e.set_ylim(0, 1)

panel(e, "e", "Redundancy")
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=IR_C, label="Thermal IR (365 clips, 320×256)"), Patch(color=VIS_C, label="Visible (285 clips, 640×512)")],
           loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6.2, handlelength=1.2)
fig.savefig(f"{OUT}/fig_halmstad_audit.pdf", bbox_inches="tight", pad_inches=0.03); fig.savefig(f"{OUT}/fig_halmstad_audit.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
print(json.dumps({k: v for k, v in stats.items() if k != "iou_vs_gap"}, indent=1))
print("iou gap1", curve["IR"][0], curve["V"][0], "gap30", curve["IR"][gaps.index(30)], curve["V"][gaps.index(30)])

# ------------------------------------------------------------------ compact variant for the short manuscript:
# a pairing test, b target size, c redundancy, d leakage experiment
runs = json.load(open(os.path.join(OUT, "..", "data", "results_halmstad.json")))["halmstad"]


def _pick(sensor, proto):
    return [r for r in runs.values() if r["sensor"] == sensor and r["protocol"] == proto and r["imgsz"] == 320]


def _ms(v):
    v = [x for x in v if x is not None]
    return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)


fig, ax = plt.subplots(1, 4, figsize=(FULL_W, 2.2), gridspec_kw={"wspace": 0.45, "width_ratios": [1, 1, 1, 1.15]})
c = ax[0]
bins_c = np.linspace(-1, 1, 21)
c.hist(corr, bins=bins_c, color=GREY, edgecolor="white", lw=0.8, label=f"same-numbered\nmedian r = {np.median(corr):.2f}")
c.hist(corr_null, bins=bins_c, weights=np.full(len(corr_null), len(corr) / len(corr_null)), histtype="step", color=INK, lw=1.0,
       label=f"shuffled (null)\nmedian r = {np.median(corr_null):.2f}")
c.legend(loc="upper left", fontsize=5.4, frameon=False, borderaxespad=0.1, handlelength=1.2)
c.set_xlabel("IR vs visible track correlation"); c.set_ylabel("Clip pairs"); c.set_ylim(0, c.get_ylim()[1] * 1.5)
panel(c, "a", "Clip pairing")
b = ax[1]; bins_b = np.arange(0, 121, 4)
for s, col in (("IR", IR_C), ("V", VIS_C)):
    b.hist(sizes[s], bins=bins_b, density=True, histtype="stepfilled", alpha=0.25, color=col)
    b.hist(sizes[s], bins=bins_b, density=True, histtype="step", lw=1.2, color=col)
    b.axvline(np.median(sizes[s]), color=col, lw=1, ls="--")
b.set_xlabel("Target size √(wh) (px)"); b.set_ylabel("Density"); b.set_xlim(0, 100)
for s_, col in (("IR", IR_C), ("V", VIS_C)):
    b.text(np.median(sizes[s_]) + 2, b.get_ylim()[1] * (0.92 if s_ == "IR" else 0.55), f"{np.median(sizes[s_]):.0f} px", fontsize=5.8, color=INK2, va="top")
panel(b, "b", "Target size")
e = ax[2]
for s, col in (("IR", IR_C), ("V", VIS_C)):
    med = [v[0] for v in curve[s]]; lo = [v[1] for v in curve[s]]; hi = [v[2] for v in curve[s]]
    e.fill_between(gaps, lo, hi, color=col, alpha=0.15, lw=0)
    e.plot(gaps, med, color=col, lw=1.4, marker="o", ms=2.5)
e.axvline(30, color=INK, lw=0.6, ls=":"); e.text(33, 0.9, "1 s", fontsize=6, color=INK2)
e.set_xscale("log"); e.set_xticks([1, 2, 5, 10, 30, 90]); e.set_xticklabels(["1", "2", "5", "10", "30", "90"])
e.set_xlabel("Frame gap (30 fps)"); e.set_ylabel("IoU with earlier box"); e.set_ylim(0, 1)
panel(e, "c", "Redundancy")
L = ax[3]
prot = [("clip-level split", lambda s: [r["test"]["map50"] for r in _pick(s, "clip")], INK2, None),
        ("random split, as reported", lambda s: [r["test"]["map50"] for r in _pick(s, "random")], GREY, "////"),
        ("same model, unseen clips", lambda s: [r["test_unseen_clips"]["map50"] for r in _pick(s, "random") if "test_unseen_clips" in r], "#dcdbd6", None)]
wd = 0.26
for j, (lab, f_, col, hatch) in enumerate(prot):
    for i, s in enumerate(("IR", "V")):
        m, sd = _ms(f_(s)); x = i + (j - 1) * wd
        L.bar(x, m, wd - 0.03, color=col, edgecolor="white" if hatch is None else GREY_D, hatch=hatch, lw=0.4, label=lab if i == 0 else None)
        L.errorbar(x, m, yerr=sd, color=INK, lw=0.7, capsize=1.5)
        L.text(x, m + sd + 0.015, f"{m:.2f}", ha="center", va="bottom", fontsize=5.2, color=INK)
L.set_xticks([0, 1]); L.set_xticklabels(["Thermal", "Visible"]); L.set_ylim(0, 1.35); L.set_yticks([0, 0.5, 1.0])
L.set_ylabel("Test mAP$_{50}$"); ygrid(L); L.legend(loc="upper left", fontsize=5.2, handlelength=1.2, frameon=False)
panel(L, "d", "Leakage")
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=IR_C, label="Thermal IR (320×256)"), Patch(color=VIS_C, label="Visible (640×512)")],
           loc="lower center", bbox_to_anchor=(0.4, -0.16), ncol=2, fontsize=6.2, handlelength=1.2)
fig.savefig(f"{OUT}/fig_halmstad_compact.pdf", bbox_inches="tight", pad_inches=0.03)
fig.savefig(f"{OUT}/fig_halmstad_compact.png", dpi=250, bbox_inches="tight", pad_inches=0.03)
