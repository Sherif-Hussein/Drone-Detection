"""Fig. (RQ6): AP50 against laptop CPU time for each detector variant, thermal and RGB panels.
Error bars: 95% session-bootstrap interval of the difference to YOLO11-small, drawn around the variant's AP50.
Reads ../data/{ir_small_trial,rgb_small_trial}/compare.json, ../data/runs/cost_bench_pc.jsonl, the CPU-trained nano
(../data/cpu_trial/nano_vs_small.json) and the shared model (../data/jointpad_trial or ../data/joint_trial)."""
import json, os

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from style import IR_C, VIS_C, FUS_C, GREY, GREY_D, INK, INK2, FULL_W, apply, panel, ygrid

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
apply()
cost = {r["tag"]: r for r in (json.loads(l) for l in open(os.path.join(DATA, "runs", "cost_bench_pc.jsonl")) if l.strip())}
ms = lambda t: cost[t]["cpu_forward_ms_median"]
jdir = "jointpad_trial" if os.path.exists(os.path.join(DATA, "jointpad_trial", "compare.json")) else "joint_trial"
J = json.load(open(os.path.join(DATA, jdir, "compare.json")))["cameras"]
NV = json.load(open(os.path.join(DATA, "cpu_trial", "nano_vs_small.json")))

# marker = model family (identity is never colour alone), colour = camera, shared model in the fusion colour
# label offsets (points) per camera and point: (dx, dy, horizontal alignment)
LAB = {("IR", "YOLO11s, 3 seeds"): (7, 8, "left"), ("IR", "YOLO11n, 3 seeds"): (-7, 6, "right"),
       ("IR", "YOLO11n, 320 px"): (0, 9, "center"), ("IR", "YOLOv5s, 320 px"): (6, -7, "left"),
       ("RGB", "YOLO11n, 320 px"): (6, -8, "left"), ("RGB", "YOLOv5s, 320 px"): (6, -6, "left"),
       ("IR", "YOLO11n, CPU-trained"): (6, 7, "left"), ("IR", "YOLO11n + P2"): (0, -9, "center"),
       ("IR", "YOLO11s + P2"): (6, 0, "left"), ("IR", "Shared YOLO11n"): (6, -2, "left"),
       ("RGB", "YOLO11s, 3 seeds"): (7, -8, "left"), ("RGB", "YOLO11n, 3 seeds"): (-7, 6, "right"),
       ("RGB", "YOLO11n, 1280 px"): (0, 9, "center"), ("RGB", "YOLO11n + P2"): (0, -13, "center"),
       ("RGB", "YOLO11s, 1280 px"): (6, -2, "left"), ("RGB", "YOLO11s + P2"): (6, 0, "left"),
       ("RGB", "Shared YOLO11n"): (6, 0, "left")}
MK = {"n": "o", "s": "s", "nP": "^", "sP": "v", "nH": "D", "sH": "P", "cpu": "o", "shared": "*", "nL": "<", "vL": "X"}
N3 = json.load(open(os.path.join(DATA, "nano3_trial", "compare.json")))
LC = json.load(open(os.path.join(DATA, "lowres_trial", "compare.json")))
LR = json.load(open(os.path.join(DATA, "lowres_trial", "lowres_trial.json")))


def points(cam):
    C = json.load(open(os.path.join(DATA, "ir_small_trial" if cam == "IR" else "rgb_small_trial", "compare.json")))
    V = C["variants"]
    if cam == "IR":
        ref_ms = ms("ir_s640")
        rows = [("YOLO11n + P2", "nP", "ir_nP2_640", "ir_nP2_640"),
                ("YOLO11s + P2", "sP", "ir_sP2_640", "ir_sP2_640")]
    else:
        ref_ms = ms("rgb_s960")
        rows = [("YOLO11n, 1280 px", "nH", "n1280", "rgb_n1280"),
                ("YOLO11s, 1280 px", "sH", "s1280", "rgb_s1280"), ("YOLO11n + P2", "nP", "nP2_960", "rgb_nP2_960"),
                ("YOLO11s + P2", "sP", "sP2_960", "rgb_sP2_960")]
    T = N3[cam]; ref_ap = sum(T["small_ap50"]) / 3; L3 = T["last"]
    nano_ms = ms("ir_n640" if cam == "IR" else "rgb_n960")
    ap3 = L3["ap50_mean"]
    pts = [("YOLO11s, 3 seeds", "s3", ref_ms, ref_ap, None),
           ("YOLO11n, 3 seeds", "n3", nano_ms, ap3, (ap3 - L3["diff"] + L3["ci95"][0], ap3 - L3["diff"] + L3["ci95"][1]))]
    pre = "ir" if cam == "IR" else "rgb"; refk = "ir_s640" if cam == "IR" else "rgb_s960"
    for lab, mk, k in (("YOLO11n, 320 px", "nL", f"{pre}_n320"), ("YOLOv5s, 320 px", "vL", f"{pre}_v5s320")):
        b = LC[f"{k} vs {refk}"]; ap = LR[k]["metrics"]["AP50"]
        pts.append((lab, mk, ms(f"{pre}_n320" if mk == "nL" else f"{pre}_v5s320"), ap,
                    (ap - b["diff"] + b["ci95_boot"][0], ap - b["diff"] + b["ci95_boot"][1])))
    for lab, mk, key, ctag in rows:
        b = V[key]["d_AP50_vs_ref_seed0"]; ap = V[key]["metrics"]["AP50"]
        pts.append((lab, mk, ms(ctag), ap, (ap - b["diff"] + b["ci95"][0], ap - b["diff"] + b["ci95"][1])))
    if cam == "IR":
        b = NV["nano_minus_small_seed0"]; ap = NV["nano"]["AP50"]
        pts.append(("YOLO11n, CPU-trained", "cpu", nano_ms * 1.07, ap, (ap - b["diff"] + b["ci95"][0], ap - b["diff"] + b["ci95"][1])))
    j = J[cam]; b = j["vs_small_seed0"]; ap = j["metrics"]["AP50"]
    pts.append(("Shared YOLO11n", "shared", nano_ms * 0.93, ap, (ap - b["diff"] + b["ci95"][0], ap - b["diff"] + b["ci95"][1])))
    return ref_ap, pts


fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.55))
for ax, cam, col, letter, title, pub in ((axes[0], "IR", IR_C, "a", "Thermal camera (640$\\times$512)", 0.812),
                                        (axes[1], "RGB", VIS_C, "b", "RGB camera (960$\\times$544 unless noted)", 0.807)):
    ref, pts = points(cam)
    ax.axhspan(ref - 0.01, ref + 0.01, color=GREY, alpha=0.25, lw=0, zorder=0)
    ax.axhline(ref, color=GREY_D, lw=0.6, ls="--", zorder=0)
    for lab, mk, x, y, ci in pts:
        c = FUS_C if mk == "shared" else col
        filled = mk not in ("cpu",)
        three = mk in ("n3", "s3")
        if ci:
            ax.plot([x, x], ci, color=c, lw=1.6 if three else 0.8, alpha=0.9 if three else 0.6, zorder=2)
        ax.plot(x, y, marker={"n3": "o", "s3": "s"}.get(mk, MK.get(mk)), ms=9 if mk == "shared" else (7.5 if three else 5.5), mfc=c if filled else "white", mec="white" if filled else c,
                mew=0.8 if filled else 1.1, ls="none", zorder=3)

    ax.set_xscale("log")
    ax.set_xticks([10, 20, 50, 100, 200, 400]); ax.set_xticklabels(["10", "20", "50", "100", "200", "400"])
    ax.xaxis.set_minor_formatter(NullFormatter()); ax.tick_params(axis="x", which="minor", length=0)
    ax.set_xlim(9, 700); ax.set_ylim(0.815 if cam == "IR" else 0.885, 0.925 if cam == "IR" else 0.975)
    ax.set_xlabel("Laptop CPU time per frame (ms, log scale)"); ax.set_ylabel("AP$_{50}$")
    ax.annotate(f"published YOLOv5s, 320 px: {pub:.3f}\n(no CPU time reported) $\\downarrow$", xy=(0.02, 0.03),
                xycoords="axes fraction", fontsize=5.8, color=INK2, va="bottom")
    ygrid(ax); panel(ax, letter, title)
from matplotlib.lines import Line2D
LEG = [("YOLO11s, 3 seeds (reference)", "s", True, 7.5), ("YOLO11n, 3 seeds (recommended)", "o", True, 7.5),
       ("YOLO11n, CPU-trained", "o", False, 5.5), ("YOLO11n + P2", "^", True, 5.5), ("YOLO11s + P2", "v", True, 5.5),
       ("YOLO11n, 1280 px", "D", True, 5.5), ("YOLO11s, 1280 px", "P", True, 5.5), ("YOLO11n, 320 px", "<", True, 5.5),
       ("YOLOv5s, 320 px", "X", True, 5.5), ("Shared YOLO11n (both cameras)", "*", True, 8)]
hs = [Line2D([], [], marker=m, ls="none", ms=sz, mfc=GREY_D if f else "white", mec="white" if f else GREY_D, mew=0.8 if f else 1.1,
             label=l) for l, m, f, sz in LEG]
fig.legend(handles=hs, loc="lower center", ncol=5, frameon=False, fontsize=6, bbox_to_anchor=(0.5, -0.13), handletextpad=0.3,
           columnspacing=1.2)
fig.tight_layout(w_pad=2.0)
out = os.path.join(HERE, "..", "submission_ijira_short", "figures", "fig_pareto")
fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.03); fig.savefig(out + ".png", dpi=300, bbox_inches="tight", pad_inches=0.03)
print("saved", out, "shared from", jdir)
