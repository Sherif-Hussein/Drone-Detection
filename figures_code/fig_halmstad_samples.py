"""Sample frames from the Halmstad dataset with ground-truth boxes and 4x zoom insets (Fig. 3 top)."""
import os, sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mcos_decoder import load_groundtruth

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "samples", "halm")
sys.path.insert(0, HERE)
from style import IR_C, VIS_C, INK, INK2, apply  # noqa: E402

apply()
COLS = [("DRONE", "close"), ("DRONE", "medium"), ("DRONE", "distant"), ("BIRD", "medium"), ("AIRPLANE", "medium"),
        ("HELICOPTER", "medium")]
CLIPS = {"IR": ["IR_DRONE_001", "IR_DRONE_003", "IR_DRONE_014", "IR_BIRD_001", "IR_AIRPLANE_013", "IR_HELICOPTER_006"],
         "V": ["V_DRONE_001", "V_DRONE_002", "V_DRONE_008", "V_BIRD_002", "V_AIRPLANE_001", "V_HELICOPTER_003"]}


def frame_with_box(clip):
    gt = load_groundtruth(os.path.join(SRC, clip + "_LABELS.mat"))
    idx = [i for i, b in enumerate(gt) if b is not None and b[2] > 0]
    k = idx[len(idx) // 2]
    cap = cv2.VideoCapture(os.path.join(SRC, clip + ".mp4")); cap.set(cv2.CAP_PROP_POS_FRAMES, k)
    ok, im = cap.read(); cap.release()
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB), np.asarray(gt[k], float)


fig, axes = plt.subplots(2, 6, figsize=(7.1, 2.12), gridspec_kw={"wspace": 0.04, "hspace": 0.05})
for r, (sensor, color) in enumerate((("IR", IR_C), ("V", VIS_C))):
    for c, clip in enumerate(CLIPS[sensor]):
        ax = axes[r, c]
        im, (x, y, w, h) = frame_with_box(clip)
        H, W = im.shape[:2]
        ax.imshow(im, extent=(0, W, H, 0)); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.add_patch(Rectangle((x - 2, y - 2), w + 4, h + 4, fill=False, ec=color, lw=1.0))
        # zoom inset around the target
        half = max(w, h) * 1.5 + 4
        cx, cy = x + w / 2, y + h / 2
        x0, x1 = int(max(cx - half, 0)), int(min(cx + half, W)); y0, y1 = int(max(cy - half, 0)), int(min(cy + half, H))
        right, low = cx > W / 2, cy > H / 2  # put the inset in the corner farthest from the target
        ins = ax.inset_axes([0.02 if right else 0.62, 0.53 if low else 0.02, 0.36, 0.45])
        ins.imshow(im[y0:y1, x0:x1], interpolation="nearest"); ins.set_xticks([]); ins.set_yticks([])
        for s in ins.spines.values():
            s.set_visible(True); s.set_edgecolor("white"); s.set_linewidth(0.8)
        # size label: same side as the inset, opposite vertical corner (never on the target)
        ax.text(0.03 if right else 0.97, 0.95 if not low else 0.05, f"{np.sqrt(w * h):.0f} px", transform=ax.transAxes,
                va="top" if not low else "bottom", ha="left" if right else "right", fontsize=6,
                color="white", bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.55))
        if r == 0:
            ax.set_title(f"{COLS[c][0].capitalize()}\n{COLS[c][1]}", fontsize=7, color=INK, pad=2)
    axes[r, 0].set_ylabel({"IR": "Thermal IR\n320×256", "V": "Visible\n640×512"}[sensor], fontsize=7, color=INK)
out = os.path.join(HERE, "fig_halmstad_samples")
fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(out + ".png", dpi=300, bbox_inches="tight", pad_inches=0.02)
print("saved", out)
