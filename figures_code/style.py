"""Shared figure style for the manuscript (print, light surface).
Colour roles are fixed across all figures: IR = orange, visible/RGB = blue, fusion = aqua (first three slots of
the validated categorical palette, which pass all-pairs CVD checks); neutral greys for baselines and context."""
import matplotlib.pyplot as plt

IR_C, VIS_C, FUS_C = "#eb6834", "#2a78d6", "#1baf7a"
GREY, GREY_D = "#b9b8b3", "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
SEQ = ["#d6e6f8", "#9cc2ee", "#5b9be3", "#2a78d6", "#1d5aa6", "#123d72"]  # single-hue blue ramp, light -> dark
FULL_W, HALF_W = 7.1, 3.46  # Scientific Reports figure widths (inches): 180 mm and 88 mm


def apply():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.edgecolor": GREY_D, "axes.linewidth": 0.6, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300})


def panel(ax, letter, title=None):
    """Bold panel letter (Nature style) with an optional short title."""
    ax.set_title(("$\\bf{" + letter + "}$  " + title) if title else "$\\bf{" + letter + "}$", loc="left", color=INK, pad=4)


def ygrid(ax):
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
