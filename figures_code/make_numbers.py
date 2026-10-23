"""Write paper/srep/numbers.tex (one macro per reported number) and table rows from the result files.
Every number quoted in the manuscript comes from here; missing inputs become red placeholders."""
import glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "srep", "numbers.tex")
M = {}
PEND = "\\pend{--}"


def f3(x, nd=3):
    return PEND if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def msd(vals, nd=3, pm=True):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not v:
        return PEND, None
    if len(v) == 1 or not pm:
        return f"{np.mean(v):.{nd}f}", float(np.mean(v))
    return f"{np.mean(v):.{nd}f}$\\pm${np.std(v, ddof=1):.{nd}f}", float(np.mean(v))


def load(p):
    p = os.path.join(DATA, p)
    return json.load(open(p)) if os.path.exists(p) else None


M["HalmStep"], M["HalmEpochs"], M["AUStep"], M["AUEpochs"] = "15", "30", "40", "30"

# ------------------------------------------------------------------ dataset statistics
hs = load(os.path.join("..", "figs", "fig_dataset_stats.json"))
if hs:
    M["HSizeIR"], M["HSizeV"] = f"{hs['size_median']['IR']:.1f}", f"{hs['size_median']['V']:.1f}"
    M["HLtSixteenIR"], M["HLtSixteenV"] = f"{100 * hs['size_share_lt16']['IR']:.0f}", f"{100 * hs['size_share_lt16']['V']:.1f}"
    M["HPairCorr"] = f"{hs['pairing']['corr_median']:.2f}"; M["HPairAgree"] = f"{100 * hs['pairing']['distance_bin_agreement']:.0f}"
    M["HPairN"] = str(hs["pairing"]["pairs_tested"])
    pr = hs["pairing"]
    M["HPairCorrNull"] = f"{pr['corr_median_null']:.2f}"; M["HPairAgreeNull"] = f"{100 * pr['distance_bin_agreement_null']:.0f}"
    M["HPairNullN"] = f"{pr['null_pairs']:,}"; M["HPairKSp"] = f"{pr['ks_x_real_vs_null'][1]:.2f}"
    M["HPairCorrY"] = f"{pr['corr_y_median']:.2f}"; M["HPairCorrSize"] = f"{pr['corr_size_median']:.2f}"
    M["HEmptyIR"] = f"{100 * hs['empty_frame_share']['IR']:.1f}"; M["HEmptyV"] = f"{100 * hs['empty_frame_share']['V']:.1f}"
    M["HFramesIR"] = f"{hs['frames_total']['IR']:,}"; M["HFramesV"] = f"{hs['frames_total']['V']:,}"
au = load(os.path.join("..", "figs", "fig_antiuav_data_stats.json"))
if au:
    M["ASizeIR"], M["ASizeRGB"] = f"{au['size_median']['IR']:.0f}", f"{au['size_median']['RGB']:.0f}"
    M["ALtSixteenIR"], M["ALtSixteenRGB"] = f"{100 * au['share_lt16']['IR']:.0f}", f"{100 * au['share_lt16']['RGB']:.0f}"
    M["ASizeRatio"] = f"{au['rgb_ir_size_ratio_median']:.1f}"
    M["ANight"], M["ADim"], M["ADay"] = (f"{100 * au['light_share'][x]:.0f}" for x in ("night", "dim", "day"))
    M["AMapAx"] = f"{au['map_ax']['median']:.2f}"; M["AMapAxLo"] = f"{au['map_ax']['p10']:.2f}"; M["AMapAxHi"] = f"{au['map_ax']['p90']:.2f}"
    M["AMapBxLo"] = f"{au['map_bx']['p10']:.0f}"; M["AMapBxHi"] = f"{au['map_bx']['p90']:.0f}"
    for sp in ("train", "val", "test"):
        M[f"AFrames{sp.capitalize()}"] = f"{au['frames'][sp]:,}"; M[f"ASeq{sp.capitalize()}"] = str(au["sequences"][sp])
    v = au["visibility"]; idx = 2  # test split
    M["AVisBoth"], M["AVisIROnly"], M["AVisRGBOnly"], M["AVisNeither"] = (f"{100 * v[k][idx]:.1f}" for k in ("Both", "IR only", "RGB only", "Neither"))

# ------------------------------------------------------------------ Halmstad results
hr = load("results_halmstad.json")
if hr:
    runs = hr["halmstad"]; pm = hr["prep_meta"]
    pick = lambda s, p, z=320: [r for r in runs.values() if r["sensor"] == s and r["protocol"] == p and r["imgsz"] == z]
    for s, S in (("IR", "IR"), ("V", "V")):
        for sp in ("train", "val", "test"):
            M[f"H{S}clips{sp}"] = str(pm["sensors"][s]["clips_by_split"][sp])
            M[f"H{S}{sp}"] = f"{pm['sensors'][s]['clip_' + sp]:,}"
        c, cm = msd([r["test"]["map50"] for r in pick(s, "clip")])
        rr, rm = msd([r["test"]["map50"] for r in pick(s, "random")])
        u, um = msd([r["test_unseen_clips"]["map50"] for r in pick(s, "random")])
        M[f"H{S}clip"], M[f"H{S}rand"], M[f"H{S}unseen"] = c, rr, u
        c95, _ = msd([r["test"]["map50_95"] for r in pick(s, "clip")]); r95, _ = msd([r["test"]["map50_95"] for r in pick(s, "random")])
        M[f"H{S}clipS"], M[f"H{S}randS"] = c95, r95
        if cm is not None and rm is not None:
            M[f"H{S}gap"] = f"{rm - cm:+.3f}"; M[f"H{S}gapRel"] = f"{100 * (rm - cm) / cm:.0f}"
        if rm is not None and um is not None:
            M[f"H{S}optim"] = f"{rm - um:+.3f}"; M[f"H{S}optimRel"] = f"{100 * (rm - um) / um:.0f}"
        for cl in ("drone", "bird", "airplane", "helicopter"):
            M[f"H{S}{cl}"] = msd([r["test"]["per_class"][cl]["ap50"] for r in pick(s, "clip") if r["test"]["per_class"].get(cl)])[0]
        for d in ("close", "medium", "distant"):
            M[f"H{S}{d}"] = msd([r["test_by_distance"][d]["map50"] for r in pick(s, "clip") if d in r.get("test_by_distance", {})])[0]
        hi = [r["test"]["map50"] for r in pick(s, "clip", 640) if r["seed"] == 0]
        lo = [r["test"]["map50"] for r in pick(s, "clip", 320) if r["seed"] == 0]
        M[f"H{S}hi"] = f3(hi[0]) if hi else PEND
        M[f"H{S}hiGain"] = f"{hi[0] - lo[0]:+.3f}" if hi and lo else PEND
        M[f"H{S}lat"] = msd([r["latency_ms_per_frame"] for r in pick(s, "clip")], 1, pm=False)[0]
    M["HalmSeeds"] = str(len({r["seed"] for r in runs.values()}))

# ------------------------------------------------------------------ Anti-UAV results
from numbers_antiuav import add_antiuav
add_antiuav(M, DATA, PEND)

import re
with open(OUT, "w", encoding="utf-8") as f:
    f.write("% generated by figs/make_numbers.py - do not edit\n")
    for k, v in sorted(M.items()):
        assert k.isalpha(), k
        v = re.sub(r"(^|[\s\[(&])-(?=\d)", r"\1$-$", v)  # typographic minus for negative numbers
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(len(M), "macros ->", OUT)
