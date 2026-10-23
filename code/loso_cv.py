"""Session-disjoint tuning by leave-one-session-out cross-validation on the test set (IR frame).

The official validation split shares recording sessions with training, so it cannot tune fusion settings. Here, for
each of the 10 test sessions, Platt calibration of both detectors and the anchored-fusion settings (rule, matching
radius tau, weight gamma of unmatched RGB detections) are fitted on the other 9 sessions and applied to the held-out
one; the held-out outputs are concatenated and compared with the thermal detector (same per-fold calibration) by a
session-cluster bootstrap. Also computed: a same-modality null, anchored fusion of the thermal detector with a
second thermal detector trained with another seed (no cross-camera map needed).

python loso_cv.py PREP RUNS OUT.json [N_BOOT]
"""
import itertools, json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import antiuav_pipeline as P
from fusion_core import AxisAffine, fuse_anchored
from analyze_antiuav import frame_records, ap_from, seq_pack, ap_of_seqs, session_of_seq, cluster_bootstrap, holm

PREP, RUNS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
NB = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
E = np.zeros((0, 5))
gt, meta = P.load_prepared(PREP)
keys = sorted(k for k, v in meta["split_of"].items() if v == "test")
fo = {}
for k in keys:
    fo.setdefault(P.seq_of(k), []).append(k)
seqs = sorted(fo)
sess = sorted({session_of_seq(s) for s in seqs})
keys_of = {h: [k for k in keys if session_of_seq(P.seq_of(k)) == h] for h in sess}
seeds = [0, 1, 2]
D = {s: pickle.load(open(os.path.join(RUNS, f"detections_s{s}.pkl"), "rb")) for s in seeds}
GRID = list(itertools.product(("proben", "max", "wbf"), (0.5, 1.0, 2.0), (0.0, 0.25, 1.0)))
DEFAULT = ("proben", 1.0, 0.0)


def ap(d, ks):
    r = frame_records(d, gt["IR"], ks)
    return ap_from(np.concatenate([r[k][0] for k in ks]), np.concatenate([r[k][1] for k in ks]), sum(r[k][2] for k in ks))


def anchored(a, b, ks, cfg, prior):
    rule, tau, gam = cfg
    return {k: fuse_anchored(a.get(k, E), b.get(k, E), rule, prior, tau, gam) for k in ks}


res = {"sessions": sess, "grid": [list(g) for g in GRID], "per_seed": {}}
outs = {n: [] for n in ("IR only", "Anchored, CV-tuned", "Anchored ProbEn (default)", "Anchored MAX (default)",
                        "IR + IR (other seed), anchored ProbEn", "IR + IR (other seed), anchored MAX")}
for s in seeds:
    Ds = D[s]
    tmaps = {q: AxisAffine.from_dict(d) for q, d in Ds["tmaps"].items()}
    ir_raw, rgb_raw = Ds["ir_test"], P.to_frame(Ds["rgb_test"], tmaps, "rgb2ir")
    ir2_raw = D[seeds[(seeds.index(s) + 1) % len(seeds)]]["ir_test"]
    cv = {n: {} for n in outs}
    chosen = {}
    for h in sess:
        train_k = [k for k in keys if k not in set(keys_of[h])]
        cir, prior = P.calibrate(ir_raw, gt["IR"], train_k)
        crgb, _ = P.calibrate(Ds["rgb_test"], gt["RGB"], train_k)  # RGB scores calibrated in the RGB camera's own frame
        cir2, _ = P.calibrate(ir2_raw, gt["IR"], train_k)
        ir, rgb, ir2 = P.apply_cal(ir_raw, cir), P.apply_cal(rgb_raw, crgb), P.apply_cal(ir2_raw, cir2)
        scores = {cfg: ap(anchored(ir, rgb, train_k, cfg, prior), train_k) for cfg in GRID}
        best = max(GRID, key=lambda c: scores[c])
        chosen[h] = {"cfg": list(best), "train_ap": scores[best], "train_ap_default": scores[DEFAULT], "train_ap_ir": ap(ir, train_k)}
        hk = keys_of[h]
        cv["IR only"].update({k: ir.get(k, E) for k in hk})
        cv["Anchored, CV-tuned"].update(anchored(ir, rgb, hk, best, prior))
        cv["Anchored ProbEn (default)"].update(anchored(ir, rgb, hk, DEFAULT, prior))
        cv["Anchored MAX (default)"].update(anchored(ir, rgb, hk, ("max", 1.0, 0.0), prior))
        cv["IR + IR (other seed), anchored ProbEn"].update(anchored(ir, ir2, hk, DEFAULT, prior))
        cv["IR + IR (other seed), anchored MAX"].update(anchored(ir, ir2, hk, ("max", 1.0, 0.0), prior))
        print("seed", s, "held out", h, "chosen", best, {k: round(v, 4) for k, v in chosen[h].items() if k != "cfg"}, flush=True)
    res["per_seed"][s] = {"chosen": chosen, "ap50": {n: ap(d, keys) for n, d in cv.items()}}
    print("seed", s, {n: round(v, 4) for n, v in res["per_seed"][s]["ap50"].items()}, flush=True)
    for n in outs:
        outs[n].append(seq_pack(frame_records(cv[n], gt["IR"], keys), fo))

res["ap50_mean"] = {n: float(np.mean([res["per_seed"][s]["ap50"][n] for s in seeds])) for n in outs}
comps = [(n, "IR only") for n in outs if n != "IR only"] + [("Anchored ProbEn (default)", "IR + IR (other seed), anchored ProbEn")]
boot = {f"{a} - {b}": cluster_bootstrap(outs[a], outs[b], seqs, NB) for a, b in comps}
adj = holm({k: v["p_two_sided"] for k, v in boot.items()})
for k, v in boot.items():
    v["p_holm"] = adj[k]
    print(k, round(v["diff"], 4), [round(c, 4) for c in v["ci95"]], "p_holm", round(v["p_holm"], 3), "equiv" if v["equivalent"] else "")
res["cluster_bootstrap_AP50"] = boot
json.dump(res, open(OUT, "w"), indent=1)
print("saved", OUT)
