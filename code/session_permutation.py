#!/usr/bin/env python3
"""
Few-cluster robustness of the session-level inference (reviewer request).

1. Exact session permutation test: under the null of no difference between two methods, the method labels are
   exchangeable within each recording session, so all 2^S label swaps (S = 10 test sessions, 1,024 patterns) give the
   exact null distribution of the AP50 difference (mean over seeds, swap applied to every seed alike). Two-sided p.
2. Holm adjustment within explicitly defined families: the comparisons against the single camera in the thermal image
   frame (family IR) and in the RGB image frame (family RGB), as in Table 2.
3. Equivalence-margin sensitivity: the 90% session-bootstrap interval judged against margins of 0.005, 0.01 and 0.02.

python session_permutation.py PREP RUNS OUT.json
"""
import itertools, json, os, pickle, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import antiuav_pipeline as P
from analyze_antiuav import frame_records, seq_pack, ap_of_seqs, cluster_bootstrap, holm, session_of_seq

MARGINS = (0.005, 0.01, 0.02)


def setup(prep):
    gt, meta = P.load_prepared(prep)
    keys = sorted(k for k, v in meta["split_of"].items() if v == "test")
    fo = {}
    for k in keys:
        fo.setdefault(P.seq_of(k), []).append(k)
    return gt, keys, fo, sorted(fo)


def permutation_test(packs_a, packs_b, seqs):
    """Exact two-sided session permutation p-value for the seed-mean AP50 difference."""
    sess = sorted({session_of_seq(s) for s in seqs})
    by = {h: [s for s in seqs if session_of_seq(s) == h] for h in sess}

    def stat(flip):
        out = []
        for pa, pb in zip(packs_a, packs_b):
            A, B = {}, {}
            for h, f in zip(sess, flip):
                for s in by[h]:
                    A[s], B[s] = (pb[s], pa[s]) if f else (pa[s], pb[s])
            out.append(ap_of_seqs(A, seqs) - ap_of_seqs(B, seqs))
        return float(np.mean(out))

    obs = stat([0] * len(sess))
    null = np.array([stat(f) for f in itertools.product((0, 1), repeat=len(sess))])
    p = float(np.mean(np.abs(null) >= abs(obs) - 1e-12))
    return {"diff": obs, "p_perm": p, "n_patterns": len(null), "n_sessions": len(sess)}


def compare(packs_a, packs_b, seqs, nboot=2000):
    b = cluster_bootstrap(packs_a, packs_b, seqs, nboot)
    r = permutation_test(packs_a, packs_b, seqs)
    r.update({"ci95_boot": b["ci95"], "ci90_boot": b["ci90"], "p_boot": b["p_two_sided"],
              "equivalent": {str(m): bool(-m < b["ci90"][0] and b["ci90"][1] < m) for m in MARGINS}})
    return r


def main():
    PREP, RUNS, OUT = sys.argv[1:4]
    gt, keys, fo, seqs = setup(PREP)
    C = pickle.load(open(os.path.join(RUNS, "methods_cache.pkl"), "rb"))["methods"]
    pk = lambda d, fr: seq_pack(frame_records(d, gt[fr], keys), fo)
    fam = {"IR": [(m, "IR only") for m in ("Anchored ProbEn", "Anchored MAX", "Anchored condition-aware", "Anchored ProbEn + temporal",
                                           "Late ProbEn (box avg.)", "Late OR (box avg.)", "Late condition-aware (box avg.)",
                                           "Pixel-level, IR grid", "Hybrid (IR+RGB+pixel)", "Anchored hybrid (IR+RGB+pixel)",
                                           "Cascade (mapped)", "Anchored ProbEn, GT-fitted map")]
                  + [("Anchored ProbEn", "Late ProbEn (box avg.)")],
           "RGB": [(m, "RGB only") for m in ("Anchored ProbEn", "Cascade + RGB detector", "Anchored: RGB + cascade",
                                            "Pixel-level, RGB grid", "Late ProbEn (box avg.)")]}
    res = {"families": {}, "margins": MARGINS}
    for fr, comps in fam.items():
        rows = {}
        for x, y in comps:
            if x in C[fr] and y in C[fr]:
                rows[f"{x} - {y}"] = compare([pk(d, fr) for d in C[fr][x]], [pk(d, fr) for d in C[fr][y]], seqs)
        adj = holm({k: v["p_perm"] for k, v in rows.items()})
        for k, v in rows.items():
            v["p_perm_holm"] = adj[k]
            print(fr, f"{k:60s} d {v['diff']:+.4f} p_perm {v['p_perm']:.3f} holm {v['p_perm_holm']:.3f} p_boot {v['p_boot']:.3f}",
                  "equiv", v["equivalent"], flush=True)
        res["families"][fr] = rows
    # RQ6 (seed 0): lightweight variants against YOLO11-small; its own family per camera
    D = os.path.join(RUNS, "..")
    rq6 = {}
    IRT = pickle.load(open(os.path.join(D, "ir_small_trial", "ir_small_trial_test_dets.pkl"), "rb"))
    RGBT = pickle.load(open(os.path.join(D, "rgb_small_trial", "rgb_small_trial_test_dets.pkl"), "rb"))
    M0 = pickle.load(open(os.path.join(RUNS, "detections_s0.pkl"), "rb"))
    CPU = pickle.load(open(os.path.join(D, "cpu_trial", "cpu_trial_test_dets.pkl"), "rb"))
    JA = pickle.load(open(os.path.join(D, "joint_trial", "joint_trial_test_dets.pkl"), "rb"))
    JB = pickle.load(open(os.path.join(D, "jointpad_trial", "jointpad_trial_test_dets.pkl"), "rb"))
    ir_ref, rgb_ref = IRT["ir_s640"], M0["rgb_test"]
    fams6 = {"RQ6-IR": [("YOLO11-nano", IRT["ir_n640"], ir_ref), ("YOLO11-nano CPU-trained", CPU, ir_ref),
                        ("YOLO11-nano + P2", IRT["ir_nP2_640"], ir_ref), ("YOLO11-small + P2", IRT["ir_sP2_640"], ir_ref),
                        ("Shared enlarged vs separate nano", JA["joint_n960_on_IR"], IRT["ir_n640"]),
                        ("Shared native vs separate nano", JB["jointpad_n960_on_IR"], IRT["ir_n640"])],
             "RQ6-RGB": [("YOLO11-nano", RGBT["n960"], rgb_ref), ("YOLO11-nano 1280", RGBT["n1280"], rgb_ref),
                         ("YOLO11-small 1280", RGBT["s1280"], rgb_ref), ("YOLO11-nano + P2", RGBT["nP2_960"], rgb_ref),
                         ("YOLO11-small + P2", RGBT["sP2_960"], rgb_ref),
                         ("Shared enlarged vs separate nano", JA["joint_n960_on_RGB"], RGBT["n960"]),
                         ("Shared native vs separate nano", JB["jointpad_n960_on_RGB"], RGBT["n960"])]}
    for fname, comps in fams6.items():
        fr = "IR" if fname.endswith("IR") else "RGB"
        rows = {name: compare([pk(a, fr)], [pk(b, fr)], seqs) for name, a, b in comps}
        adj = holm({k: v["p_perm"] for k, v in rows.items()})
        for k, v in rows.items():
            v["p_perm_holm"] = adj[k]
            print(fname, f"{k:40s} d {v['diff']:+.4f} p_perm {v['p_perm']:.3f} holm {v['p_perm_holm']:.3f} p_boot {v['p_boot']:.3f}",
                  "equiv", v["equivalent"], flush=True)
        rq6[fname] = rows
    res["families"].update(rq6)
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
