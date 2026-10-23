"""Session-cluster bootstrap (with Holm correction) of the single-detector thermal ablation, seed 0.

python ablation_cluster.py PREP ABLATION_DIR [COMBO_DIR] OUT.json
"""
import glob, json, os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import antiuav_pipeline as P
from analyze_antiuav import frame_records, seq_pack, cluster_bootstrap, holm, suite

PREP, OUT = sys.argv[1], sys.argv[-1]
dirs = sys.argv[2:-1]
gt, meta = P.load_prepared(PREP)
kt = sorted(k for k, v in meta["split_of"].items() if v == "test")
fo = {}
for k in kt:
    fo.setdefault(P.seq_of(k), []).append(k)
seqs = sorted(fo)
dets = {}
for d in dirs:
    for f in glob.glob(os.path.join(d, "ablation_*_s0.pkl")):
        dets[os.path.basename(f)[len("ablation_"):].rsplit("_s", 1)[0]] = pickle.load(open(f, "rb"))["test"]
pack = {v: [seq_pack(frame_records(d, gt["IR"], kt), fo)] for v, d in dets.items()}
res = {v: {"AP50": float(suite(d, gt["IR"], kt)["AP50"])} for v, d in dets.items()}
boot = {v: cluster_bootstrap(pack[v], pack["ir_base"], seqs, 2000) for v in dets if v != "ir_base"}
adj = holm({v: b["p_two_sided"] for v, b in boot.items()})
for v, b in boot.items():
    b["p_holm"] = adj[v]; res[v]["vs_base"] = b
    print(v, round(res[v]["AP50"], 4), round(b["diff"], 4), [round(x, 4) for x in b["ci95"]], "p_holm", round(b["p_holm"], 3))
json.dump(res, open(OUT, "w"), indent=1)
