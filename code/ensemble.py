#!/usr/bin/env python3
"""
Same-camera ensembles (all members draw boxes on the same image, so no cross-camera registration is involved).

Weighted boxes fusion (WBF; Solovyev et al.): detections of all members are clustered by IoU >= iou_thr, each cluster's
box is the score-weighted mean of its members' boxes and its score is the mean score over ALL members (a member that
missed contributes 0), which penalises boxes found by only one model. A ProbEn-style score combination of calibrated
members is also provided.

python ensemble.py --prep PREP --members name=path.pkl:key ... --frame IR --out OUT.json
  path.pkl:key  -> a pickle with dict[key] = {frame_key: (N,5) detections} in the frame's own coordinates
"""
import argparse, json, os, pickle, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import antiuav_pipeline as P  # noqa: E402
from fusion_core import iou_matrix, proben_pair  # noqa: E402
from analyze_antiuav import ap_from, bootstrap, frame_records, seq_pack  # noqa: E402

E = np.zeros((0, 5))


def wbf(dets, iou_thr=0.55, skip=0.01):
    n = len(dets)
    allb = [np.c_[d[d[:, 4] >= skip], np.full(int((d[:, 4] >= skip).sum()), m)] for m, d in enumerate(dets) if len(d)]
    if not allb:
        return E
    b = np.concatenate(allb); b = b[np.argsort(-b[:, 4])]
    clusters, fused = [], []
    for row in b:
        if fused:
            ious = iou_matrix(row[None, :4], np.array([f[:4] for f in fused]))[0]
            j = int(np.argmax(ious))
            if ious[j] >= iou_thr:
                clusters[j].append(row)
                c = np.array(clusters[j]); w = c[:, 4:5]
                fused[j] = np.r_[(c[:, :4] * w).sum(0) / w.sum(), c[:, 4].mean()]
                continue
        clusters.append([row]); fused.append(np.r_[row[:4], row[4]])
    out = []
    for c, f in zip(clusters, fused):
        c = np.array(c)
        best = {}
        for r in c:  # one vote per member (its best box in the cluster)
            best[int(r[5])] = max(best.get(int(r[5]), 0.0), r[4])
        out.append([*f[:4], sum(best.values()) / n])
    return np.array(out, float).reshape(-1, 5)


def ap(d, g, keys, thr=0.5):
    r = frame_records(d, g, keys, thr)
    return ap_from(np.concatenate([v[0] for v in r.values()]), np.concatenate([v[1] for v in r.values()]), sum(v[2] for v in r.values()))


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--prep", required=True); a.add_argument("--members", nargs="+", required=True)
    a.add_argument("--frame", default="IR"); a.add_argument("--out", required=True); a.add_argument("--bootstrap", type=int, default=1000)
    a = a.parse_args()
    gt, meta = P.load_prepared(a.prep)
    G = gt[a.frame]
    kt = sorted(k for k, v in meta["split_of"].items() if v == "test")
    seq_frames = {}
    for k in kt:
        seq_frames.setdefault(P.seq_of(k), []).append(k)
    seqs = sorted(seq_frames)
    mem = {}
    for spec in a.members:
        name, rest = spec.split("=", 1); path, key = rest.rsplit(":", 1)
        mem[name] = pickle.load(open(path, "rb"))[key]
    res = {"members": {n: {"AP50": ap(d, G, kt), "AP50_95": float(np.mean([ap(d, G, kt, t) for t in np.arange(0.5, 0.96, 0.05)]))}
                       for n, d in mem.items()}}
    names = list(mem)
    ens = {k: wbf([mem[n].get(k, E) for n in names]) for k in kt}
    res["ensemble_wbf"] = {"members": names, "AP50": ap(ens, G, kt),
                           "AP50_95": float(np.mean([ap(ens, G, kt, t) for t in np.arange(0.5, 0.96, 0.05)]))}
    best = max(names, key=lambda n: res["members"][n]["AP50"])
    pk = lambda d: [seq_pack(frame_records(d, G, kt), seq_frames)]
    res["bootstrap_vs_best_member"] = {"best_member": best, **bootstrap(pk(ens), pk(mem[best]), seqs, a.bootstrap)}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
