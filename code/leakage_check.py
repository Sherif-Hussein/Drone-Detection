#!/usr/bin/env python3
"""
Near-duplicate / same-scene check of the 100 extra Anti-UAV410 thermal sequences (E1 training data) against the
Anti-UAV300 TEST frames, with positive and negative controls.

Two independent similarity measures per thermal frame:
  * dHash: 64-bit difference hash of the 9x8 grey image (Hamming distance; near-identical frames give <= ~10 bits)
  * CNN embedding: ImageNet ResNet-18 global feature (cosine similarity; same scene gives high values)
Reference distributions (nearest-neighbour similarity to the test set, excluding the frame's own sequence):
  positive control  AU300 test frames vs other sequences of the SAME test session  (what "same scene" looks like)
  negative control  AU300 train/val frames (sessions disjoint from the test sessions) vs the test set
  query             Anti-UAV410 extra frames vs the test set
A query frame is flagged when it is at least as close to a test frame as the 5th percentile of the positive control
on either measure. Outputs leakage_check.json and a montage of the most similar query/test pairs.

python leakage_check.py --work /tmp/skyfuse_au --out OUT
"""
import argparse, glob, json, os

import cv2
import numpy as np


def dhash(im):
    g = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im, (9, 8), interpolation=cv2.INTER_AREA)
    return np.packbits((g[:, 1:] > g[:, :-1]).flatten())


def hamming(a, B):
    return np.unpackbits(np.bitwise_xor(B, a[None, :]), axis=1).sum(1)


def session_of(name):
    """AU300 frame key '<split>_<date>_<time>_<n>_<m>__<frame>' -> '<date>_<time>'."""
    return "_".join(name.split("_", 1)[1].split("_")[:2])


def seq_of(name):
    return name.split("__")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    import torch, torchvision
    torch.set_num_threads(os.cpu_count())
    meta = json.load(open(os.path.join(a.work, "prep", "meta.json")))
    ir_dir = os.path.join(a.work, "prep", "IR", "images")
    test = sorted(k for k, v in meta["split_of"].items() if v == "test")
    trval = sorted(k for k, v in meta["split_of"].items() if v in ("train", "val"))
    query = sorted(glob.glob(os.path.join(a.work, "a410x", "images", "*.jpg")))
    print("test", len(test), "train/val", len(trval), "query", len(query), flush=True)

    w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    net = torchvision.models.resnet18(weights=w); net.fc = torch.nn.Identity(); net.eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def features(paths):
        H, F = [], []
        for i in range(0, len(paths), a.batch):
            ims = [cv2.imread(p) for p in paths[i:i + a.batch]]
            H += [dhash(im) for im in ims]
            x = np.stack([cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (224, 224), interpolation=cv2.INTER_AREA) for im in ims])
            x = (torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255 - mean) / std
            with torch.no_grad():
                f = net(x).numpy()
            F.append(f / np.linalg.norm(f, axis=1, keepdims=True))
            if i % (a.batch * 20) == 0:
                print("  features", i, "/", len(paths), flush=True)
        return np.stack(H), np.concatenate(F)

    pth = lambda ks: [os.path.join(ir_dir, k + ".jpg") for k in ks]
    Ht, Ft = features(pth(test)); Hr, Fr = features(pth(trval)); Hq, Fq = features(query)
    t_seq = np.array([seq_of(k) for k in test]); t_ses = np.array([session_of(k) for k in test])

    def nearest(H, F, exclude_seq=None, only_session=None, only_seq=None, exclude_self=False):
        hm, cs, arg = [], [], []
        S = F @ Ft.T
        for i in range(len(H)):
            m = np.ones(len(test), bool)
            if exclude_seq is not None:
                m &= t_seq != exclude_seq[i]
            if only_session is not None:
                m &= t_ses == only_session[i]
            if only_seq is not None:
                m &= t_seq == only_seq[i]
            if exclude_self:
                m[i] = False
            if not m.any():
                hm.append(np.nan); cs.append(np.nan); arg.append(-1); continue
            h = hamming(H[i], Ht[m]).astype(float); s = S[i][m]
            hm.append(h.min()); cs.append(s.max()); arg.append(int(np.flatnonzero(m)[np.argmax(s)]))
        return np.array(hm), np.array(cs), np.array(arg)

    pos_h, pos_c, _ = nearest(Ht, Ft, exclude_seq=t_seq, only_session=t_ses)      # same session, other sequence
    seq_h, seq_c, _ = nearest(Ht, Ft, only_seq=t_seq, exclude_self=True)          # same sequence, other frame
    neg_h, neg_c, _ = nearest(Hr, Fr)                                              # disjoint sessions
    q_h, q_c, q_arg = nearest(Hq, Fq)
    ok = ~np.isnan(pos_c)
    control = "same session, other sequence"
    if ok.sum() < 20:  # too few same-session pairs (small subsets): fall back to same-sequence frames
        pos_h, pos_c, ok, control = seq_h, seq_c, ~np.isnan(seq_c), "same sequence, other frame (fallback)"
    thr_c = float(np.percentile(pos_c[ok], 5)); thr_h = float(np.percentile(pos_h[ok], 95))
    pct = lambda v: {p: float(np.nanpercentile(v, p)) for p in (1, 5, 25, 50, 75, 95, 99)}
    # dHash is weakly discriminative on plain-sky thermal frames (unrelated frames also give small distances), so the
    # same-scene flag uses the CNN similarity; "near-identical" requires both measures to agree
    flag = q_c >= thr_c
    neg_flag = neg_c >= thr_c
    near_q = (q_c >= 0.99) & (q_h <= 2); near_n = (neg_c >= 0.99) & (neg_h <= 2)
    qseq = np.array(["_".join(os.path.basename(p).split("__")[0].split("_")) for p in query])
    per_seq = {}
    for s in sorted(set(qseq)):
        m = qseq == s
        per_seq[s] = {"frames": int(m.sum()), "flagged": int(flag[m].sum()), "max_cos": float(q_c[m].max()), "min_hamming": float(q_h[m].min())}
    res = {"n": {"test": len(test), "trainval_negative_control": len(trval), "query_a410x": len(query)},
           "thresholds": {"cosine_ge": thr_c, "hamming_le": thr_h, "rule": "5th percentile of the positive control", "positive_control": control},
           "cosine": {"positive_control": pct(pos_c), "same_sequence_other_frame": pct(seq_c), "negative_disjoint_sessions": pct(neg_c), "query_a410x": pct(q_c)},
           "hamming": {"positive_control": pct(pos_h), "same_sequence_other_frame": pct(seq_h), "negative_disjoint_sessions": pct(neg_h), "query_a410x": pct(q_h)},
           "flagged": {"query_frames": int(flag.sum()), "query_share": float(flag.mean()),
                       "negative_control_share": float(neg_flag.mean()),
                       "query_sequences_with_any_flag": sorted(s for s, v in per_seq.items() if v["flagged"] > 0)},
           "near_identical_cos_ge_0.99_and_hamming_le_2": {"query_frames": int(near_q.sum()), "negative_control_frames": int(near_n.sum())},
           "per_query_sequence": per_seq}
    json.dump(res, open(os.path.join(a.out, "leakage_check.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_query_sequence"}, indent=1), flush=True)

    # montage of the 12 most similar query/test pairs for visual inspection
    top = np.argsort(-q_c)[:12]
    tiles = []
    for i in top:
        qi = cv2.resize(cv2.imread(query[i]), (256, 205)); ti = cv2.resize(cv2.imread(pth([test[q_arg[i]]])[0]), (256, 205))
        pair = np.hstack([qi, np.full((205, 6, 3), 255, np.uint8), ti])
        cv2.putText(pair, f"cos {q_c[i]:.3f}  ham {int(q_h[i])}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        tiles.append(pair)
    rows = [np.hstack(tiles[j:j + 3]) for j in range(0, len(tiles), 3)]
    cv2.imwrite(os.path.join(a.out, "leakage_top_pairs.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("saved")


if __name__ == "__main__":
    main()
