"""Cross-check of the custom AP evaluator against pycocotools on the single-camera test detections (all seeds).

python coco_crosscheck.py PREP RUNS OUT.json
"""
import contextlib, io, json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import antiuav_pipeline as P
from fusion_core import Platt
from analyze_antiuav import suite
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

PREP, RUNS, OUT = sys.argv[1:4]
gt, meta = P.load_prepared(PREP)
keys = sorted(k for k, v in meta["split_of"].items() if v == "test")


def coco_ap(dets, g):
    imgs = [{"id": i, "file_name": k} for i, k in enumerate(keys)]
    anns, dts = [], []
    for i, k in enumerate(keys):
        b = g.get(k)
        if b is not None:
            x0, y0, x1, y1 = map(float, b[:4])
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": 1, "bbox": [x0, y0, x1 - x0, y1 - y0],
                         "area": (x1 - x0) * (y1 - y0), "iscrowd": 0})
        for d in dets.get(k, np.zeros((0, 5))):
            dts.append({"image_id": i, "category_id": 1, "bbox": [float(d[0]), float(d[1]), float(d[2] - d[0]), float(d[3] - d[1])],
                        "score": float(d[4])})
    c = COCO()
    c.dataset = {"images": imgs, "annotations": anns, "categories": [{"id": 1, "name": "drone"}]}
    with contextlib.redirect_stdout(io.StringIO()):
        c.createIndex()
        cd = c.loadRes(dts)
        e = COCOeval(c, cd, "bbox"); e.params.maxDets = [1, 10, 100]
        e.evaluate(); e.accumulate(); e.summarize()
    return {"AP50": float(e.stats[1]), "AP50:95": float(e.stats[0]), "AP75": float(e.stats[2])}


out = {}
for s in (0, 1, 2):
    R = json.load(open(os.path.join(RUNS, f"results_antiuav_s{s}.json")))
    D = pickle.load(open(os.path.join(RUNS, f"detections_s{s}.pkl"), "rb"))
    cal = R["calibration"]
    for cam, Gk, c in (("IR", "IR", cal["ir"]), ("RGB", "RGB", cal["rgb"])):
        d = P.apply_cal(D[f"{cam.lower()}_test"], Platt(c["A"], c["B"]))
        ours = suite(d, gt[Gk], keys)
        coco = coco_ap(d, gt[Gk])
        out[f"{cam} s{s}"] = {"ours": {k: float(ours[k]) for k in ("AP50", "AP50:95", "AP75") if k in ours}, "pycocotools": coco}
        print(cam, s, out[f"{cam} s{s}"], flush=True)
json.dump(out, open(OUT, "w"), indent=1)
