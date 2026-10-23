#!/usr/bin/env python3
"""
Compute and memory cost of one detector configuration on this PC (run each configuration in its own process).
The architecture is built with one class, as trained; cost does not depend on the trained weights. Reports parameters,
GFLOPs at the real input shape, FP32 ONNX size, network forward latency with ONNX Runtime on the CPU (batch 1, letterboxed
frame shape, NMS excluded) and the peak extra RAM of the process for loading and running the model.

python cost_bench.py --cfg yolo11n-p2.yaml --shape 512 640 --tag ir_nP2_640 --out cost.jsonl
"""
import argparse, json, os, sys, tempfile, time

import numpy as np
import psutil

HERE = os.path.dirname(os.path.abspath(__file__))


def rss_peak():
    mi = psutil.Process().memory_info()
    return getattr(mi, "peak_wset", mi.rss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True); ap.add_argument("--shape", type=int, nargs=2, required=True)
    ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=50)
    a = ap.parse_args()
    a.out = os.path.abspath(a.out)
    import torch
    import onnxruntime as ort
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.torch_utils import get_flops
    cfg = os.path.join(HERE, a.cfg.replace("n-p2", "-p2").replace("s-p2", "-p2")) if "p2" in a.cfg else a.cfg
    scale = "n" if ("11n" in a.cfg or "v5n" in a.cfg) else "s"
    import yaml
    family = ("v5", "yolov5.yaml") if "yolov5" in a.cfg else ("11", "yolo11.yaml")
    d = yaml.safe_load(open(cfg if os.path.exists(cfg) else os.path.join(
        os.path.dirname(__import__("ultralytics").__file__), "cfg", "models", *family)))
    d["scale"] = scale; d["nc"] = 1
    m = DetectionModel(d, nc=1, verbose=False).eval()
    params = int(sum(p.numel() for p in m.parameters()))
    h, w = a.shape
    gflops = float(get_flops(m, [h, w]))
    onnx_path = os.path.join(tempfile.mkdtemp(), a.tag + ".onnx")
    x = torch.zeros(1, 3, h, w)
    for mod in m.modules():  # export the raw head output
        if hasattr(mod, "export"):
            mod.export = True
    torch.onnx.export(m, x, onnx_path, opset_version=17, input_names=["images"], output_names=["out"], dynamo=False)
    onnx_mb = os.path.getsize(onnx_path) / 2 ** 20
    del m
    import gc; gc.collect()
    base = psutil.Process().memory_info().rss
    peak0 = rss_peak()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp = np.random.default_rng(0).random((1, 3, h, w), dtype=np.float32)
    for _ in range(5):
        sess.run(None, {"images": inp})
    t = []
    for _ in range(a.runs):
        s = time.perf_counter(); sess.run(None, {"images": inp}); t.append((time.perf_counter() - s) * 1000)
    peak = rss_peak()
    r = {"tag": a.tag, "cfg": a.cfg, "input_hw": [h, w], "params_M": params / 1e6, "GFLOPs": gflops,
         "onnx_fp32_MB": onnx_mb, "cpu_forward_ms_median": float(np.median(t)), "cpu_forward_ms_p90": float(np.percentile(t, 90)),
         "ram_extra_MB": (max(peak, peak0) - base) / 2 ** 20, "cpu": "i7-8550U (4C/8T)", "backend": "onnxruntime " + ort.__version__}
    with open(a.out, "a") as f:
        f.write(json.dumps(r) + "\n")
    print(json.dumps(r))


if __name__ == "__main__":
    main()
