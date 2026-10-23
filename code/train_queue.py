#!/usr/bin/env python3
"""
Train a queue of YOLO models one after another on one device (one process per GPU, so that two
queues can run in parallel on a 2-GPU machine). Each finished run gets a DONE marker and is skipped
when the queue is re-run.

python train_queue.py --project RUNS --device 0 --seed 0 --epochs 30 \
    --jobs '[{"name": "ir_s0", "data": ".../IR/data.yaml", "imgsz": 640}]'
"""
import argparse, json, os, time

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True); p.add_argument("--jobs", required=True)
    p.add_argument("--device", default="0"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="yolo11s.pt"); p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=10); p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--claim_dir", default=None, help="shared directory: several workers pull from one job list")
    a = p.parse_args()
    if a.claim_dir:
        os.makedirs(a.claim_dir, exist_ok=True)
    for job in json.loads(a.jobs):
        run = os.path.join(a.project, job["name"])
        if os.path.exists(os.path.join(run, "DONE")):
            print("skip", job["name"], flush=True); continue
        if a.claim_dir:  # atomic claim: only one worker trains each job
            try:
                os.close(os.open(os.path.join(a.claim_dir, job["name"]), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            except FileExistsError:
                continue
            print(f"[device {a.device}] claimed", job["name"], flush=True)
        t = time.time()
        YOLO(job.get("model", a.model)).train(
            data=job["data"], imgsz=job["imgsz"], epochs=job.get("epochs", a.epochs), batch=job.get("batch", a.batch),
            workers=a.workers, device=a.device, project=a.project, name=job["name"], exist_ok=True,
            seed=job.get("seed", a.seed), deterministic=True, patience=job.get("patience", a.patience), plots=False, verbose=False, cos_lr=True,
            **job.get("args", {}))
        with open(os.path.join(run, "DONE"), "w") as f:
            f.write(str(round(time.time() - t, 1)))
        print("done", job["name"], round((time.time() - t) / 60, 1), "min", flush=True)


if __name__ == "__main__":
    main()
