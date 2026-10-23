"""Generate the Kaggle kernels (notebook + kernel-metadata.json) from the tested sources in code/."""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "code")
OUT = os.path.join(HERE, "kernels")
USER = "shhussein"


def src(name):
    return open(os.path.join(CODE, name), encoding="utf-8").read()


def nb(cells):
    out = []
    for c in cells:
        kind, text = ("markdown", c[3:]) if c.startswith("#MD") else ("code", c)
        cell = {"cell_type": kind, "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        out.append(cell)
    return {"cells": out, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}


def writefile(name):
    return f"%%writefile {name}\n" + src(name)


def kernel(slug, cells, gpu, datasets=(), kernels=(), internet=True):
    d = os.path.join(OUT, slug); os.makedirs(d, exist_ok=True)
    json.dump(nb(cells), open(os.path.join(d, slug + ".ipynb"), "w", encoding="utf-8"), indent=1)
    meta = {"id": f"{USER}/{slug}", "title": slug, "code_file": slug + ".ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": gpu, "enable_tpu": False,
            "enable_internet": internet, "dataset_sources": list(datasets), "competition_sources": [],
            "kernel_sources": list(kernels), "model_sources": []}
    if gpu:
        meta["machine_shape"] = "NvidiaTeslaT4"
    json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=1)
    print("wrote", d)


SETUP = """
import subprocess, sys, os, json, time, glob, shutil
os.makedirs('/tmp/code', exist_ok=True); os.chdir('/tmp/code')
OUT = '/kaggle/working'
print('free disk GB /tmp:', round(shutil.disk_usage('/tmp').free / 1e9, 1), ' /kaggle/working:', round(shutil.disk_usage(OUT).free / 1e9, 1))
"""

GPU_SETUP = SETUP + """
for attempt in range(4):  # retry: transient network failures on the worker
    if subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics', 'mcos-decoder']).returncode == 0:
        break
    time.sleep(30)
import torch
print('torch', torch.__version__, 'GPUs:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
DEVICES = ','.join(str(i) for i in range(torch.cuda.device_count())) or 'cpu'
from ultralytics import YOLO
for attempt in range(5):  # download the weights once, before parallel training processes start
    try:
        YOLO('yolo11s.pt'); break
    except Exception as e:
        print('weights download failed, retrying', e); time.sleep(30)
"""

# ------------------------------------------------------------------ Anti-UAV prep (CPU, no GPU quota)
AU_STEP = 40
prep_cells = [
    "#MD# SkyFuse - Anti-UAV300 preparation (CPU)\nSamples synchronised IR/RGB frames, builds YOLO sets, crop-verifier and pixel-fusion training sets, "
    "dataset statistics and sample images. Output is consumed by the GPU kernels.",
    SETUP,
    writefile("fusion_core.py"), writefile("antiuav_pipeline.py"), writefile("antiuav_prep.py"),
    """
import antiuav_pipeline as P
DATA = None
for cand in sorted(glob.glob('/kaggle/input/**/', recursive=True), key=len):
    if os.path.isdir(os.path.join(cand, 'test')) and os.path.isdir(os.path.join(cand, 'train')):
        DATA = cand; break
print('DATA', DATA)
r = subprocess.run([sys.executable, 'antiuav_prep.py', '--data', DATA, '--work', '/tmp/skyfuse_au', '--step', '%d'], capture_output=True, text=True)
print(r.stdout[-6000:]); print(r.stderr[-6000:])
assert r.returncode == 0
""" % AU_STEP,
    """
# small metadata files for the offline analysis
for f in ['gt.json', 'meta.json', 'data_stats.json', 'maps_trainval.json']:
    shutil.copy('/tmp/skyfuse_au/prep/' + f, OUT)
# sample images for the paper: the middle sampled frame of every sequence (IR native, RGB at 960 px)
# plus a native-resolution RGB crop around the target
import cv2, numpy as np
gt, meta = P.load_prepared('/tmp/skyfuse_au/prep')
os.makedirs('/tmp/samples', exist_ok=True)
info = {}
for seq, ks in meta['frames_of'].items():
    cands = [k for k in ks if gt['IR'][k] is not None and gt['RGB'][k] is not None] or ks
    k = cands[len(cands) // 2]
    shutil.copy(f'/tmp/skyfuse_au/prep/IR/images/{k}.jpg', f'/tmp/samples/{k}_IR.jpg')
    shutil.copy(f'/tmp/skyfuse_au/prep/RGBs/images/{k}.jpg', f'/tmp/samples/{k}_RGBs.jpg')
    b = gt['RGB'][k]
    if b is not None:
        im = cv2.imread(f'/tmp/skyfuse_au/prep/RGB/images/{k}.jpg'); H, W = im.shape[:2]
        cx, cy = int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2); h = 128
        x0, y0 = int(np.clip(cx - h, 0, W - 2 * h)), int(np.clip(cy - h, 0, H - 2 * h))
        cv2.imwrite(f'/tmp/samples/{k}_RGBcrop.jpg', im[y0:y0 + 2 * h, x0:x0 + 2 * h], [cv2.IMWRITE_JPEG_QUALITY, 95])
        info[k] = {'crop_xy': [x0, y0]}
    bi = gt['IR'][k]
    if bi is not None:
        im = cv2.imread(f'/tmp/skyfuse_au/prep/IR/images/{k}.jpg'); H, W = im.shape[:2]
        cx, cy = int((bi[0] + bi[2]) / 2), int((bi[1] + bi[3]) / 2); h = 48
        x0, y0 = int(np.clip(cx - h, 0, W - 2 * h)), int(np.clip(cy - h, 0, H - 2 * h))
        cv2.imwrite(f'/tmp/samples/{k}_IRcrop.jpg', im[y0:y0 + 2 * h, x0:x0 + 2 * h], [cv2.IMWRITE_JPEG_QUALITY, 95])
        info.setdefault(k, {})['ir_crop_xy'] = [x0, y0]
json.dump(info, open('/tmp/samples/samples.json', 'w'))
shutil.make_archive(OUT + '/samples', 'zip', '/tmp/samples')
print(len(info), 'samples')
""",
    """
# pack the prepared data for the GPU kernels (split into 1.5 GB parts)
r = subprocess.run('tar -C /tmp -cf - skyfuse_au | split -b 1500M - /kaggle/working/antiuav_prep.tar.part_', shell=True)
assert r.returncode == 0
print(subprocess.run('ls -la /kaggle/working', shell=True, capture_output=True, text=True).stdout)
""",
]

# ------------------------------------------------------------------ Anti-UAV GPU, one kernel per seed
def au_gpu_cells(seed, robust):
    return [
        f"#MD# SkyFuse - Anti-UAV300 fusion-level study, seed {seed} (GPU)\nTrains IR, RGB, crop-verifier and pixel-fusion detectors on two GPUs, "
        "saves raw detections and evaluates single sensors, late fusion (6 rules), cascade, pixel-level and hybrid fusion"
        + (", plus robustness to degraded/missing sensors." if robust else "."),
        GPU_SETUP,
        writefile("fusion_core.py"), writefile("antiuav_pipeline.py"), writefile("train_queue.py"), writefile("run_antiuav.py"),
        """
parts = sorted(glob.glob('/kaggle/input/**/antiuav_prep.tar.part_*', recursive=True))
print(parts)
assert parts, 'attach the output of the skyfuse-antiuav-prep kernel'
r = subprocess.run('cat ' + ' '.join(parts) + ' | tar -C /tmp -xf -', shell=True); assert r.returncode == 0
print(os.listdir('/tmp/skyfuse_au'))
""",
        f"""
cmd = [sys.executable, 'run_antiuav.py', '--work', '/tmp/skyfuse_au', '--out', OUT, '--seed', '{seed}', '--devices', DEVICES,
       '--epochs', '30', '--patience', '10', '--model', 'yolo11s.pt']{" + ['--robustness', '--pix_rgb']" if robust else ""}
with open(OUT + '/run_s{seed}.log', 'w') as lf:  # keep the full log in the outputs
    p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
print('exit', p.returncode)
print(open(OUT + '/run_s{seed}.log').read()[-3000:])
print(open(OUT + '/results_antiuav_s{seed}.json').read()[:4000])
""",
    ]


# ------------------------------------------------------------------ Halmstad GPU
H_EPOCHS, H_STEP = 30, 15
q0 = [f"V:clip:640:{H_EPOCHS}:0", f"IR:clip:320:{H_EPOCHS}:0", f"IR:random:320:{H_EPOCHS}:0", f"V:clip:320:{H_EPOCHS}:1",
      f"V:random:320:{H_EPOCHS}:1", f"IR:clip:320:{H_EPOCHS}:2", f"IR:random:320:{H_EPOCHS}:2"]
q1 = [f"IR:clip:640:{H_EPOCHS}:0", f"V:clip:320:{H_EPOCHS}:0", f"V:random:320:{H_EPOCHS}:0", f"IR:clip:320:{H_EPOCHS}:1",
      f"IR:random:320:{H_EPOCHS}:1", f"V:clip:320:{H_EPOCHS}:2", f"V:random:320:{H_EPOCHS}:2"]
halm_cells = [
    "#MD# SkyFuse - Halmstad experiments (GPU)\nLeakage test (clip-level split vs random frame split, with the random-split model also scored on unseen clips), "
    "IR vs visible by class and distance, and input-resolution effects. Two training queues run in parallel on two GPUs.",
    GPU_SETUP,
    writefile("halmstad_prep.py"), writefile("run_halmstad.py"),
    """
if not os.path.exists('Drone-detection-dataset'):
    subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/DroneDetectionThesis/Drone-detection-dataset.git'], check=True)
D = 'Drone-detection-dataset/Data'
os.makedirs('hroot/labels', exist_ok=True); os.makedirs('hroot/videos', exist_ok=True)
for f in glob.glob(D + '/Video_*/*'):
    dst = 'hroot/labels/' if f.endswith('_LABELS.mat') else ('hroot/videos/' if f.endswith('.mp4') else None)
    if dst and not os.path.exists(dst + os.path.basename(f)):
        os.symlink(os.path.abspath(f), dst + os.path.basename(f))
shutil.copy(D + '/Video_dataset_description.xlsx', 'hroot/')
print(len(os.listdir('hroot/labels')), 'label files,', len(os.listdir('hroot/videos')), 'videos')
""",
    f"""
subprocess.run([sys.executable, 'halmstad_prep.py', '--root', 'hroot', '--out', 'hdata', '--step', '{H_STEP}', '--seed', '0'], check=True)
shutil.copy('hdata/prep_meta.json', OUT)
for s in ['IR', 'V']:
    shutil.copy(f'hdata/{{s}}/frames.csv', f'{{OUT}}/frames_{{s}}.csv')
""",
    f"""
Q = [{q0!r}, {q1!r}]
devs = DEVICES.split(',')
base = [sys.executable, 'run_halmstad.py', '--data', 'hdata', '--out', 'hres', '--model', 'yolo11s.pt', '--budget_h', '4.3']
if len(devs) >= 2:
    procs = [subprocess.Popen(base + ['--device', devs[i], '--configs', *Q[i]]) for i in range(2)]
    print([p.wait() for p in procs])
else:
    subprocess.run(base + ['--device', devs[0], '--configs', *Q[0], *Q[1]])
""",
    """
res = {os.path.basename(f)[:-5]: json.load(open(f)) for f in sorted(glob.glob('hres/*.json'))}
json.dump({'halmstad': res, 'prep_meta': json.load(open('hdata/prep_meta.json'))}, open(OUT + '/results_halmstad.json', 'w'), indent=1)
print('saved', len(res), 'runs')
""",
]

def ablation_cells(variants, epochs=30, extra_args=()):
    return [
        "#MD# SkyFuse - single-detector ablation on Anti-UAV300 (GPU)\nOne change at a time (seed 0): " + ", ".join(variants) + ".",
        GPU_SETUP.replace("YOLO('yolo11s.pt'); break", "YOLO('yolo11s.pt'); YOLO('yolo11m.pt'); break"),
        writefile("fusion_core.py"), writefile("antiuav_pipeline.py"), writefile("train_queue.py"), writefile("run_ablation.py"),
        """
parts = sorted(glob.glob('/kaggle/input/**/antiuav_prep.tar.part_*', recursive=True))
print(parts)
assert parts, 'attach the output of the skyfuse-antiuav-prep kernel'
r = subprocess.run('cat ' + ' '.join(parts) + ' | tar -C /tmp -xf -', shell=True); assert r.returncode == 0
extra = glob.glob('/kaggle/input/**/a410x.tar', recursive=True)  # optional extra thermal scenes (Anti-UAV410)
print('a410x:', extra)
if extra:
    subprocess.run(['tar', '-C', '/tmp', '-xf', extra[0]], check=True)
""",
        f"""
cmd = [sys.executable, 'run_ablation.py', '--work', '/tmp/skyfuse_au', '--out', OUT, '--devices', DEVICES, '--epochs', '{epochs}',
       '--variants', {", ".join(repr(v) for v in variants)}{"".join(", " + repr(x) for x in extra_args)}]
with open(OUT + '/run_ablation.log', 'w') as lf:
    p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
print('exit', p.returncode)
print(open(OUT + '/run_ablation.log').read()[-3000:])
""",
    ]


a410_cells = [
    "#MD# SkyFuse - extra thermal sequences from Anti-UAV410 (CPU)\nThe 100 sequences that are not part of Anti-UAV300, excluding the 10 Anti-UAV300 test sessions; every 40th frame.",
    SETUP, writefile("fusion_core.py"), writefile("antiuav_pipeline.py"), writefile("a410_extra_prep.py"),
    "%%writefile au300_names.json\n" + open(os.path.join(CODE, "au300_names.json")).read(),
    """
DATA = None
for cand in sorted(glob.glob('/kaggle/input/**/', recursive=True), key=len):
    if glob.glob(os.path.join(cand, '*', '*', 'IR_label.json')):
        DATA = cand; break
print('DATA', DATA)
r = subprocess.run([sys.executable, 'a410_extra_prep.py', '--data', DATA, '--au300', 'au300_names.json', '--out', '/tmp/skyfuse_au/a410x', '--step', '40'], capture_output=True, text=True)
print(r.stdout[-3000:]); print(r.stderr[-3000:]); assert r.returncode == 0
shutil.copy('/tmp/skyfuse_au/a410x/a410x_info.json', OUT)
r = subprocess.run('tar -C /tmp -cf /kaggle/working/a410x.tar skyfuse_au/a410x', shell=True); assert r.returncode == 0
print(subprocess.run('ls -la /kaggle/working', shell=True, capture_output=True, text=True).stdout)
""",
]


if __name__ == "__main__":
    kernel("skyfuse-a410x-prep", a410_cells, gpu=False, datasets=["huseyincavus/anti-uav410"], internet=False)
    kernel("skyfuse-ablation-ir", ablation_cells(["ir_base", "ir_fe", "ir_aug", "ir_sin", "ir_a410"]), gpu=True, kernels=[f"{USER}/skyfuse-antiuav-prep", f"{USER}/skyfuse-a410x-prep"])
    kernel("skyfuse-ablation-combo", ablation_cells(["ir_combo"]), gpu=True, kernels=[f"{USER}/skyfuse-antiuav-prep", f"{USER}/skyfuse-a410x-prep"])
    kernel("skyfuse-antiuav-prep", prep_cells, gpu=False, datasets=["mm1991/anti-uav-dataset-300"], internet=False)
    for s in (0, 1, 2):
        kernel(f"skyfuse-antiuav-s{s}", au_gpu_cells(s, s == 0), gpu=True, kernels=[f"{USER}/skyfuse-antiuav-prep"])
    kernel("skyfuse-halmstad", halm_cells, gpu=True)
