"""GPU Kaggle kernels: small-drone trials. RGB: YOLO11n vs YOLO11s, 960 vs 1280 px, extra P2 head.
Thermal: YOLO11n (GPU-trained) and YOLO11n/s with an extra P2 head at 640 px.

python build_rgb_trial_kernel.py [rgb|ir]
"""
import sys
from build_kernels import kernel, writefile, GPU_SETUP, USER

UNPACK = """
parts = sorted(glob.glob('/kaggle/input/**/antiuav_prep.tar.part_*', recursive=True))
print('prep parts', parts)
r = subprocess.run('cat ' + ' '.join(parts) + ' | tar -C /tmp -xf -', shell=True); assert r.returncode == 0
print(os.listdir('/tmp/skyfuse_au/prep'))
"""


def cells(title, text, modality, log):
    return [
        f"#MD# {title}\n{text}",
        GPU_SETUP + """
for attempt in range(5):
    try:
        YOLO('yolo11n.pt'); break
    except Exception as e:
        print('weights download failed, retrying', e); time.sleep(30)
""",
        writefile("fusion_core.py"), writefile("antiuav_pipeline.py"), writefile("analyze_antiuav.py"),
        writefile("train_queue.py"), writefile("rgb_small_trial.py"), writefile("yolo11-p2.yaml"),
        UNPACK,
        f"""
with open(OUT + '/{log}', 'w') as lf:
    r = subprocess.run([sys.executable, 'rgb_small_trial.py', '--work', '/tmp/skyfuse_au', '--out', OUT, '--devices', DEVICES,
                        '--modality', '{modality}'], stdout=lf, stderr=subprocess.STDOUT)
print('exit', r.returncode); print(open(OUT + '/{log}').read()[-6000:])
for name in os.listdir(OUT + '/runs'):
    for f in glob.glob(OUT + f'/runs/{{name}}/weights/*.pt'):
        os.remove(f)  # best weights are kept in OUT/weights
""",
    ]


rgb_cells = cells("SkyFuse - RGB small-drone trial (GPU)",
                  "Seed-0 RGB detectors with the main-study recipe: YOLO11n at 960 and 1280 px, YOLO11n and YOLO11s with an "
                  "extra P2 (stride-4) head at 960 px, and YOLO11s at 1280 px; scored on the official test split.",
                  "RGB", "rgb_small_trial.log")
ir_cells = cells("SkyFuse - thermal small-drone trial (GPU)",
                 "Seed-0 thermal detectors with the main-study recipe at 640 px: YOLO11s, YOLO11n, and YOLO11n and YOLO11s "
                 "with an extra P2 (stride-4) head; scored on the official test split. The thermal test frames are saved "
                 "for the offline two-way cueing experiment.",
                 "IR", "ir_small_trial.log") + ["""
# thermal test frames for the offline two-way cueing experiment (re-detection in cued thermal crops)
import zipfile
with zipfile.ZipFile(OUT + '/ir_test_images.zip', 'w', zipfile.ZIP_STORED) as z:
    for f in sorted(glob.glob('/tmp/skyfuse_au/prep/IR/images/test_*.jpg')):
        z.write(f, os.path.basename(f))
print('zipped', len(z.namelist()) if False else len(glob.glob('/tmp/skyfuse_au/prep/IR/images/test_*.jpg')), 'frames,',
      round(os.path.getsize(OUT + '/ir_test_images.zip') / 2**20, 1), 'MB')
"""]

joint_cells = cells("SkyFuse - joint thermal+RGB detector (GPU)",
                    "One YOLO11n trained on thermal and 960-px RGB frames together (shared weights), seed 0, main-study "
                    "recipe; scored on each camera separately.",
                    "JOINT", "joint_trial.log")

jointpad_cells = cells("SkyFuse - joint thermal+RGB detector, thermal at native scale (GPU)",
                       "One YOLO11n trained on thermal and 960-px RGB frames together; thermal frames are padded (not "
                       "enlarged) onto a 960x768 canvas so thermal drones keep their size. Seed 0, main-study recipe; "
                       "scored on each camera separately.",
                       "JOINTPAD", "jointpad_trial.log")

nano3_cells = cells("SkyFuse - YOLO11-nano, three seeds per camera (GPU)",
                    "Thermal (640 px) and RGB (960 px) YOLO11-nano, seeds 0-2, main-study recipe; best-validation and last-epoch "
                    "checkpoints both scored on the official test split.", "NANO3", "nano3_trial.log")
lowres_cells = cells("SkyFuse - 320-pixel single-camera baselines (GPU)",
                     "YOLO11-nano and YOLOv5s (Ultralytics yolov5su) at 320 px on each camera, seed 0, main-study recipe, "
                     "for a like-for-like comparison with published 320-px baselines.", "LOWRES", "lowres_trial.log")

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "rgb"
    src = [f"{USER}/skyfuse-antiuav-prep"]
    if which == "rgb":
        kernel("skyfuse-rgb-small-trial", rgb_cells, gpu=True, kernels=src, internet=True)
    elif which == "ir":
        kernel("skyfuse-ir-small-trial", ir_cells, gpu=True, kernels=src, internet=True)
    elif which == "nano3":
        kernel("skyfuse-nano3-trial", nano3_cells, gpu=True, kernels=src, internet=True)
    elif which == "lowres":
        kernel("skyfuse-lowres-trial", lowres_cells, gpu=True, kernels=src, internet=True)
    elif which == "joint":
        kernel("skyfuse-joint-trial", joint_cells, gpu=True, kernels=src, internet=True)
    else:
        kernel("skyfuse-jointpad-trial", jointpad_cells, gpu=True, kernels=src, internet=True)
