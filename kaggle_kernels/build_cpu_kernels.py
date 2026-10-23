"""CPU-only Kaggle kernels (no GPU quota): Anti-UAV410 leakage check and a YOLO11n CPU training trial."""
from build_kernels import kernel, writefile, SETUP, USER

UNPACK = """
parts = sorted(glob.glob('/kaggle/input/**/antiuav_prep.tar.part_*', recursive=True))
print('prep parts', parts)
r = subprocess.run('cat ' + ' '.join(parts) + ' | tar -C /tmp -xf -', shell=True); assert r.returncode == 0
extra = glob.glob('/kaggle/input/**/a410x.tar', recursive=True)
print('a410x', extra)
subprocess.run(['tar', '-C', '/tmp', '-xf', extra[0]], check=True)
print(os.listdir('/tmp/skyfuse_au'), len(os.listdir('/tmp/skyfuse_au/a410x/images')))
"""

leak_cells = [
    "#MD# SkyFuse - Anti-UAV410 leakage check (CPU)\nNear-duplicate and same-scene check of the 100 extra Anti-UAV410 "
    "thermal sequences against the Anti-UAV300 test frames, with same-session (positive) and disjoint-session (negative) controls.",
    SETUP, writefile("leakage_check.py"), UNPACK,
    """
import torch; print('torch', torch.__version__, 'threads', os.cpu_count())
with open(OUT + '/leakage_check.log', 'w') as lf:
    r = subprocess.run([sys.executable, 'leakage_check.py', '--work', '/tmp/skyfuse_au', '--out', OUT], stdout=lf, stderr=subprocess.STDOUT)
print('exit', r.returncode); print(open(OUT + '/leakage_check.log').read()[-6000:])
""",
]

trial_cells = [
    "#MD# SkyFuse - YOLO11n thermal detector trained on CPU only (trial)\nTrains on the prepared Anti-UAV300 IR set with a "
    "10-hour cap and scores the official test split with the paper's evaluator; logs hours per epoch.",
    SETUP + """
for attempt in range(4):
    if subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics']).returncode == 0:
        break
    time.sleep(30)
from ultralytics import YOLO
for attempt in range(5):
    try:
        YOLO('yolo11n.pt'); break
    except Exception as e:
        print('weights download failed, retrying', e); time.sleep(30)
""",
    writefile("fusion_core.py"), writefile("antiuav_pipeline.py"), writefile("analyze_antiuav.py"), writefile("cpu_trial.py"),
    UNPACK,
    """
with open(OUT + '/cpu_trial.log', 'w') as lf:
    r = subprocess.run([sys.executable, 'cpu_trial.py', '--work', '/tmp/skyfuse_au', '--out', OUT, '--model', 'yolo11n.pt',
                        '--epochs', '30', '--hours', '10', '--imgsz', '640', '--workers', '4'], stdout=lf, stderr=subprocess.STDOUT)
print('exit', r.returncode); print(open(OUT + '/cpu_trial.log').read()[-4000:])
""",
]

if __name__ == "__main__":
    src = [f"{USER}/skyfuse-antiuav-prep", f"{USER}/skyfuse-a410x-prep"]
    kernel("skyfuse-leakage-check", leak_cells, gpu=False, kernels=src, internet=True)
    kernel("skyfuse-yolo11n-cpu-trial", trial_cells, gpu=False, kernels=src, internet=True)
