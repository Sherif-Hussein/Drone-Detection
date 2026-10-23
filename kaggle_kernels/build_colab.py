"""Build the Google Colab notebook that continues the enhancement experiments on a Colab GPU.
Data come from the Kaggle outputs of skyfuse-antiuav-prep and skyfuse-a410x-prep; results go to Google Drive."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "code")
OUT = os.path.join(HERE, "colab", "SkyFuse_enhancement_colab.ipynb")


def src(n):
    return open(os.path.join(CODE, n), encoding="utf-8").read()


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(keepends=True)}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n").splitlines(keepends=True)}


cells = [
    md("""
# SkyFuse — enhanced thermal detector (Google Colab)

**Before you start:** `Runtime → Change runtime type → T4 GPU`, then run the cells from top to bottom.

* Part A (≈1.5 h): ablations **E7** (extra Anti-UAV410 scenes + fast CLAHE/top-hat channels) and **E8** (same with the
  original elliptical top-hat), seed 0, session-disjoint selection — same protocol as E0–E6.
* Part B (≈2.5 h): the final configuration on the **official split**, seeds 0, 1, 2 (one cell per seed).
* Every finished run is written immediately to Google Drive (`MyDrive/skyfuse_colab/`). If Colab disconnects, re-run the
  setup cells and continue with the next unfinished run — finished runs are skipped.
* If the Drive sign-in fails ("credential propagation was unsuccessful"), the notebook carries on and keeps results on the
  Colab disk (`/content/skyfuse_out`); then run `download_results()` after each run, because that disk is wiped when
  the session ends.
* Afterwards, download the files from `MyDrive/skyfuse_colab/` and copy them into
  `C:\\Sherif\\Drone Detection\\work\\paper\\data\\colab\\`.
"""),
    code("""
import os, sys, subprocess, getpass, glob, json, time, shutil
print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'], capture_output=True, text=True).stdout)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics', 'kaggle'], check=True)
from google.colab import drive
try:
    drive.mount('/content/drive')
    OUT = '/content/drive/MyDrive/skyfuse_colab'
except Exception as e:  # Drive authorisation failed or was declined: keep results on the Colab disk and download them
    print('Google Drive not mounted (' + str(e) + ').')
    print('Results will be kept in /content/skyfuse_out -- run the download cell after each part.')
    OUT = '/content/skyfuse_out'
os.makedirs(OUT, exist_ok=True)
os.makedirs('/content/code', exist_ok=True); os.chdir('/content/code')
print('outputs ->', OUT)
"""),
    md("### Kaggle token (typed here, not stored in the notebook)"),
    code("""
os.environ['KAGGLE_API_TOKEN'] = getpass.getpass('Kaggle API token (KGAT_...): ').strip()
r = subprocess.run(['kaggle', 'kernels', 'status', 'shhussein/skyfuse-antiuav-prep'], capture_output=True, text=True)
print(r.stdout or r.stderr)
"""),
    md("### Download the prepared data from Kaggle (≈1.6 GB) and unpack to /tmp/skyfuse_au"),
    code("""
if not os.path.exists('/tmp/skyfuse_au/prep/meta.json'):
    os.makedirs('/content/dl', exist_ok=True)
    for k in ('shhussein/skyfuse-antiuav-prep', 'shhussein/skyfuse-a410x-prep'):
        r = subprocess.run(['kaggle', 'kernels', 'output', k, '-p', '/content/dl', '--file-pattern', r'(tar|part_)'],
                           capture_output=True, text=True); print(k, r.returncode, (r.stderr or '')[-300:])
    parts = sorted(glob.glob('/content/dl/antiuav_prep.tar.part_*')); print(parts)
    assert parts, 'prep data not downloaded'
    subprocess.run('cat ' + ' '.join(parts) + ' | tar -C /tmp -xf -', shell=True, check=True)
    subprocess.run(['tar', '-C', '/tmp', '-xf', '/content/dl/a410x.tar'], check=True)
    shutil.rmtree('/content/dl', ignore_errors=True)
print(os.listdir('/tmp/skyfuse_au'), len(open('/tmp/skyfuse_au/a410x/extra_train.txt').read().split()))
from ultralytics import YOLO
YOLO('yolo11s.pt')
"""),
]
for n in ("fusion_core.py", "antiuav_pipeline.py", "train_queue.py", "run_ablation.py"):
    cells.append(code(f"%%writefile {n}\n" + src(n)))
cells.append(code("""
def run(variant, seed, select_sessions):
    tag = f'{variant}_s{seed}_' + ('sd' if select_sessions else 'official')
    dst = os.path.join(OUT, tag); os.makedirs(dst, exist_ok=True)
    if os.path.exists(os.path.join(dst, f'ablation_s{seed}.json')):
        print('already done:', tag); return
    cmd = [sys.executable, 'run_ablation.py', '--work', '/tmp/skyfuse_au', '--out', dst, '--variants', variant,
           '--seeds', str(seed), '--devices', '0', '--epochs', '30', '--workers', '2', '--select_sessions', str(select_sessions)]
    t = time.time()
    with open(os.path.join(dst, 'run.log'), 'w') as lf:
        p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    print(tag, 'exit', p.returncode, round((time.time() - t) / 60, 1), 'min')
    print(open(os.path.join(dst, 'run.log')).read()[-1500:])
"""))
cells.append(code("""
def download_results():
    # zip everything finished so far (weights included) and send it to the browser
    z = shutil.make_archive('/content/skyfuse_results', 'zip', OUT)
    from google.colab import files
    files.download(z)
"""))
cells += [
    md("## Part A — E7 and E8 (seed 0, session-disjoint selection; ≈40–45 min each)"),
    code("run('ir_a410fe', 0, 6)      # E7: extra scenes + fast CLAHE/top-hat channels, no extra augmentation"),
    code("run('ir_a410fe_ell', 0, 6)  # E8: extra scenes + elliptical top-hat channels, no extra augmentation"),
    code("download_results()   # needed if Drive is not mounted; harmless otherwise"),
    md("""
## Part B — final configuration on the official split, seeds 0, 1, 2 (≈45 min each)

Send the Part A files to Claude first; it will tell you which variant to put in `FINAL`
(`ir_a410`, `ir_a410fe` or `ir_a410fe_ell`).
"""),
    code("FINAL = 'ir_a410'   # <- set after Part A"),
    code("run(FINAL, 0, 0)"),
    code("run(FINAL, 1, 0)"),
    code("run(FINAL, 2, 0)"),
    code("download_results()"),
    md("Done. Download `MyDrive/skyfuse_colab/` (all sub-folders) and copy it to `work\\paper\\data\\colab\\`."),
]
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(nb, open(OUT, "w", encoding="utf-8"), indent=1)
print("wrote", OUT)
