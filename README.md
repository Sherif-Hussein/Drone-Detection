# Drone detection with unaligned thermal and visible cameras

Code and results for the paper:

> S. Hussein, *Fusion of unaligned thermal and visible cameras for drone detection: a controlled analysis and lightweight detectors for CPU-only stations* (submitted to the International Journal of Intelligent Robotics and Applications).

The study asks two questions. Does combining a thermal camera and a visible (RGB) camera that are not aligned really improve drone detection? And how small can the detectors be, so that a station can run on a laptop CPU? It compares late, cascade, pixel-level and hybrid fusion with identical YOLO11 detectors, three seeds and session-level statistics on the public Anti-UAV300 dataset, audits the Halmstad dataset, and tests YOLO11-nano detectors, an extra small-object head, different input sizes and a detector shared by both cameras.

## Contents

| Folder | What it holds |
|---|---|
| `code/` | Data preparation, training queues, fusion (`fusion_core.py`, `antiuav_pipeline.py`), evaluation and statistics (`analyze_antiuav*.py`, `session_permutation.py`, `loso_cv.py`), lightweight-detector trials (`rgb_small_trial.py`, `nano3_compare.py`, `nano_fusion.py`, `joint_compare.py`), cost measurement (`cost_bench.py`, `e2e_bench.py`), the leakage check (`leakage_check.py`) and the YOLO11 model file with an extra stride-4 head (`yolo11-p2.yaml`) |
| `kaggle_kernels/` | Scripts that build the Kaggle notebooks used for data preparation and GPU training |
| `figures_code/` | Scripts that draw the figures of the paper and the supplement |
| `results/` | Result files: metrics (JSON), saved test detections (pickle), training curves, permutation tests, laptop cost measurements and logs |

The datasets, the trained weights and the manuscript are not included.

## Datasets

- Anti-UAV300: https://github.com/ZhaoJ9014/Anti-UAV (also used: the additional thermal sequences of Anti-UAV410)
- Halmstad drone dataset: https://doi.org/10.5281/zenodo.5500576

## Requirements

Python 3.11 and the packages in `requirements.txt`. Training used Kaggle notebooks with NVIDIA T4 GPUs; all fusion, statistics and cost measurements run on a CPU from the saved detections.

## Reproducing the main analyses

1. Build and run the Kaggle notebooks with the scripts in `kaggle_kernels/` (data preparation runs on a CPU, training on a GPU). Each notebook writes its detections and metrics to its output folder.
2. Copy the outputs into `results/` (the files used in the paper are already there).
3. Run the analyses on a CPU, for example:

```bash
python code/session_permutation.py results/antiuav results/runs results/runs/session_permutation.json
python code/nano3_compare.py results/antiuav results/runs results/nano3_trial results/nano3_trial/compare.json
python code/nano_fusion.py results/antiuav results/runs results/nano3_trial results/nano3_trial/nano_fusion_last.json last
python code/cost_bench.py --cfg yolo11n.yaml --shape 512 640 --tag ir_n640 --out results/runs/cost_bench_pc.jsonl
```

4. Draw the figures with the scripts in `figures_code/`.

## Citation

If you use this code or these results, please cite the paper above (full reference to be added on publication); GitHub's "Cite this repository" button gives the details from `CITATION.cff`. A Zenodo DOI for this repository will be added before publication.

## License

The code is released under the MIT License, copyright (c) 2026 Sherif Hussein. See `LICENSE`.
