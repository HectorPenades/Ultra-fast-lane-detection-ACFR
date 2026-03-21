# Ultra-Fast-Lane-Detection-V2 — ACFR Fork

Fork of the PyTorch implementation of [Ultra Fast Deep Lane Detection with Hybrid Anchor Driven Ordinal Classification (UFLDv2)](https://arxiv.org/abs/2206.07389), adapted to work with the **CULane\_cropped\_left** dataset from ACFR.

The original repository is [cfzd/Ultra-Fast-Lane-Detection-V2](https://github.com/cfzd/Ultra-Fast-Lane-Detection-V2). For a full list of changes made to the original code, see [CHANGES.md](./CHANGES.md).

---

## Table of Contents

1. [Dataset](#dataset)
2. [Installation](#installation)
3. [Data preparation](#data-preparation)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Inference](#inference)
7. [Configuration reference](#configuration-reference)
8. [TensorBoard](#tensorboard)

---

## Dataset

This fork is designed for **CULane\_cropped\_left**, a 2-lane dataset derived from CULane for the ACFR vehicle front camera. Key properties:

| Property | Value |
|---|---|
| Resolution | 1640 × 590 px |
| Lanes | 2 (left and right) |
| Train images | 133 235 |
| Test images | 10 770 |
| Annotation format | CULane-style `.lines.txt` + PNG masks |

The dataset README at the dataset root covers directory structure, list file formats, mask format, and cache generation.

---

## Installation

### 1. Clone

```bash
git clone https://github.com/HectorPenades/Ultra-fast-lane-detection-ACFR
cd Ultra-fast-lane-detection-ACFR
```

### 2. Conda environment

```bash
conda create -n lane python=3.9 -y
conda activate lane
```

### 3. PyTorch

Install PyTorch matching your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/). Example for CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Python dependencies

```bash
pip install -r requirements.txt
```

### 5. NVIDIA DALI

```bash
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist \
    --upgrade nvidia-dali-cuda110
```

Adjust the `cuda110` suffix to match your CUDA version (`cuda120`, etc.).

### 6. my\_interp

The DALI pipeline uses `my_interp` for lane coordinate interpolation. A pure-PyTorch fallback is included so training works without a CUDA compiler:

```bash
# Try to compile the CUDA extension (optional, faster)
cd my_interp
sh build.sh
cd ..

# Verify — works with or without the compiled extension
python -c "import my_interp; print('run' in dir(my_interp))"
# Expected output: True
```

If `build.sh` fails, the Python fallback in `my_interp/__init__.py` is used automatically. No manual action needed.

### 7. CULane evaluation binary

The evaluator binary must be compiled before running evaluation. It requires OpenCV with `highgui` and `imgcodecs`. The simplest approach is to use the OpenCV bundled in a conda environment:

```bash
# Point to a conda environment that has opencv installed (e.g. via conda-forge)
OPENCV_PKG=/path/to/conda/envs/your-env

cd evaluation/culane
g++ -std=c++11 -fopenmp -DCPU_ONLY \
    -Iinclude \
    -I${OPENCV_PKG}/include/opencv4 \
    $(find src/ -name "*.cpp") \
    -o evaluate \
    -L${OPENCV_PKG}/lib \
    -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs \
    -Wl,-rpath,${OPENCV_PKG}/lib \
    -fopenmp
cd ../..

# Verify
./evaluation/culane/evaluate --help
```

> The binary is gitignored. It must be recompiled after cloning on a new machine.

---

## Data preparation

### Set dataset root

Edit `configs/culane_cropped_res34.py` to point to your local copy of CULane\_cropped\_left:

```python
data_root = '/path/to/CULane_cropped_left'
```

### Generate annotation caches

The DALI training pipeline reads lane coordinates from JSON caches, not directly from PNG masks. The caches must be generated once (or after any change to the masks):

```bash
cd /path/to/CULane_cropped_left

# Training cache (~133k images)
python scripts/4_generate_cache.py \
    --data_root . \
    --list list/train.txt \
    --mask_dir laneseg_label_w16 \
    --output culane_anno_cache.json \
    --workers 8

# Test cache (~10k images)
python scripts/4_generate_cache.py \
    --data_root . \
    --list list/test.txt \
    --mask_dir laneseg_label_w16_test \
    --output culane_anno_cache_test.json \
    --workers 8
```

### Optional: fast evaluation subset

To evaluate on a random 20 % subset of the test set instead of all 10 770 images, generate `list/test_mini.txt` once:

```bash
cd /path/to/CULane_cropped_left
python - << 'EOF'
import random
random.seed(42)
with open('list/test.txt') as f:
    lines = f.readlines()
mini = sorted(random.sample(lines, k=int(len(lines) * 0.20)))
with open('list/test_mini.txt', 'w') as f:
    f.writelines(mini)
print(f'Written {len(mini)} lines to list/test_mini.txt')
EOF
```

Then set `test_list = 'list/test_mini.txt'` in the config to use it.

### Optional: train on the test domain

To create a training split from the test images (keeping `test_mini.txt` as held-out evaluation):

```bash
cd /path/to/CULane_cropped_left
python - << 'EOF'
import json, os

with open('list/test_mini.txt') as f:
    mini_set = set(l.strip() for l in f if l.strip())
with open('list/test.txt') as f:
    all_test = [l.strip() for l in f if l.strip()]
with open('culane_anno_cache_test.json') as f:
    cache = json.load(f)

cache_norm = {k.replace('\\','/').lstrip('/'): v for k, v in cache.items()}
train_from_test = [p for p in all_test if p not in mini_set]

lines = []
for img_path in train_from_test:
    parts = img_path.lstrip('/').split('/')
    driver = parts[0]
    fname = os.path.splitext(parts[-1])[0] + '.png'
    mask_path = f'/laneseg_label_w16_test/{driver}/{fname}'
    lane_data = cache_norm.get(img_path.lstrip('/'))
    if lane_data:
        l1 = int(any(pt[0] > -9999 for pt in lane_data[0]))
        l2 = int(any(pt[0] > -9999 for pt in lane_data[1]))
    else:
        l1, l2 = 0, 0
    lines.append(f'{img_path} {mask_path} {l1} {l2}')

with open('list/train_gt_from_test.txt', 'w') as f:
    f.write('\n'.join(lines) + '\n')
print(f'Written {len(lines)} lines to list/train_gt_from_test.txt')
EOF
```

Then set in the config:
```python
train_list = 'list/train_gt_from_test.txt'
anno_cache = 'culane_anno_cache_test.json'
test_list  = 'list/test_mini.txt'
```

---

## Training

All commands are run from the repository root.

### Train from scratch

```bash
python train.py configs/culane_cropped_res34.py --num_workers 8
```

### Resume from checkpoint

```bash
python train.py configs/culane_cropped_res34.py \
    --resume /path/to/logs/TIMESTAMP/checkpoints/model_best.pth \
    --num_workers 8
```

### Fine-tune from an existing checkpoint

```bash
python train.py configs/culane_cropped_res34.py \
    --finetune /path/to/model_best.pth \
    --learning_rate 0.001 \
    --epoch 20 \
    --num_workers 8
```

### Quick sanity check (1 epoch, no augmentations)

```bash
python train.py configs/culane_cropped_res34.py \
    --epoch 1 \
    --use_augmentations False \
    --num_workers 4
```

### CLI flags

| Flag | Description |
|---|---|
| `--num_workers N` | DALI data loader threads |
| `--batch_size N` | Override batch size from config |
| `--epoch N` | Override number of epochs |
| `--learning_rate F` | Override learning rate |
| `--log_path /path/` | Directory for logs and checkpoints |
| `--use_augmentations False` | Disable geometric augmentations |
| `--vis_interval N` | Steps between TensorBoard image logs |
| `--eval_only` | Run evaluation only (requires `--test_model`) |
| `--test_model /path/` | Checkpoint to evaluate with `--eval_only` |

### Training outputs

Logs and checkpoints are saved under `cfg.log_path`:

```
logs/
└── YYYYMMDD_HHMMSS_lr_5e-03_b_32_culane_cropped/
    ├── checkpoints/
    │   └── model_best.pth          # Best model by F(IoU=0.5, margin=30)
    ├── eval_history.jsonl          # Per-epoch evaluation results
    ├── culane_eval_tmp_eval_results.json
    └── events.out.tfevents.*       # TensorBoard log
```

---

## Evaluation

### Evaluate a checkpoint

```bash
python train.py configs/culane_cropped_res34.py \
    --eval_only \
    --test_model /path/to/model_best.pth \
    --num_workers 4
```

The evaluator runs the CULane binary for IoU ∈ {0.3, 0.4, 0.5, 0.6} × margin ∈ {30, 40, 50, 60} px and prints the total F-measure for each combination.

### Evaluation outputs

| File | Contents |
|---|---|
| `<test_work_dir>/culane_eval_tmp_eval_results.json` | Full results JSON, keys like `"0.5_m30"` |
| `<test_work_dir>/txt/out_single_*.txt` | Raw evaluator output per IoU/margin |

---

## Inference

`infer.py` runs a trained checkpoint on one image or a folder of images and saves annotated results. It does not require a test list or annotation cache.

### Usage

```bash
# Single image
python infer.py --model /path/to/model_best.pth --input image.jpg

# Folder of images (walks subdirectories)
python infer.py --model /path/to/model_best.pth --input /path/to/images/

# Custom output directory + save .lines.txt alongside each result
python infer.py --model /path/to/model_best.pth \
                --input /path/to/images/ \
                --output /path/to/results/ \
                --save_txt
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` / `-m` | required | Path to checkpoint `.pth` |
| `--input` / `-i` | required | Image file or folder |
| `--output` / `-o` | `inference_output/` | Output directory |
| `--config` / `-c` | `configs/culane_cropped_res34.py` | Config file |
| `--save_txt` | off | Also save `.lines.txt` with raw coordinates |

### Outputs

For each input image `path/to/img.jpg` the script writes:

- `<output>/path/to/img_lanes.jpg` — original image with lanes drawn (green = left, red = right)
- `<output>/path/to/img_lanes.lines.txt` — one lane per line, `x1 y1 x2 y2 ...` in original pixel coordinates (only with `--save_txt`)

The input directory structure is mirrored in the output directory.

---

## Configuration reference

All options are in `configs/culane_cropped_res34.py`.

### Dataset

```python
dataset   = 'CULane_cropped'
data_root = '/path/to/CULane_cropped_left'
```

### Training data

```python
# None → default: list/train_gt.txt + culane_anno_cache.json (133k images)
train_list = None
anno_cache = None

# Example: train on test-domain images (80% of test set, no overlap with test_mini.txt)
# train_list = 'list/train_gt_from_test.txt'
# anno_cache = 'culane_anno_cache_test.json'
```

### Test data

```python
# None → list/test.txt (10 770 images, full evaluation, ~25 min)
# Fast subset → 'list/test_mini.txt' (2 154 images, random 20%, seed=42, ~5 min)
test_list = None
```

### Model

```python
backbone = '34'       # ResNet backbone: '18', '34', '50', '101'
use_aux  = False      # Auxiliary segmentation head — requires mask labels in train_gt.txt
fc_norm  = True
```

### Loss weights

```python
sim_loss_w   = 0.0    # Similarity loss: penalises abrupt X jumps between adjacent row anchors
shp_loss_w   = 0.0    # Shape/variance loss
mean_loss_w  = 0.05   # Existence prediction loss weight (applied to both row and col)
```

`sim_loss_w > 0` tends to reduce false positives by enforcing geometrically smooth lane predictions. `mean_loss_w` controls how strongly the model is penalised for predicting a lane in an anchor where there is none.

### Grid

```python
num_row      = 72    # Row anchors (Y): 590 / 72 = 8.2 px/step
num_col      = 81    # Col anchors (X): 1640 / 81 = 20.3 px/step
num_cell_row = 200   # X bins for row predictions: 1640 / 200 = 8.2 px/bin
num_cell_col = 100   # Y bins for col predictions: 590 / 100 = 5.9 px/bin
griding_num  = 200   # Must equal num_cell_row
```

### Detection filter

Applied at evaluation time to suppress short detections before running the evaluator binary.

```python
# None → original thresholds: row >50% of anchors, col >25% of anchors
# GT lanes typically span >90% of anchors; values of 0.5–0.7 are a reasonable range.
min_row_frac = None   # e.g. 0.6 → requires >43 of 72 row anchors to be valid
min_col_frac = None   # e.g. 0.35 → requires >28 of 81 col anchors to be valid
```

---

## TensorBoard

```bash
tensorboard --logdir /path/to/logs
```

Key scalars:

| Tag | Description |
|---|---|
| `train/loss` | Total training loss |
| `train/top1` / `top2` / `top3` | Row localisation accuracy (±1/2/3 bins over GT anchors) |
| `train/col_top1` / `col_top2` / `col_top3` | Col localisation accuracy |
| `train/ext_row` / `ext_col` | Lane existence prediction accuracy |
| `CuEval/total` | F-measure at IoU=0.5, margin=30 (criterion for saving best model) |
| `eval/input_with_pred` | Model predictions overlaid on input image |
| `eval/input_with_lines` | Generated `.lines.txt` overlaid on input (what the evaluator sees) |
