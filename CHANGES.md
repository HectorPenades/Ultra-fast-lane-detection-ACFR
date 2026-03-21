# Changes from the original UFLDv2

This file documents all modifications made to [cfzd/Ultra-Fast-Lane-Detection-V2](https://github.com/cfzd/Ultra-Fast-Lane-Detection-V2) in this fork. The goal is to make it possible to reproduce the setup starting from the original repository.

---

## 1. New dataset: `CULane_cropped` (`dataset = 'CULane_cropped'`)

The original code supports `CULane`, `Tusimple`, and `CurveLanes`. Support for a new dataset identifier `CULane_cropped` was added throughout the codebase. All changes are additive (existing datasets are unaffected).

### `utils/common.py`

- `merge_config()`: added `elif cfg.dataset == 'CULane_cropped':` branch that sets `row_anchor = linspace(0, 1, num_row)` and `col_anchor = linspace(0, 1, num_col)` (full image, no sky crop).
- `get_train_loader()`: added `elif cfg.dataset == 'CULane_cropped':` branch that reads `train_list` and `anno_cache` from config (see §4).

### `data/dataloader.py`

- `get_test_loader()`: changed `if dataset == 'CULane':` to `if dataset in ('CULane', 'CULane_cropped'):` so the test loader is constructed for the new dataset.
- Added `test_list=None` parameter: `list_path = os.path.join(data_root, test_list) if test_list else os.path.join(data_root, 'list/test.txt')`.

### `data/dali_data.py` — `LaneExternalIterator`

- Added `elif dataset_name == 'CULane_cropped':` cache path branch pointing to `culane_anno_cache.json`.
- Added `anno_cache=None` parameter: when provided, overrides the default cache path. Supports both relative paths (resolved against `data_root`) and absolute paths.
- Added `anno_cache=None` to `TrainCollect.__init__()` and propagated to `LaneExternalIterator`.

### `evaluation/eval_wrapper.py`

- `eval_lane()`: changed `elif cfg.dataset == 'CULane':` to `elif cfg.dataset in ('CULane', 'CULane_cropped'):` so the CULane evaluation branch is entered for the new dataset.
- `run_test()`: changed `if dataset == "CULane":` to `if dataset in ('CULane', 'CULane_cropped'):` so `generate_lines_local` and `generate_lines_col_local` are called (previously fell through to `raise NotImplementedError`).
- `generate_lines_local()`: added `elif dataset == 'CULane_cropped': lane_list = [0, 1]` (slots 0 and 1 for 2-lane model, instead of CULane's [1, 2]).
- `generate_lines_col_local()`: same, `elif dataset == 'CULane_cropped': lane_list = [0, 1]`.

### `factory.py`

- `get_loss_dict()`: added `'CULane_cropped'` to the existing `['Tusimple', 'CULane', 'CULane_cropped']` condition so the loss dictionary is constructed correctly.

### New config: `configs/culane_cropped_res34.py`

Full config for CULane\_cropped with ResNet-34 backbone. Key values:
- `num_row=72`, `num_col=81`, `num_cell_row=200`, `num_cell_col=100`
- `row_anchor = linspace(0, 1, 72)` covering the full image height (0–590 px)
- `num_lanes=2`, `warmup_iters=695`

---

## 2. `my_interp` — Python fallback

**Problem:** the original code requires a compiled CUDA extension (`my_interp`). If it is not compiled, Python imports the `my_interp/` directory as an empty namespace package and `my_interp.run` does not exist, crashing at the first training batch.

**Fix:** `my_interp/__init__.py` was created. It first tries to import the compiled `.so` extension; if that fails it provides a vectorised PyTorch implementation of the same interpolation logic.

**Additional fix — sentinel corruption after DALI augmentation:**

The DALI pipeline applies affine transforms (rotation, shear) uniformly to all coordinates including sentinel values (`x = -99999`). After a 6° rotation, the sentinel `x ≈ −99999` becomes `x ≈ −99417`, which passed the old validity threshold `x > −99998`. Two changes were made in `my_interp/__init__.py`:

- Validity threshold changed from `SENTINEL + 1 = −99998` to `VALID_THRESHOLD = −9999.0`.
- Added `torch.argsort(src, dim=2)` sort before `torch.searchsorted`, because DALI augmentation can reorder the Y values of the sentinel points, breaking the sorted-input requirement of `searchsorted`.

---

## 3. Evaluation pipeline fixes

### `evaluation/eval_wrapper.py` — `eval_lane()`

- **`cfg.tta` AttributeError:** `if not cfg.tta:` raised `AttributeError` when `tta` was not in the config. Changed to `if not getattr(cfg, 'tta', False):`.
- **`call_culane_eval` result aggregation:** the fallback path (no `test_split/` directory) now runs the evaluator once per IoU/margin combination and maps the single result to all 9 split keys. The outer loop in `eval_lane` sums TP/FP/FN over all splits; since all splits are identical in this case the final F-measure is mathematically equivalent to the single-run result.

### `evaluation/culane/evaluate` — binary compilation

The original `Makefile` and `CMakeLists.txt` assume a system OpenCV installation. On machines where only a conda OpenCV is available, compilation requires pointing the compiler explicitly at the conda library:

```bash
OPENCV_PKG=/path/to/conda/env
g++ -std=c++11 -fopenmp -DCPU_ONLY \
    -Iinclude -I${OPENCV_PKG}/include/opencv4 \
    $(find src/ -name "*.cpp") -o evaluate \
    -L${OPENCV_PKG}/lib \
    -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs \
    -Wl,-rpath,${OPENCV_PKG}/lib -fopenmp
```

The binary is gitignored (`evaluation/culane/evaluate`).

---

## 4. Configurable training and evaluation lists

New optional config keys in `configs/culane_cropped_res34.py` (all default to `None`):

| Key | Default | Description |
|---|---|---|
| `train_list` | `None` | Relative path to train list inside `data_root`. `None` → `list/train_gt.txt`. |
| `anno_cache` | `None` | Relative or absolute path to annotation cache JSON. `None` → dataset default. |
| `test_list` | `None` | Relative path to test list inside `data_root`. `None` → `list/test.txt`. |

Propagation chain for `test_list`: `eval_lane()` → `run_test()` → `get_test_loader()` → `call_culane_eval()`.

Propagation chain for `train_list` / `anno_cache`: `get_train_loader()` → `TrainCollect` → `LaneExternalIterator`.

---

## 5. Detection length filter

**Problem:** the model can generate short lane detections (spanning few anchors) that are false positives. These pass the original threshold (`valid.sum() > num_cls/2` for row, `> num_cls/4` for col) but do not correspond to real lanes.

**Fix:** two new optional config keys control the minimum fraction of anchors that must be valid for a detection to be written to the `.lines.txt` prediction file:

```python
min_row_frac = None   # None → original: > num_cls // 2 (50%)
min_col_frac = None   # None → original: > num_cls // 4 (25%)
```

When set, the threshold is `int(num_cls * frac)`. Row and col are independent because the original thresholds differed (50 % vs 25 %).

Propagation chain: config → `eval_lane()` → `run_test()` → `generate_lines_local()` / `generate_lines_col_local()`.

---

## 6. Training sanity check

Added a first-batch check at the start of each training run (`train.py`) that reads one batch from the DALI loader and prints the number of valid `labels_row_float` entries per lane. This catches data pipeline issues (e.g. all-sentinel labels) before the full training loop starts.

---

## 7. Evaluation debug and visualisation

- `run_test()` logs `eval/input_with_pred` (model output) and `eval/input_with_lines` (what the evaluator binary sees) to TensorBoard every 20 batches.
- `eval_history.jsonl` is appended after every epoch with full TP/FP/FN per IoU/margin, enabling offline analysis of training progression.
- A JSON summary `culane_eval_tmp_eval_results.json` is written to both `test_work_dir` and the log directory after each evaluation.

---

## 8. `infer.py` — standalone inference script

New script `infer.py` at the repository root. Runs a trained checkpoint on a single image or a folder of images without requiring a test list, annotation cache, or the DALI library.

**Inputs:** `--model` (checkpoint path), `--input` (image or folder), `--output` (output directory), `--config` (defaults to `configs/culane_cropped_res34.py`), `--save_txt` (also write `.lines.txt`).

**Outputs:** annotated images (`*_lanes.jpg`) and optionally `.lines.txt` files with lane coordinates in original image pixel space. The input directory structure is mirrored in the output directory.

The post-processing in `infer.py` (`pred_to_coords`) is equivalent to `generate_lines_local` + `generate_lines_col_local` from `eval_wrapper.py` but written as a self-contained function. Lane slot indices are `[0, 1]` for both row and col branches (correct for 2-lane CULane\_cropped model). The original `demo.py` used `[1, 2]` for row and `[0, 3]` for col which are the CULane 4-lane indices.

---

## 9. `--eval_only` flag

Added `--eval_only` CLI argument to `train.py`. When set together with `--test_model /path/to/checkpoint.pth`, the script loads the checkpoint, runs evaluation, prints results, and exits without entering the training loop.
