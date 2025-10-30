# Ultra-Fast-Lane-Detection-V2

ACRA2025 modification of PyTorch implementation of the paper "[Ultra Fast Deep Lane Detection with Hybrid Anchor Driven Ordinal Classification](https://arxiv.org/abs/2206.07389)".


![](ufldv2.png "vis")

# Demo 
none


# Install
Please see [INSTALL.md](./INSTALL.md)

# Get started
Please modify the `data_root` in any configs you would like to run. We will use `configs/culane_res18.py` as an example.

To train the model, you can run:
```
python train.py configs/culane_res18.py --log_path /path/to/your/work/dir
```
or
```
python -m torch.distributed.launch --nproc_per_node=8 train.py configs/culane_res18.py --log_path /path/to/your/work/dir
```
It should be noted that if you use different number of GPUs, the learning rate should be adjusted accordingly. The configs' learning rates correspond to 8-GPU training on CULane and CurveLanes datasets. **If you want to train on CULane or CurveLanes with single GPU, please decrease the learning rate by a factor of 1/8.** On the Tusimple, the learning rate corresponds to single GPU training.

## Changes in this repository / Fork (user-adaptations)
This branch contains several practical improvements and dataset adaptations added to make the codebase easier to use on custom datasets and for faster verification:

- Input resizing and no-crop
    - Images (and corresponding masks/labels) are resized together to the train resolution defined in configs (default used here: 800x288). The test loader uses the same resize so evaluation inputs match training preprocessing.
    - Forced top-cropping at test time was removed so evaluation uses full resized images.

- Masks and label conventions
    - Support for masks with values {0,1,2} (background + two lane labels) is supported. Ensure your dataset lists and mask formats follow the CULane-style layout used by the dataset loaders.

- Anchors and row/col sampling
    - `cfg.row_anchor` (in configs) is normalized to [0,1] and used by the evaluation to map sampled row positions back to original image coordinates (1640x590). The per-config `num_row` and `num_col` control how many samples are taken (e.g. 72 rows, 81 cols in some configs).
    - `data/constant.py` contains the `culane_row_anchor` and `culane_col_anchor` constants (anchors in resized pixel space) used by the PyTorch loader.

- DALI and DataLoader tuning
    - The DALI pipeline and PyTorch loader were adjusted so the resize is applied consistently (no extra cropping). The number of worker threads used by the DALI/PyTorch loaders can be configured via the CLI flag `--num_workers` or the `cfg.num_workers` entry in configs to speed up IO on your machine.

- Evaluation improvements
    - New CLI flag `--eval_only` was added so you can run evaluation with a checkpoint without entering the training loop.
    - The evaluator no longer assumes a fixed number of lane output slots (previously used hard-coded indices like [1,2] or [0,3]). It now sanitizes lane indices to avoid index errors when your model outputs a different `num_lanes` (for example 2 lanes).
    - The CULane evaluator is now run for two standard tolerances by default: IoU=0.3 and IoU=0.5. The evaluator runs the external CULane binary twice on the same detection files (so inference is done once, then the binary is run twice with different `-t` values), and both results are saved.
    - A JSON summary is written after evaluation with both results at:
        `<cfg.test_work_dir>/culane_eval_tmp_eval_results.json` (keys: "0.3" and "0.5"). The per-split `out*_3.txt` and `out*_5.txt` files are also available under `<cfg.test_work_dir>/culane_eval_tmp/txt/`.

- Triage & debug helpers
    - When running evaluation the first batch's prediction tensor shapes are printed, to help confirm `loc_row`/`exist_row`/`loc_col`/`exist_col` shapes and detect mismatches early.

 
## Examples
Here are some practical example commands that reflect the changes in this fork (resize/no-crop, configurable num workers, eval-only, etc.). Adjust paths and flags as needed.

- Single-GPU training (with configurable data loader workers):

```bash
python train.py configs/culane_res34_train.py --num_workers 14
```




