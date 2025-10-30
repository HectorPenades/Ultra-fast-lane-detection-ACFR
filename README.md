# Ultra-Fast-Lane-Detection-V2
PyTorch implementation of the paper "[Ultra Fast Deep Lane Detection with Hybrid Anchor Driven Ordinal Classification](https://arxiv.org/abs/2206.07389)".


![](ufldv2.png "vis")

# Demo 
<a href="https://youtu.be/VkvpoHlaMe0
" target="_blank"><img src="http://img.youtube.com/vi/VkvpoHlaMe0/0.jpg" 
alt="Demo" width="240" height="180" border="10" /></a>


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

If you want different behavior (e.g. evaluate only at a single IoU), or to change the output paths, those are easy to configure — tell me and I can add a CLI/`cfg` option such as `--eval_ious 0.3,0.5` or a custom output filename.
# If you want different behavior (e.g. evaluate only at a single IoU), or to change the output paths, those are easy to configure — tell me and I can add a CLI/`cfg` option such as `--eval_ious 0.3,0.5` or a custom output filename.

## Examples
Here are some practical example commands that reflect the changes in this fork (resize/no-crop, configurable num workers, eval-only, etc.). Adjust paths and flags as needed.

- Single-GPU training (with configurable data loader workers):

```bash
python train.py configs/culane_res34_train.py --log_path /path/to/work --num_workers 8
```

- Multi-GPU distributed training (8 GPUs):

```bash
python -m torch.distributed.launch --nproc_per_node=8 train.py configs/culane_res34_train.py --log_path /path/to/work --num_workers 16
```

- Evaluate a checkpoint only (runs inference once, then the CULane evaluator for IoU=0.3 and 0.5 and writes a JSON summary):

```bash
python train.py configs/culane_res34_train.py --eval_only --test_model /path/to/model.pth --test_work_dir ./tmp --num_workers 8
```

After evaluation finishes you will find the per-IoU files under `./tmp/culane_eval_tmp/txt/` (e.g. `out0_normal_3.txt`, `out0_normal_5.txt`) and a summary JSON at:

```
./tmp/culane_eval_tmp_eval_results.json
```

- Visualize results on images / demo (same interface as original):

```bash
python demo.py configs/culane_res18.py --test_model /path/to/your/culane_res18.pth
```

If you want custom IoU thresholds or different output paths, I can add a CLI flag (e.g. `--eval_ious 0.3,0.5`) — useful if you want to try other thresholds.

# Trained models
We provide trained models on CULane, Tusimple, and CurveLanes.

| Dataset    | Backbone | F1   | Link |
|------------|----------|-------|------|
| CULane     | ResNet18 | 75.0  |  [Google](https://drive.google.com/file/d/1oEjJraFr-3lxhX_OXduAGFWalWa6Xh3W/view?usp=sharing)/[Baidu](https://pan.baidu.com/s/1Z3W4y3eA9xrXJ51-voK4WQ?pwd=pdzs)    |
| CULane     | ResNet34 | 76.0  |   [Google](https://drive.google.com/file/d/1AjnvAD3qmqt_dGPveZJsLZ1bOyWv62Yj/view?usp=sharing)/[Baidu](https://pan.baidu.com/s/1PHNpVHboQlmpjM5NXl9IxQ?pwd=jw8f)   |
| Tusimple   | ResNet18 | 96.11 |   [Google](https://drive.google.com/file/d/1Clnj9-dLz81S3wXiYtlkc4HVusCb978t/view?usp=sharing)/[Baidu](https://pan.baidu.com/s/1umHo0RZIAQ1l_FzL2aZomw?pwd=6xs1)   |
| Tusimple   | ResNet34 | 96.24 |   [Google](https://drive.google.com/file/d/1pkz8homK433z39uStGK3ZWkDXrnBAMmX/view?usp=sharing)/[Baidu](https://pan.baidu.com/s/1Eq7oxnDoE0vcQGzs1VsGZQ?pwd=b88p)   |
| CurveLanes | ResNet18 | 80.42 |   [Google](https://drive.google.com/file/d/1VfbUvorKKMG4tUePNbLYPp63axgd-8BX/view?usp=sharing)/[Baidu](https://pan.baidu.com/s/1jCqKqgSQdh6nwC5pYpYO1A?pwd=urhe)   |
| CurveLanes | ResNet34 | 81.34 |   [Google](https://drive.google.com/file/d/1O1kPSr85Icl2JbwV3RBlxWZYhLEHo8EN/view?usp=sharing)/[Baidu](https://pan.baidu.com/s/1fk2Wg-1QoHXTnTlasSM6uQ?pwd=4mn3)   |

For evaluation, run
```Shell
mkdir tmp

python test.py configs/culane_res18.py --test_model /path/to/your/model.pth --test_work_dir ./tmp
```

Same as training, multi-gpu evaluation is also supported.
```Shell
mkdir tmp

python -m torch.distributed.launch --nproc_per_node=8 test.py configs/culane_res18.py --test_model /path/to/your/model.pth --test_work_dir ./tmp
```

# Visualization
We provide a script to visualize the detection results. Run the following commands to visualize on the testing set of CULane.
```
python demo.py configs/culane_res18.py --test_model /path/to/your/culane_res18.pth
```

# Tensorrt Deploy
We also provide a python script to do tensorrt inference on videos.

1. Convert to onnx model
    ```
    python deploy/pt2onnx.py --config_path configs/culane_res34.py --model_path weights/culane_res34.pth
    ```
    Or you can download the onnx model using the following script: https://github.com/PINTO0309/PINTO_model_zoo/blob/main/324_Ultra-Fast-Lane-Detection-v2/download.sh. And copy `ufldv2_culane_res34_320x1600.onnx` to `weights/ufldv2_culane_res34_320x1600.onnx`

2. Convert to tensorrt model

    Use trtexec to convert engine model

    `trtexec --onnx=weights/culane_res34.onnx --saveEngine=weights/culane_res34.engine`

3. Do inference
    ```
    python deploy/trt_infer.py --config_path  configs/culane_res34.py --engine_path weights/culane_res34.engine --video_path example.mp4
    ```

# Citation

```BibTeX
@InProceedings{qin2020ultra,
author = {Qin, Zequn and Wang, Huanyu and Li, Xi},
title = {Ultra Fast Structure-aware Deep Lane Detection},
booktitle = {The European Conference on Computer Vision (ECCV)},
year = {2020}
}

@ARTICLE{qin2022ultrav2,
  author={Qin, Zequn and Zhang, Pengyi and Li, Xi},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Ultra Fast Deep Lane Detection With Hybrid Anchor Driven Ordinal Classification}, 
  year={2022},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TPAMI.2022.3182097}
}
```
