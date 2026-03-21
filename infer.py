"""
infer.py — Standalone inference for UFLDv2 (CULane_cropped_left)

Usage:
    # Single image
    python infer.py --model /path/to/model_best.pth --input /path/to/image.jpg

    # Folder of images
    python infer.py --model /path/to/model_best.pth --input /path/to/images/

    # Custom output directory
    python infer.py --model /path/to/model_best.pth --input /path/ --output /path/out/

    # Also save .lines.txt prediction files
    python infer.py --model /path/to/model_best.pth --input /path/ --save_txt

    # Different config (e.g. ResNet-50)
    python infer.py --model /path/to/model_best.pth --input /path/ \
                    --config configs/culane_cropped_res50.py
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

# ── repo imports ──────────────────────────────────────────────────────────────
from utils.common import get_model, merge_config

# ── constants ─────────────────────────────────────────────────────────────────
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

# One BGR colour per lane slot (left lane = green, right lane = red)
LANE_COLORS = [
    (0, 220, 0),    # slot 0 — left lane
    (0, 0, 220),    # slot 1 — right lane
    (220, 220, 0),  # slot 2 (extra, unused in 2-lane model)
    (220, 0, 220),  # slot 3 (extra)
]


# ─────────────────────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str, cfg) -> torch.nn.Module:
    """Load model from checkpoint. Returns model in eval mode on GPU (if available)."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = get_model(cfg)

    state = torch.load(checkpoint_path, map_location='cpu')
    # Checkpoints saved by train.py have a 'model' key; handle both formats.
    state_dict = state.get('model', state)

    # Strip 'module.' prefix added by DistributedDataParallel
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k[7:] if k.startswith('module.') else k] = v

    net.load_state_dict(cleaned, strict=True)
    net.to(device)
    net.eval()
    return net, device


def build_transform(cfg) -> transforms.Compose:
    """Preprocessing pipeline identical to the test loader."""
    return transforms.Compose([
        transforms.Resize((cfg.train_height, cfg.train_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


# ─────────────────────────────────────────────────────────────────────────────
def predict(net, img_tensor: torch.Tensor, device: str) -> dict:
    """Run a single image through the model. img_tensor: (1, 3, H, W)."""
    with torch.no_grad():
        return net(img_tensor.to(device))


def pred_to_coords(pred: dict, cfg, orig_w: int, orig_h: int,
                   local_width: int = 1) -> list[list[tuple[int, int]]]:
    """
    Convert model output to pixel coordinates in the original image space.

    Returns a list of detected lanes. Each lane is a list of (x, y) tuples.
    Lane order: row-branch lanes first, then col-branch lanes.
    """
    row_anchor = cfg.row_anchor   # normalised [0,1], length num_row
    col_anchor = cfg.col_anchor   # normalised [0,1], length num_col

    num_grid_row = pred['loc_row'].shape[1]
    num_cls_row  = pred['loc_row'].shape[2]
    num_grid_col = pred['loc_col'].shape[1]
    num_cls_col  = pred['loc_col'].shape[2]
    num_lane     = pred['loc_row'].shape[3]

    loc_row  = pred['loc_row'].cpu()    # (1, num_grid_row, num_cls_row, num_lanes)
    loc_col  = pred['loc_col'].cpu()
    val_row  = pred['exist_row'].argmax(1).cpu()  # (1, num_cls_row, num_lanes)
    val_col  = pred['exist_col'].argmax(1).cpu()

    max_row  = loc_row.argmax(1)        # (1, num_cls_row, num_lanes)
    max_col  = loc_col.argmax(1)

    min_row_valid = num_cls_row // 2
    min_col_valid = num_cls_col // 4

    lane_list = list(range(num_lane))   # [0, 1] for 2-lane model
    coords = []

    # ── row branch: each anchor is a horizontal line (fixed Y), predicts X ──
    for i in lane_list:
        if val_row[0, :, i].sum() <= min_row_valid:
            continue
        pts = []
        for k in range(num_cls_row):
            if not val_row[0, k, i]:
                continue
            lo = max(0, max_row[0, k, i] - local_width)
            hi = min(num_grid_row - 1, max_row[0, k, i] + local_width)
            ind = torch.arange(lo, hi + 1)
            x_norm = (loc_row[0, ind, k, i].softmax(0) * ind.float()).sum() + 0.5
            x_px = int(x_norm / (num_grid_row - 1) * orig_w)
            y_px = int(row_anchor[k] * orig_h)
            pts.append((x_px, y_px))
        if pts:
            coords.append(pts)

    # ── col branch: each anchor is a vertical line (fixed X), predicts Y ──
    for i in lane_list:
        if val_col[0, :, i].sum() <= min_col_valid:
            continue
        pts = []
        for k in range(num_cls_col):
            if not val_col[0, k, i]:
                continue
            lo = max(0, max_col[0, k, i] - local_width)
            hi = min(num_grid_col - 1, max_col[0, k, i] + local_width)
            ind = torch.arange(lo, hi + 1)
            y_norm = (loc_col[0, ind, k, i].softmax(0) * ind.float()).sum() + 0.5
            y_px = int(y_norm / (num_grid_col - 1) * orig_h)
            x_px = int(col_anchor[k] * orig_w)
            pts.append((x_px, y_px))
        if pts:
            coords.append(pts)

    return coords


# ─────────────────────────────────────────────────────────────────────────────
def draw_lanes(image: np.ndarray, lanes: list[list[tuple[int, int]]],
               dot_radius: int = 4, line_thickness: int = 2) -> np.ndarray:
    """Draw lane points and connecting polylines on a copy of the image."""
    vis = image.copy()
    for idx, lane in enumerate(lanes):
        color = LANE_COLORS[idx % len(LANE_COLORS)]
        pts = np.array(lane, dtype=np.int32)
        # Connect consecutive points as a polyline
        cv2.polylines(vis, [pts], isClosed=False, color=color,
                      thickness=line_thickness, lineType=cv2.LINE_AA)
        # Draw each anchor point
        for (x, y) in lane:
            cv2.circle(vis, (x, y), dot_radius, color, -1, lineType=cv2.LINE_AA)
    return vis


def lanes_to_lines_txt(lanes: list[list[tuple[int, int]]]) -> str:
    """Format lanes as CULane-style .lines.txt content (one lane per line)."""
    lines = []
    for lane in lanes:
        line = ' '.join(f'{x} {y}' for x, y in lane)
        lines.append(line)
    return '\n'.join(lines) + '\n' if lines else ''


# ─────────────────────────────────────────────────────────────────────────────
def collect_images(input_path: str) -> list[str]:
    """Return sorted list of image paths from a file or directory."""
    if os.path.isfile(input_path):
        return [input_path]
    if os.path.isdir(input_path):
        paths = []
        for root, _, files in os.walk(input_path):
            for fname in sorted(files):
                if os.path.splitext(fname)[1].lower() in IMG_EXTENSIONS:
                    paths.append(os.path.join(root, fname))
        return sorted(paths)
    raise FileNotFoundError(f'Input not found: {input_path}')


def output_path_for(img_path: str, input_root: str, output_dir: str,
                    suffix: str = '_lanes') -> str:
    """Mirror the input directory structure inside output_dir."""
    rel = os.path.relpath(img_path, input_root)
    base, ext = os.path.splitext(rel)
    return os.path.join(output_dir, base + suffix + ext)


# ─────────────────────────────────────────────────────────────────────────────
def run(args):
    # ── config ────────────────────────────────────────────────────────────────
    sys.argv = [sys.argv[0], args.config]
    _, cfg = merge_config()

    # ── model ─────────────────────────────────────────────────────────────────
    print(f'Loading checkpoint: {args.model}')
    net, device = load_model(args.model, cfg)
    print(f'Model ready on {device}')

    transform = build_transform(cfg)

    # ── images ────────────────────────────────────────────────────────────────
    if os.path.isfile(args.input):
        input_root = os.path.dirname(args.input)
    else:
        input_root = args.input

    images = collect_images(args.input)
    if not images:
        print('No images found.')
        return

    os.makedirs(args.output, exist_ok=True)
    print(f'Processing {len(images)} image(s) → {args.output}')

    for img_path in images:
        # load
        pil_img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = pil_img.size

        # preprocess
        tensor = transform(pil_img).unsqueeze(0)  # (1, 3, H, W)

        # infer
        pred = predict(net, tensor, device)

        # postprocess
        lanes = pred_to_coords(pred, cfg, orig_w, orig_h)

        # annotated image
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        vis = draw_lanes(bgr, lanes)

        out_img = output_path_for(img_path, input_root, args.output, '_lanes')
        os.makedirs(os.path.dirname(out_img), exist_ok=True)
        cv2.imwrite(out_img, vis)

        # optional .lines.txt
        if args.save_txt:
            out_txt = os.path.splitext(out_img)[0] + '.lines.txt'
            with open(out_txt, 'w') as f:
                f.write(lanes_to_lines_txt(lanes))

        n_lanes = len(lanes)
        print(f'  {os.path.relpath(img_path, input_root):50s}  '
              f'{n_lanes} lane(s) detected → {os.path.relpath(out_img, args.output)}')

    print('Done.')


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='UFLDv2 inference — draws detected lanes on images.')

    parser.add_argument('--model', '-m', required=True,
                        help='Path to trained checkpoint (.pth)')
    parser.add_argument('--input', '-i', required=True,
                        help='Path to a single image or a folder of images')
    parser.add_argument('--output', '-o', default='inference_output',
                        help='Output directory (default: inference_output/)')
    parser.add_argument('--config', '-c',
                        default='configs/culane_cropped_res34.py',
                        help='Config file (default: configs/culane_cropped_res34.py)')
    parser.add_argument('--save_txt', action='store_true',
                        help='Also save .lines.txt prediction files')

    args = parser.parse_args()
    run(args)
