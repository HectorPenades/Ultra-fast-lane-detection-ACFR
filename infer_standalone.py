"""
infer_standalone.py — Fully self-contained UFLDv2 inference for CULane_cropped_left.

This file has NO dependencies on the rest of the repository.
All model code is inlined. Only standard pip packages are required:
    pip install torch torchvision opencv-python Pillow numpy

Usage:
    # Single image
    python infer_standalone.py --model /path/to/model_best.pth --input /path/to/image.jpg

    # Folder of images (walks subdirectories)
    python infer_standalone.py --model /path/to/model_best.pth --input /path/to/images/

    # Custom output directory + save .lines.txt alongside each result
    python infer_standalone.py --model /path/to/model_best.pth \\
                               --input /path/to/images/ \\
                               --output /path/to/results/ \\
                               --save_txt

Trained for: CULane_cropped_left (1640×590 px, 2 lanes, ResNet-34 backbone).
"""

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image

# ── Hardcoded model config (culane_cropped_res34) ─────────────────────────────
# Modify these only if you retrain with a different config.
_CFG = {
    'backbone':       '34',
    'use_aux':        False,
    'fc_norm':        True,
    'num_lanes':      2,
    'train_width':    800,
    'train_height':   288,
    'num_cell_row':   200,   # X bins for row predictions
    'num_cell_col':   100,   # Y bins for col predictions
    'num_row':        72,    # number of row anchors
    'num_col':        81,    # number of col anchors
    # Anchors span the full image (0=top, 1=bottom / 0=left, 1=right)
    'row_anchor':     np.linspace(0.0, 1.0, 72),
    'col_anchor':     np.linspace(0.0, 1.0, 81),
}

# ── Visual style ──────────────────────────────────────────────────────────────
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
LANE_COLORS = [
    (0, 220, 0),    # slot 0 — left lane  (green)
    (0, 0, 220),    # slot 1 — right lane (red)
    (220, 220, 0),  # slot 2 (unused in 2-lane model)
    (220, 0, 220),  # slot 3 (unused)
]


# ═══════════════════════════════════════════════════════════════════════════════
# Inlined model code (from model/backbone.py, model/model_culane.py,
#                          utils/common.py)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Weight initialisation ─────────────────────────────────────────────────────
def _real_init_weights(m):
    if isinstance(m, list):
        for sub in m:
            _real_init_weights(sub)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, std=0.01)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Module):
        for sub in m.children():
            _real_init_weights(sub)


def _initialize_weights(*models):
    for model in models:
        _real_init_weights(model)


# ── ResNet backbone ───────────────────────────────────────────────────────────
class _Resnet(nn.Module):
    """Thin torchvision ResNet wrapper that returns (layer2, layer3, layer4)."""

    def __init__(self, layers, pretrained=False):
        super().__init__()
        # Support both old (pretrained=) and new (weights=) torchvision APIs.
        try:
            from torchvision.models import (
                ResNet18_Weights, ResNet34_Weights,
                ResNet50_Weights, ResNet101_Weights,
            )
            _w = {
                '18':  ResNet18_Weights.IMAGENET1K_V1  if pretrained else None,
                '34':  ResNet34_Weights.IMAGENET1K_V1  if pretrained else None,
                '50':  ResNet50_Weights.IMAGENET1K_V1  if pretrained else None,
                '101': ResNet101_Weights.IMAGENET1K_V1 if pretrained else None,
            }
            _builders = {
                '18':  torchvision.models.resnet18,
                '34':  torchvision.models.resnet34,
                '50':  torchvision.models.resnet50,
                '101': torchvision.models.resnet101,
            }
            if layers not in _builders:
                raise ValueError(f'Unsupported backbone: {layers}')
            model = _builders[layers](weights=_w[layers])
        except ImportError:
            # Older torchvision — use deprecated pretrained= argument
            _builders = {
                '18':  torchvision.models.resnet18,
                '34':  torchvision.models.resnet34,
                '50':  torchvision.models.resnet50,
                '101': torchvision.models.resnet101,
            }
            if layers not in _builders:
                raise ValueError(f'Unsupported backbone: {layers}')
            model = _builders[layers](pretrained=pretrained)

        self.conv1   = model.conv1
        self.bn1     = model.bn1
        self.relu    = model.relu
        self.maxpool = model.maxpool
        self.layer1  = model.layer1
        self.layer2  = model.layer2
        self.layer3  = model.layer3
        self.layer4  = model.layer4

    def forward(self, x):
        x  = self.conv1(x)
        x  = self.bn1(x)
        x  = self.relu(x)
        x  = self.maxpool(x)
        x  = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x2, x3, x4


# ── parsingNet ────────────────────────────────────────────────────────────────
class _ParsingNet(nn.Module):
    """UFLDv2 lane parsing network (use_aux=False variant)."""

    def __init__(self, backbone='34',
                 num_grid_row=200, num_cls_row=72,
                 num_grid_col=100, num_cls_col=81,
                 num_lane_on_row=2, num_lane_on_col=2,
                 input_height=288, input_width=800,
                 fc_norm=True):
        super().__init__()
        self.num_grid_row    = num_grid_row
        self.num_cls_row     = num_cls_row
        self.num_grid_col    = num_grid_col
        self.num_cls_col     = num_cls_col
        self.num_lane_on_row = num_lane_on_row
        self.num_lane_on_col = num_lane_on_col

        self.dim1 = num_grid_row * num_cls_row * num_lane_on_row
        self.dim2 = num_grid_col * num_cls_col * num_lane_on_col
        self.dim3 = 2 * num_cls_row * num_lane_on_row
        self.dim4 = 2 * num_cls_col * num_lane_on_col
        self.total_dim = self.dim1 + self.dim2 + self.dim3 + self.dim4

        self.input_dim = (input_height // 32) * (input_width // 32) * 8

        self.model = _Resnet(backbone, pretrained=False)

        # 1×1 conv to reduce channels to 8
        in_ch = 512 if backbone in ('18', '34', '34fca') else 2048
        self.pool = nn.Conv2d(in_ch, 8, 1)

        mlp_mid = 2048
        self.cls = nn.Sequential(
            nn.LayerNorm(self.input_dim) if fc_norm else nn.Identity(),
            nn.Linear(self.input_dim, mlp_mid),
            nn.ReLU(),
            nn.Linear(mlp_mid, self.total_dim),
        )

        _initialize_weights(self.cls)

    def forward(self, x):
        _, _, fea = self.model(x)
        fea = self.pool(fea)
        fea = fea.view(-1, self.input_dim)
        out = self.cls(fea)

        return {
            'loc_row':   out[:, :self.dim1].view(
                             -1, self.num_grid_row, self.num_cls_row, self.num_lane_on_row),
            'loc_col':   out[:, self.dim1:self.dim1 + self.dim2].view(
                             -1, self.num_grid_col, self.num_cls_col, self.num_lane_on_col),
            'exist_row': out[:, self.dim1 + self.dim2:self.dim1 + self.dim2 + self.dim3].view(
                             -1, 2, self.num_cls_row, self.num_lane_on_row),
            'exist_col': out[:, -self.dim4:].view(
                             -1, 2, self.num_cls_col, self.num_lane_on_col),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════════

def _build_model(cfg: dict) -> _ParsingNet:
    return _ParsingNet(
        backbone       = cfg['backbone'],
        num_grid_row   = cfg['num_cell_row'],
        num_cls_row    = cfg['num_row'],
        num_grid_col   = cfg['num_cell_col'],
        num_cls_col    = cfg['num_col'],
        num_lane_on_row= cfg['num_lanes'],
        num_lane_on_col= cfg['num_lanes'],
        input_height   = cfg['train_height'],
        input_width    = cfg['train_width'],
        fc_norm        = cfg['fc_norm'],
    )


def load_model(checkpoint_path: str, cfg: dict):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = _build_model(cfg)

    state = torch.load(checkpoint_path, map_location='cpu')
    state_dict = state.get('model', state)

    # Strip 'module.' prefix added by DistributedDataParallel
    cleaned = {(k[7:] if k.startswith('module.') else k): v
               for k, v in state_dict.items()}

    net.load_state_dict(cleaned, strict=True)
    net.to(device)
    net.eval()
    return net, device


# ═══════════════════════════════════════════════════════════════════════════════
# Pre/post-processing
# ═══════════════════════════════════════════════════════════════════════════════

def build_transform(cfg: dict) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((cfg['train_height'], cfg['train_width'])),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def pred_to_coords(pred: dict, cfg: dict, orig_w: int, orig_h: int,
                   local_width: int = 1) -> list:
    """
    Convert raw model output to pixel coordinates in the original image.

    Returns a list of detected lanes; each lane is a list of (x, y) tuples.
    Row-branch lanes come first, then col-branch lanes.
    """
    row_anchor = cfg['row_anchor']
    col_anchor = cfg['col_anchor']

    num_grid_row = pred['loc_row'].shape[1]
    num_cls_row  = pred['loc_row'].shape[2]
    num_grid_col = pred['loc_col'].shape[1]
    num_cls_col  = pred['loc_col'].shape[2]
    num_lane     = pred['loc_row'].shape[3]

    loc_row = pred['loc_row'].cpu()
    loc_col = pred['loc_col'].cpu()
    val_row = pred['exist_row'].argmax(1).cpu()   # (1, num_cls_row, num_lanes)
    val_col = pred['exist_col'].argmax(1).cpu()

    max_row = loc_row.argmax(1)  # (1, num_cls_row, num_lanes)
    max_col = loc_col.argmax(1)

    min_row_valid = num_cls_row // 2
    min_col_valid = num_cls_col // 4

    lane_list = list(range(num_lane))  # [0, 1] for 2-lane model
    coords = []

    # ── row branch: fixed Y, predicts X ──────────────────────────────────────
    for i in lane_list:
        if val_row[0, :, i].sum() <= min_row_valid:
            continue
        pts = []
        for k in range(num_cls_row):
            if not val_row[0, k, i]:
                continue
            lo  = max(0, max_row[0, k, i] - local_width)
            hi  = min(num_grid_row - 1, max_row[0, k, i] + local_width)
            ind = torch.arange(lo, hi + 1)
            x_norm = (loc_row[0, ind, k, i].softmax(0) * ind.float()).sum() + 0.5
            x_px   = int(x_norm / (num_grid_row - 1) * orig_w)
            y_px   = int(row_anchor[k] * orig_h)
            pts.append((x_px, y_px))
        if pts:
            coords.append(pts)

    # ── col branch: fixed X, predicts Y ──────────────────────────────────────
    for i in lane_list:
        if val_col[0, :, i].sum() <= min_col_valid:
            continue
        pts = []
        for k in range(num_cls_col):
            if not val_col[0, k, i]:
                continue
            lo  = max(0, max_col[0, k, i] - local_width)
            hi  = min(num_grid_col - 1, max_col[0, k, i] + local_width)
            ind = torch.arange(lo, hi + 1)
            y_norm = (loc_col[0, ind, k, i].softmax(0) * ind.float()).sum() + 0.5
            y_px   = int(y_norm / (num_grid_col - 1) * orig_h)
            x_px   = int(col_anchor[k] * orig_w)
            pts.append((x_px, y_px))
        if pts:
            coords.append(pts)

    return coords


# ═══════════════════════════════════════════════════════════════════════════════
# Visualisation + I/O helpers
# ═══════════════════════════════════════════════════════════════════════════════

def draw_lanes(image: np.ndarray, lanes: list,
               dot_radius: int = 4, line_thickness: int = 2) -> np.ndarray:
    vis = image.copy()
    for idx, lane in enumerate(lanes):
        color = LANE_COLORS[idx % len(LANE_COLORS)]
        pts = np.array(lane, dtype=np.int32)
        cv2.polylines(vis, [pts], isClosed=False, color=color,
                      thickness=line_thickness, lineType=cv2.LINE_AA)
        for (x, y) in lane:
            cv2.circle(vis, (x, y), dot_radius, color, -1, lineType=cv2.LINE_AA)
    return vis


def lanes_to_lines_txt(lanes: list) -> str:
    lines = [' '.join(f'{x} {y}' for x, y in lane) for lane in lanes]
    return '\n'.join(lines) + '\n' if lines else ''


def collect_images(input_path: str) -> list:
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
    rel = os.path.relpath(img_path, input_root)
    base, ext = os.path.splitext(rel)
    return os.path.join(output_dir, base + suffix + ext)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def run(args):
    cfg = _CFG  # hardcoded config

    print(f'Loading checkpoint: {args.model}')
    net, device = load_model(args.model, cfg)
    print(f'Model ready on {device}')

    transform = build_transform(cfg)

    input_root = os.path.dirname(args.input) if os.path.isfile(args.input) else args.input
    images = collect_images(args.input)
    if not images:
        print('No images found.')
        return

    os.makedirs(args.output, exist_ok=True)
    print(f'Processing {len(images)} image(s) → {args.output}')

    for img_path in images:
        pil_img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = pil_img.size

        tensor = transform(pil_img).unsqueeze(0)  # (1, 3, H, W)

        with torch.no_grad():
            pred = net(tensor.to(device))

        lanes = pred_to_coords(pred, cfg, orig_w, orig_h)

        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        vis = draw_lanes(bgr, lanes)

        out_img = output_path_for(img_path, input_root, args.output, '_lanes')
        os.makedirs(os.path.dirname(out_img), exist_ok=True)
        cv2.imwrite(out_img, vis)

        if args.save_txt:
            out_txt = os.path.splitext(out_img)[0] + '.lines.txt'
            with open(out_txt, 'w') as f:
                f.write(lanes_to_lines_txt(lanes))

        print(f'  {os.path.relpath(img_path, input_root):50s}  '
              f'{len(lanes)} lane(s) → {os.path.relpath(out_img, args.output)}')

    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='UFLDv2 standalone inference — draws detected lanes on images.\n'
                    'No repo dependencies. Requires: torch torchvision opencv-python Pillow numpy')

    parser.add_argument('--model', '-m', required=True,
                        help='Path to trained checkpoint (.pth)')
    parser.add_argument('--input', '-i', required=True,
                        help='Path to a single image or a folder of images')
    parser.add_argument('--output', '-o', default='inference_output',
                        help='Output directory (default: inference_output/)')
    parser.add_argument('--save_txt', action='store_true',
                        help='Also save .lines.txt prediction files')

    args = parser.parse_args()
    run(args)
