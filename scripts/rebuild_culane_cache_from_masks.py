#!/usr/bin/env python3
"""
Rebuild a provisional CULane annotation cache using the segmentation mask images.

This script reads the training list (list/train_gt.txt), loads the mask image for
each sample, and for each lane index (1..4) samples x-coordinates at the
standard anno row anchors (250,260,...,590). If multiple lane pixels exist at a
row we take the median x; if none found we set -99999 (same sentinel used by
the original cache script).

Output is a JSON mapping relative image path -> [[[x,y],...], ...] with shape
num_lanes x num_points x 2 (matching existing code expectations).

This produces a provisional file named `culane_anno_cache_rebuilt.json` by
default to avoid overwriting any existing cache.
"""
import os
import cv2
import numpy as np
import tqdm
import json
import argparse


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, help='Root of CULane dataset (contains list/train_gt.txt and laneseg_label_w16)')
    p.add_argument('--out', default='culane_anno_cache_rebuilt.json', help='Output JSON filename (relative to root)')
    p.add_argument('--list', default='list/train_gt.txt', help='Train list relative to root')
    return p.parse_args()


def main():
    args = get_args()
    root = args.root
    list_path = os.path.join(root, args.list)
    if not os.path.exists(list_path):
        raise FileNotFoundError(f"Train list not found: {list_path}")

    # Row anchors in original image coordinates used by the cache script
    the_anno_row_anchor = np.array([250, 260, 270, 280, 290, 300, 310, 320, 330, 340,
                                     350, 360, 370, 380, 390, 400, 410, 420, 430, 440,
                                     450, 460, 470, 480, 490, 500, 510, 520, 530, 540,
                                     550, 560, 570, 580, 590])

    with open(list_path, 'r') as fp:
        lines = fp.readlines()

    cache = {}
    entries = 0
    for line in tqdm.tqdm(lines):
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        img_rel = parts[0].lstrip('/')
        mask_rel = parts[1].lstrip('/')
        mask_path = os.path.join(root, mask_rel)
        if not os.path.exists(mask_path):
            # try alternative join
            mask_path = os.path.join(root, mask_rel.lstrip('/'))
        if not os.path.exists(mask_path):
            print(f"[WARN] mask not found for {img_rel}: tried {mask_path}")
            # create empty/all-sentinel points matching original cache format
            all_points = np.zeros((4, len(the_anno_row_anchor), 2), dtype=float)
            all_points[:, :, 1] = np.tile(the_anno_row_anchor, (4, 1))
            all_points[:, :, 0] = -99999.0
            cache[img_rel] = all_points.tolist()
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            print(f"[WARN] failed to read mask for {img_rel}: {mask_path}")
            # create empty/all-sentinel points matching original cache format
            all_points = np.zeros((4, len(the_anno_row_anchor), 2), dtype=float)
            all_points[:, :, 1] = np.tile(the_anno_row_anchor, (4, 1))
            all_points[:, :, 0] = -99999.0
            cache[img_rel] = all_points.tolist()
            continue

        # if mask has multiple channels, take first channel
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        h, w = mask.shape

        # prepare output array: 4 lanes x Npoints x 2
        # np.float is deprecated in recent numpy versions; use built-in float
        all_points = np.zeros((4, len(the_anno_row_anchor), 2), dtype=float)
        all_points[:, :, 1] = np.tile(the_anno_row_anchor, (4, 1))
        all_points[:, :, 0] = -99999.0

        # For each lane id (1..4), for each anchor row y find all x where mask==lane_id
        for lane_id in range(1, 5):
            for j, y in enumerate(the_anno_row_anchor):
                if y < 0 or y >= h:
                    continue
                row = mask[int(y), :]
                xs = np.where(row == lane_id)[0]
                if xs.size == 0:
                    # no pixel for this lane at this row
                    continue
                # choose median x (robust to multiple pixels)
                x_med = float(np.median(xs))
                all_points[lane_id - 1, j, 0] = x_med

        # store points for this image
        cache[img_rel] = all_points.tolist()
        entries += 1

    out_path = os.path.join(root, args.out)
    with open(out_path, 'w') as f:
        json.dump(cache, f)
    print(f"Wrote provisional cache to: {out_path}  (entries={entries})")
    # Print a few sample keys for quick verification
    sample_keys = list(cache.keys())[:5]
    print('Sample keys written:', sample_keys)


if __name__ == '__main__':
    main()
