import os
import cv2
import numpy as np
import tqdm
import json
import argparse


def get_args():
    p = argparse.ArgumentParser(description='Generate culane_anno_cache.json from lines.txt using label masks')
    p.add_argument('--root', required=True, help='Root of dataset (contains list/train_gt.txt)')
    p.add_argument('--list', default='list/train_gt.txt', help='Train list relative to root')
    p.add_argument('--out', default='culane_anno_cache.json', help='Output JSON filename (relative to root)')
    p.add_argument('--filter', default=None, help='Optional substring: only include images whose path contains this')
    p.add_argument('--num_lanes', type=int, default=4, help='Number of lanes to produce in cache (default 4)')
    p.add_argument('--verbose', action='store_true', help='Show progress messages and warnings')
    return p.parse_args()


def build_cache_for_lines(root, lines, filter_substr=None, verbose=False, num_lanes=4):
    the_anno_row_anchor = np.array([250, 260, 270, 280, 290, 300, 310, 320, 330, 340,
                                     350, 360, 370, 380, 390, 400, 410, 420, 430, 440,
                                     450, 460, 470, 480, 490, 500, 510, 520, 530, 540,
                                     550, 560, 570, 580, 590])
    cache_dict = {}
    entries = 0
    for line in tqdm.tqdm(lines):
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        img_rel = parts[0].lstrip('/')
        if filter_substr is not None and filter_substr not in img_rel:
            continue
        label_rel = parts[1].lstrip('/')

        img_key = img_rel

        # read label image (mask)
        label_path = os.path.join(root, label_rel)
        if not os.path.exists(label_path):
            if verbose:
                print('[WARN] label not found:', label_path)
            continue
        label_img = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        if label_img is None:
            if verbose:
                print('[WARN] failed to read label image:', label_path)
            continue
        if label_img.ndim == 3:
            label_img = label_img[:, :, 0]

        # read lines.txt corresponding to image
        txt_path = img_rel.replace('jpg', 'lines.txt')
        txt_path = os.path.join(root, txt_path)
        if not os.path.exists(txt_path):
            if verbose:
                print('[WARN] lines.txt not found for', img_rel, 'expected at', txt_path)
            continue
        with open(txt_path, 'r') as f:
            lanes = f.readlines()

        # create lane array sized by num_lanes
        all_points = np.zeros((num_lanes, len(the_anno_row_anchor), 2), dtype=float)
        all_points[:, :, 1] = np.tile(the_anno_row_anchor, (num_lanes, 1))
        all_points[:, :, 0] = -99999.0

        for lane in lanes:
            ll = lane.strip().split()
            if len(ll) < 2:
                continue
            point_x = ll[::2]
            point_y = ll[1::2]
            try:
                mid_idx = int(len(point_x) / 2)
                mid_x = int(float(point_x[mid_idx]))
                mid_y = int(float(point_y[mid_idx]))
            except Exception:
                continue
            # lane order determined by label image at mid point (1..num_lanes)
            h, w = label_img.shape
            if mid_y - 1 < 0 or mid_y - 1 >= h or mid_x - 1 < 0 or mid_x - 1 >= w:
                continue
            lane_order = int(label_img[mid_y - 1, mid_x - 1])
            if lane_order <= 0 or lane_order > num_lanes:
                if verbose:
                    print('[WARN] invalid lane order', lane_order, 'for', img_rel)
                continue
            for i in range(len(point_x)):
                try:
                    p1x = float(point_x[i])
                    py = int(float(point_y[i]))
                    pos = int((py - 250) / 10)
                    if pos < 0 or pos >= len(the_anno_row_anchor):
                        continue
                    all_points[lane_order - 1, pos, 0] = p1x
                except Exception:
                    continue

        cache_dict[img_key] = all_points.tolist()
        entries += 1

    return cache_dict, entries


if __name__ == '__main__':
    args = get_args()
    root = args.root
    list_path = os.path.join(root, args.list)
    if not os.path.exists(list_path):
        raise FileNotFoundError('Train list not found: ' + list_path)
    with open(list_path, 'r') as f:
        lines = f.readlines()

    cache, n = build_cache_for_lines(root, lines, filter_substr=args.filter, verbose=args.verbose, num_lanes=args.num_lanes)
    out_path = os.path.join(root, args.out)
    with open(out_path, 'w') as f:
        json.dump(cache, f)
    print('Wrote cache to', out_path, 'entries=', n)






