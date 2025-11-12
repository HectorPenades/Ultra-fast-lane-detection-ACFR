import os
import cv2
import numpy as np
import tqdm
import json
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='The root of the dataset')
    parser.add_argument('--num_lanes', type=int, default=4, help='Number of lanes to generate in cache (default 4)')
    parser.add_argument('--out', default='culane_anno_cache.json', help='Output filename (relative to root)')
    return parser

if __name__ == '__main__':
    args = get_args().parse_args()
    culane_root = args.root
    train_list = os.path.join(culane_root, 'list/train_gt.txt')

    with open(train_list, 'r') as fp:
        res = fp.readlines()

    cache_dict = {}

    for line in tqdm.tqdm(res):
        info = line.split(' ')

        # label image
        label_path = os.path.join(culane_root, info[1][1:])
        label_img = cv2.imread(label_path)[:, :, 0]

        # txt de la lane
        txt_path = info[0][1:].replace('jpg', 'lines.txt')
        txt_path = os.path.join(culane_root, txt_path)
        lanes = open(txt_path, 'r').readlines()

        # *** IMPORTANTE: 4 lanes, 35 anchors (anchors computed from image height) ***
        # compute 35 row anchors spanning the full image height instead of hardcoding 250..590
        h, w = label_img.shape
        num_anchors = 35
        the_anno_row_anchor = np.linspace(0, h - 1, num=num_anchors).astype(float)

        # allocate points: num_lanes x num_anchors x 2
        num_lanes = args.num_lanes
        all_points = np.zeros((num_lanes, num_anchors, 2), dtype=float)
        # set y for each anchor
        all_points[:, :, 1] = np.tile(the_anno_row_anchor, (num_lanes, 1))
        # X inicial a -99999 (sin lane)
        all_points[:, :, 0] = -99999.0

        for lane_idx, lane in enumerate(lanes):
            ll = lane.strip().split(' ')
            point_x = ll[::2]
            point_y = ll[1::2]

            mid_x = int(float(point_x[int(len(point_x) / 2)]))
            mid_y = int(float(point_y[int(len(point_x) / 2)]))
            lane_order = int(label_img[mid_y - 1, mid_x - 1])
            # ignore lanes not marked in the segmentation mask
            if lane_order == 0:
                continue

            for i in range(len(point_x)):
                p1x = float(point_x[i])
                p1y = float(point_y[i])

                # map the y coordinate to the nearest anchor position
                # compute step (may be non-integer)
                step = the_anno_row_anchor[1] - the_anno_row_anchor[0]
                pos = int(round((p1y - the_anno_row_anchor[0]) / float(step)))
                if 0 <= pos < num_anchors and 1 <= lane_order <= num_lanes:
                    all_points[lane_order - 1, pos, 0] = p1x

        cache_dict[info[0][1:]] = all_points.tolist()

    out_name = args.out
    out_path = os.path.join(culane_root, out_name)
    with open(out_path, 'w') as f:
        json.dump(cache_dict, f)
    print('Wrote cache to', out_path, 'entries=', len(cache_dict))
