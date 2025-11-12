#!/usr/bin/env python3
"""
Filter CULane .lines.txt detection files with several heuristics and run the CULane evaluator
for each filtered variant, collecting TP/FP/FN/Precision/Recall/Fmeasure results.

Usage: python scripts/filter_and_eval.py --detect_dir <path> --anno_dir <path> --im_dir <path> --list_im_file <path>

Defaults tuned to your earlier values: width_lane=60, iou=0.3, im_width=1640, im_height=590

This script writes filtered copies into <out_root>/<variant>/ and writes a summary JSON.
"""

import argparse
import os
import shutil
import math
import json
import subprocess
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--detect_dir', required=True, help='directory containing .lines.txt detection files (can be nested)')
    p.add_argument('--anno_dir', required=True, help='annotation directory (CULane)')
    p.add_argument('--im_dir', required=True, help='image directory')
    p.add_argument('--list_im_file', required=True, help='list of images (test list)')
    p.add_argument('--out_root', default='culane_eval_filtered', help='where to put filtered folders')
    p.add_argument('--width_lane', type=int, default=60)
    p.add_argument('--iou', type=float, default=0.3)
    p.add_argument('--im_width', type=int, default=1640)
    p.add_argument('--im_height', type=int, default=590)
    p.add_argument('--evaluate_bin', default=os.path.join('evaluation','culane','evaluate'), help='path to CULane evaluator binary')
    p.add_argument('--dry_run', action='store_true', help='do not run evaluator, only create filtered folders')
    return p.parse_args()


def read_lines_file(path):
    lanes = []
    if not os.path.exists(path):
        return lanes
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) % 2 != 0:
                # malformed
                continue
            pts = []
            for i in range(0, len(parts), 2):
                try:
                    x = float(parts[i])
                    y = float(parts[i+1])
                except Exception:
                    continue
                pts.append((x,y))
            lanes.append(pts)
    return lanes


def lane_length_px(pts):
    if len(pts) < 2:
        return 0.0
    L = 0.0
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i-1][0]
        dy = pts[i][1] - pts[i-1][1]
        L += math.hypot(dx, dy)
    return L


def lane_max_y(pts):
    if len(pts) == 0:
        return -1
    return max(p[1] for p in pts)


def write_lines_file(path, lanes):
    # each lane is written as: x1 y1 x2 y2 ...\n
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for lane in lanes:
            if len(lane) == 0:
                f.write('\n')
                continue
            parts = []
            for x,y in lane:
                parts.append(f"{x:.2f}")
                parts.append(f"{y:.2f}")
            f.write(' '.join(parts) + '\n')


def apply_filters_to_file(src_path, dst_path, cfg):
    lanes = read_lines_file(src_path)
    before = len(lanes)
    out_lanes = []
    for lane in lanes:
        L = lane_length_px(lane)
        npts = len(lane)
        maxy = lane_max_y(lane)
        keep = True
        # min length
        if cfg['min_length'] > 0 and L < cfg['min_length']:
            keep = False
        # min points
        if cfg['min_points'] > 0 and npts < cfg['min_points']:
            keep = False
        # bottom fraction: require lane reaching bottom_frac of image height (max_y >= im_height*(1-bottom_frac))
        if cfg['bottom_frac'] > 0.0:
            if maxy < cfg['im_height'] * (1.0 - cfg['bottom_frac']):
                keep = False
        # if passes, keep
        if keep:
            out_lanes.append(lane)
    write_lines_file(dst_path, out_lanes)
    return before, len(out_lanes)


def collect_txt_files(root):
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith('.lines.txt'):
                rel = os.path.relpath(os.path.join(dirpath, fn), root)
                files.append(rel)
    return files


def run_evaluator(evaluate_bin, anno_dir, detect_dir, im_dir, list_im_file, width_lane, iou, im_w, im_h, out_file):
    cmd = [evaluate_bin, '-a', anno_dir, '-d', detect_dir, '-i', im_dir, '-l', list_im_file, '-w', str(width_lane), '-t', str(iou), '-c', str(im_w), '-r', str(im_h), '-o', out_file]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, check=False)
        return proc.returncode, proc.stdout
    except Exception as e:
        return -1, str(e)


def parse_eval_stdout(stdout):
    # extract tp: N fp: M fn: K and precision/recall/Fmeasure
    lines = stdout.splitlines()
    out = {}
    for line in lines:
        line = line.strip()
        if line.startswith('tp:') or ('tp:' in line and 'fp:' in line and 'fn:' in line):
            # token parse
            parts = line.replace(',', ' ').split()
            for i,p in enumerate(parts):
                if p.startswith('tp:'):
                    try:
                        out['tp'] = int(p.split(':')[1])
                    except Exception:
                        pass
                if p == 'tp:':
                    try:
                        out['tp'] = int(parts[i+1])
                    except Exception:
                        pass
                if p.startswith('fp:'):
                    try:
                        out['fp'] = int(p.split(':')[1])
                    except Exception:
                        pass
                if p == 'fp:':
                    try:
                        out['fp'] = int(parts[i+1])
                    except Exception:
                        pass
                if p.startswith('fn:'):
                    try:
                        out['fn'] = int(p.split(':')[1])
                    except Exception:
                        pass
                if p == 'fn:':
                    try:
                        out['fn'] = int(parts[i+1])
                    except Exception:
                        pass
        if line.startswith('precision:'):
            try:
                out['precision'] = float(line.split(':')[1].strip())
            except Exception:
                pass
        if line.startswith('recall:'):
            try:
                out['recall'] = float(line.split(':')[1].strip())
            except Exception:
                pass
        if line.startswith('Fmeasure:'):
            try:
                out['Fmeasure'] = float(line.split(':')[1].strip())
            except Exception:
                pass
    return out


def main():
    args = parse_args()
    detect_dir = args.detect_dir
    anno_dir = args.anno_dir
    im_dir = args.im_dir
    list_im_file = args.list_im_file
    out_root = args.out_root
    evaluate_bin = args.evaluate_bin

    configs = [
        {'name':'baseline','min_length':0,'min_points':0,'bottom_frac':0.0, 'im_height':args.im_height},
        {'name':'len40_pts2','min_length':40,'min_points':2,'bottom_frac':0.0, 'im_height':args.im_height},
        {'name':'len60_pts3','min_length':60,'min_points':3,'bottom_frac':0.0, 'im_height':args.im_height},
        {'name':'bottom05_len40','min_length':40,'min_points':2,'bottom_frac':0.05, 'im_height':args.im_height},
        {'name':'bottom10_len40','min_length':40,'min_points':2,'bottom_frac':0.10, 'im_height':args.im_height},
        {'name':'pts4','min_length':0,'min_points':4,'bottom_frac':0.0, 'im_height':args.im_height},
    ]

    txt_files = collect_txt_files(detect_dir)
    if len(txt_files) == 0:
        print('No .lines.txt files found under detect_dir:', detect_dir)
        return

    summary = {}
    os.makedirs(out_root, exist_ok=True)

    for cfg in configs:
        name = cfg['name']
        print('Running config:', name, cfg)
        out_dir = os.path.join(out_root, name)
        # create fresh out_dir by copying tree structure or creating directories
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        # replicate directory structure and apply filters
        for rel in txt_files:
            src = os.path.join(detect_dir, rel)
            dst = os.path.join(out_dir, rel)
            before, after = apply_filters_to_file(src, dst, {**cfg, 'im_height': args.im_height})
            # we allow empty files; evaluator expects files to exist
        # write a summary of counts per-file
        # (optional) run evaluator
        out_eval_file = os.path.join(out_root, f'{name}_eval_out.txt')
        if args.dry_run:
            print('Dry run: filtered files written to', out_dir)
            summary[name] = {'status':'dry_run', 'filtered_dir': out_dir}
            continue
        # run evaluator
        retcode, stdout = run_evaluator(evaluate_bin, anno_dir, out_dir, im_dir, list_im_file, args.width_lane, args.iou, args.im_width, args.im_height, out_eval_file)
        parsed = parse_eval_stdout(stdout)
        parsed['retcode'] = retcode
        parsed['stdout_snippet'] = '\n'.join(stdout.splitlines()[:40])
        parsed['filtered_dir'] = out_dir
        parsed['out_eval_file'] = out_eval_file
        summary[name] = parsed
        # write interim summary
        with open(os.path.join(out_root, 'summary.json'), 'w') as fp:
            json.dump(summary, fp, indent=2)
        print('Config', name, 'done. Parsed:', parsed)

    print('All done. Summary written to', os.path.join(out_root, 'summary.json'))

if __name__ == '__main__':
    main()
