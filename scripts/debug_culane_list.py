#!/usr/bin/env python3
"""
Quick debugger for CULane train list vs dataset layout.

Given a dataset root and the train list (e.g. `list/train_gt.txt`), this script
reports how many entries exist, how many mask files are present, how many
`lines.txt` files are present, and prints a few example problematic lines.

Use it before running the cache builder to understand why the produced JSON is
much smaller than expected.

Example:
  python scripts/debug_culane_list.py --root /path/to/dataset --list list/train_gt.txt --sample 10
"""
import os
import argparse
import tqdm


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, help='Dataset root')
    p.add_argument('--list', default='list/train_gt.txt', help='Train list relative to root')
    p.add_argument('--sample', type=int, default=5, help='How many sample problematic lines to show')
    return p.parse_args()


def main():
    args = get_args()
    root = args.root
    list_path = os.path.join(root, args.list)
    if not os.path.exists(list_path):
        print('Train list not found:', list_path)
        return

    with open(list_path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    total = len(lines)
    missing_mask = []
    missing_lines = []
    unreadable_mask = []
    bad_label_point = []

    for ln in tqdm.tqdm(lines):
        parts = ln.split()
        if len(parts) < 2:
            missing_mask.append((ln, 'bad_format'))
            continue
        img_rel = parts[0].lstrip('/')
        mask_rel = parts[1].lstrip('/')
        mask_path = os.path.join(root, mask_rel)
        if not os.path.exists(mask_path):
            missing_mask.append((ln, mask_path))
            continue
        # check corresponding lines.txt
        txt_path = img_rel.replace('jpg', 'lines.txt')
        txt_full = os.path.join(root, txt_path)
        if not os.path.exists(txt_full):
            missing_lines.append((ln, txt_full))
            continue

    print('\nSummary for list:', list_path)
    print('  total lines:', total)
    print('  missing mask files:', len(missing_mask))
    print('  missing lines.txt files:', len(missing_lines))

    nshow = min(args.sample, len(missing_mask))
    if nshow > 0:
        print('\nSample missing masks:')
        for i in range(nshow):
            print(' -', missing_mask[i])

    nshow = min(args.sample, len(missing_lines))
    if nshow > 0:
        print('\nSample missing lines.txt:')
        for i in range(nshow):
            print(' -', missing_lines[i])

    if len(missing_mask) == 0 and len(missing_lines) == 0:
        print('\nAll referenced mask and lines.txt files exist for the entries in the list.')
    else:
        print('\nIf many files are missing, check whether the paths in the train list use a different root/prefix than the dataset root you passed.')


if __name__ == '__main__':
    main()
