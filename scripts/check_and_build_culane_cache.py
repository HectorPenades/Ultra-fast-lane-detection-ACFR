#!/usr/bin/env python3
"""
Check and optionally rebuild/patch a CULane annotation cache (`culane_anno_cache.json`) using the lane segmentation masks.

This script reads the train list (default `list/train_gt.txt`), reconstructs the 4 x N x 2 points per image from the mask files
using median x per anchor row (same heuristic as `rebuild_culane_cache_from_masks.py`) and compares those values to the ones
present in the specified cache JSON. It reports missing entries, mismatches and can optionally write a corrected cache.

Usage examples:
  # dry-run compare
  python scripts/check_and_build_culane_cache.py --root /path/to/CULane

  # fix missing or mismatched entries and write a new cache file
  python scripts/check_and_build_culane_cache.py --root /path/to/CULane --fix --out culane_anno_cache_fixed.json

Options:
  --mode {masks}    (only 'masks' currently supported) 
  --tol FLOAT       tolerance in pixels for comparing x coordinates (default 1.0)
  --fix             write corrected entries into output cache (overwrites or creates file)
  --out PATH        output filename (relative to root) when --fix is used (default: culane_anno_cache_rebuilt_fixed.json)
  --list PATH       train list relative to root (default: list/train_gt.txt)
  --verbose         show sample diffs for first mismatches

The script mirrors the lookup behavior used by the DALI `LaneExternalIterator` so the same key normalization is applied.
"""
import os
import json
import argparse
import cv2
import numpy as np
import tqdm
import datetime


THE_ANNO_ROW_ANCHOR = np.array([250, 260, 270, 280, 290, 300, 310, 320, 330, 340,
                                 350, 360, 370, 380, 390, 400, 410, 420, 430, 440,
                                 450, 460, 470, 480, 490, 500, 510, 520, 530, 540,
                                 550, 560, 570, 580, 590])


def build_points_from_mask(mask, anchors=THE_ANNO_ROW_ANCHOR):
    """Return array shape (4, Nanchors, 2) where [:,:,0] are x (or -99999) and [:,:,1] are anchor y."""
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    h, w = mask.shape
    all_points = np.zeros((4, len(anchors), 2), dtype=float)
    all_points[:, :, 1] = np.tile(anchors, (4, 1))
    all_points[:, :, 0] = -99999.0
    for lane_id in range(1, 5):
        for j, y in enumerate(anchors):
            if y < 0 or y >= h:
                continue
            row = mask[int(y), :]
            xs = np.where(row == lane_id)[0]
            if xs.size == 0:
                continue
            all_points[lane_id - 1, j, 0] = float(np.median(xs))
    return all_points


def find_cache_entry(cache, img_rel):
    """Normalize key forms and try to find the cache entry (mirrors LaneExternalIterator)."""
    if cache is None:
        return None
    k_norm = img_rel.replace('\\', '/').lstrip('/')
    if k_norm in cache:
        return cache[k_norm]
    base = os.path.basename(k_norm)
    if base in cache:
        return cache[base]
    parts = k_norm.split('/')
    if len(parts) >= 2:
        last2 = '/'.join(parts[-2:])
        if last2 in cache:
            return cache[last2]
    if len(parts) >= 3:
        last3 = '/'.join(parts[-3:])
        if last3 in cache:
            return cache[last3]
    return None


def compare_points(a, b, tol=1.0):
    """Compare two (4,N,2) arrays. Returns (match_bool, mismatch_count, diff_map)
    diff_map is boolean mask where True indicates mismatch on x coordinate.
    Two values considered equal if both are sentinel (-99999) or their abs diff <= tol.
    """
    if a is None or b is None:
        return False, None, None

    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)

    # Ensure arrays have shape (4, N, 2). If they can be reshaped (same total size), do so.
    def _reshape_to_expected(x):
        anchors_len = THE_ANNO_ROW_ANCHOR.shape[0]
        expected_size = 4 * anchors_len * 2
        if x.shape == (4, anchors_len, 2):
            return x
        if x.size == expected_size:
            try:
                return x.reshape((4, anchors_len, 2))
            except Exception:
                pass
        # common transpose case
        if x.ndim == 3 and x.shape[0] == anchors_len and x.shape[1] == 4 and x.shape[2] == 2:
            return x.transpose(1, 0, 2)
        return None

    a_norm = _reshape_to_expected(a_arr)
    b_norm = _reshape_to_expected(b_arr)
    if a_norm is None or b_norm is None:
        # shapes incompatible
        return False, None, (a_arr.shape, b_arr.shape)

    a_x = a_norm[:, :, 0].astype(float)
    b_x = b_norm[:, :, 0].astype(float)
    # sentinel equality
    a_sentinel = (a_x == -99999.0)
    b_sentinel = (b_x == -99999.0)
    both_sentinel = a_sentinel & b_sentinel
    diff = np.zeros_like(a_x, dtype=bool)
    # Where both finite, check tol
    comp = np.abs(a_x - b_x) <= tol
    # For positions where either is sentinel but not both, mark mismatch
    diff = ~comp
    diff[both_sentinel] = False
    mismatch_count = int(np.sum(diff))
    match = mismatch_count == 0
    return match, mismatch_count, diff


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, help='Root of CULane dataset')
    p.add_argument('--out', default='culane_anno_cache_rebuilt_fixed.json', help='Output cache filename relative to root when --fix')
    p.add_argument('--list', default='list/train_gt.txt', help='Train list relative to root')
    p.add_argument('--cache', default='culane_anno_cache.json', help='Existing cache filename relative to root')
    p.add_argument('--mode', choices=['masks'], default='masks', help='Source to rebuild from (currently only masks supported)')
    p.add_argument('--tol', type=float, default=1.0, help='Tolerance in pixels for x coordinate comparison')
    p.add_argument('--fix', action='store_true', help='Write corrected/filled entries to output cache')
    p.add_argument('--verbose', action='store_true', help='Print sample diffs for mismatches')
    args = p.parse_args()

    root = args.root
    list_path = os.path.join(root, args.list)
    if not os.path.exists(list_path):
        raise FileNotFoundError(f"Train list not found: {list_path}")

    # load existing cache if present
    cache_path = os.path.join(root, args.cache)
    cache = None
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            try:
                cache = json.load(f)
            except Exception as e:
                print('Failed to parse existing cache JSON:', e)
                cache = None
    else:
        print('No existing cache found at', cache_path)

    with open(list_path, 'r') as fp:
        lines = fp.readlines()

    total = 0
    missing_cache = 0
    missing_mask = 0
    matched = 0
    mismatched = 0
    mismatched_examples = []
    updated_cache = {} if (args.fix or cache is None) else dict(cache)

    for line in tqdm.tqdm(lines):
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        total += 1
        img_rel = parts[0].lstrip('/')
        mask_rel = parts[1].lstrip('/')
        mask_path = os.path.join(root, mask_rel)
        if not os.path.exists(mask_path):
            # try alternate
            mask_path = os.path.join(root, mask_rel.lstrip('/'))
        if not os.path.exists(mask_path):
            missing_mask += 1
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        expected = build_points_from_mask(mask)

        cache_entry = find_cache_entry(cache, img_rel) if cache is not None else None
        if cache_entry is None:
            missing_cache += 1
            # if fix requested, add expected
            if args.fix:
                updated_cache[img_rel.replace('\\', '/').lstrip('/')] = expected.tolist()
            continue

        match, mismatch_count, diff_map = compare_points(np.array(cache_entry), expected, tol=args.tol)
        if match:
            matched += 1
        else:
            mismatched += 1
            if args.verbose and len(mismatched_examples) < 5:
                mismatched_examples.append((img_rel, mismatch_count, diff_map, cache_entry, expected.tolist()))
            if args.fix:
                # overwrite with expected
                updated_cache_key = img_rel.replace('\\', '/').lstrip('/')
                updated_cache[updated_cache_key] = expected.tolist()

    print('\nSummary:')
    print(f'  total processed: {total}')
    print(f'  matched: {matched}')
    print(f'  mismatched: {mismatched}')
    print(f'  missing in cache: {missing_cache}')
    print(f'  missing masks: {missing_mask}')

    if mismatched > 0 and args.verbose:
        print('\nSample mismatches (up to 5):')
        for img_rel, mismatch_count, diff_map, cached, expected in mismatched_examples:
            print(f' - {img_rel}: mismatches={mismatch_count}')
            # show coordinates where mismatched
            diff_idx = np.argwhere(diff_map)
            print('   example mismatched indices (lane,anchor):', diff_idx[:10].tolist())

    if args.fix:
        out_path = os.path.join(root, args.out)
        # backup existing cache if present
        if os.path.exists(cache_path):
            bak = cache_path + '.bak.' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            try:
                os.rename(cache_path, bak)
                print('Backed up existing cache to', bak)
            except Exception as e:
                print('Failed to backup existing cache:', e)
        with open(out_path, 'w') as f:
            json.dump(updated_cache, f)
        print('Wrote fixed cache to:', out_path)


if __name__ == '__main__':
    main()
