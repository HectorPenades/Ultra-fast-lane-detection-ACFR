
from data.dataloader import get_test_loader
from evaluation.tusimple.lane2 import LaneEval
from utils.dist_utils import is_main_process, dist_print, get_rank, get_world_size, dist_tqdm, synchronize
import os, json, torch, scipy
import numpy as np
import platform
from scipy.optimize import leastsq
import time
from data.constant import culane_col_anchor, culane_row_anchor
from PIL import Image
import matplotlib

# Use a non-interactive backend to avoid Tkinter-related crashes when workers
# or non-main threads create figures. Agg is a safe choice for saving images.
matplotlib.use('Agg')

# Enable/disable verbose evaluation debug prints. Default: off. Can be set
# via environment variable UFLD_EVAL_DEBUG=1 or by setting cfg.eval_debug
# (True/False) before calling `eval_lane`.
import os as _os
EVAL_DEBUG = _os.environ.get('UFLD_EVAL_DEBUG', '0') == '1'

def eprint(*args, **kwargs):
    """Conditional wrapper around dist_print used for noisy evaluation debug.
    Use `EVAL_DEBUG = True` to enable these messages.
    """
    if EVAL_DEBUG:
        try:
            dist_print(*args, **kwargs)
        except Exception:
            # Best-effort printing; don't raise from debug logger
            pass


def _prepare_disp_img(img_vis, force_swap=False):
    """Return a HxWx3 uint8 RGB image for matplotlib display.
    Accepts float [0,1] or uint8 [0,255], handles grayscale or RGBA, and
    applies a conservative BGR->RGB heuristic swap when channel means indicate BGR.
    """
    arr = np.asarray(img_vis)
    # convert floats in [0,1] to uint8
    if arr.dtype != np.uint8:
        try:
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).astype('uint8')
        except Exception:
            # fallback: try a direct astype
            arr = arr.astype('uint8')

    # ensure 3 channels
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]

    # conservative BGR->RGB heuristic: if green dominates both other channels,
    # it's likely the image is actually BGR stored as (B,G,R) -> swap for display
    try:
        r_mean = float(arr[..., 0].mean())
        g_mean = float(arr[..., 1].mean())
        b_mean = float(arr[..., 2].mean())
        eprint(f"[eval vis] img channel means R:{r_mean:.1f} G:{g_mean:.1f} B:{b_mean:.1f}")
        if force_swap:
            eprint("[eval vis] force-swap enabled: swapping channels for display (BGR->RGB)")
            arr = arr[..., ::-1]
        else:
            # Auto-swap is disabled by default to avoid incorrectly swapping RGB images.
            # We still log strong channel imbalances so you can decide to force-swap.
            if g_mean > r_mean * 1.5 and g_mean > b_mean * 1.5:
                eprint("[eval vis] channel means suggest green-dominant image (possible BGR)." +
                       " If the image appears wrong, call _prepare_disp_img(..., force_swap=True).")
            elif b_mean > r_mean * 1.5 and b_mean > g_mean * 1.5:
                eprint("[eval vis] channel means suggest blue-dominant image (possible BGR)." +
                       " If the image appears wrong, call _prepare_disp_img(..., force_swap=True).")
    except Exception:
        pass

    return arr

def _sanitize_lane_list(desired, num_lane):
    # Keep only indices that exist in the current model output; if none left, fall back to all lanes
    filtered = [i for i in desired if 0 <= i < num_lane]
    if len(filtered) == 0:
        return list(range(num_lane))
    return filtered

def generate_lines(out, out_ext, shape, names, output_path, griding_num, localization_type='abs', flip_updown=False):

    grid = torch.arange(out.shape[1]) + 0.5
    grid = grid.view(1,-1,1,1).cuda()
    loc = (out.softmax(1) * grid).sum(1) 
    
    loc = loc / (out.shape[1]-1) * 1640
    # n, num_cls, num_lanes
    valid = out_ext.argmax(1)
    # n, num_cls, num_lanes
    valid = valid.cpu()
    loc = loc.cpu()

    num_lane = out.shape[-1]
    lane_list = _sanitize_lane_list([1, 2], num_lane)

    for j in range(valid.shape[0]):

        name = names[j]
        line_save_path = os.path.join(output_path, name[:-3] + 'lines.txt')
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(line_save_path, 'w') as fp:
            # culane_row_anchor in data.constant.py is defined in the 288px
            # resized-space. Convert it back to original image height (590)
            # before writing.
            for i in lane_list:
                if valid[j,:,i].sum() > 2:
                    for k in range(valid.shape[1]):
                        if valid[j,k,i]:
                            row_y = (culane_row_anchor[k] / 288.0) * 590
                            fp.write('%.3f %.3f '% ( loc[j,k,i] , row_y))
                    fp.write('\n')

def generate_lines_col(out_col,out_col_ext, shape, names, output_path, griding_num, localization_type='abs', flip_updown=False):
    
    grid = torch.arange(out_col.shape[1]) + 0.5
    grid = grid.view(1,-1,1,1).cuda()
    loc = (out_col.softmax(1) * grid).sum(1) 
    
    loc = loc / (out_col.shape[1]-1) * 590
    # n, num_cls, num_lanes
    valid = out_col_ext.argmax(1)
    # n, num_cls, num_lanes
    valid = valid.cpu()
    loc = loc.cpu()

    num_lane = out_col.shape[-1]
    lane_list = _sanitize_lane_list([0, 3], num_lane)

    for j in range(valid.shape[0]):

        name = names[j]
        line_save_path = os.path.join(output_path, name[:-3] + 'lines.txt')
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(line_save_path, 'a') as fp:
            # culane_col_anchor is defined on the 800px resized width; map to
            # original image width (1640)
            for i in lane_list:
                if valid[j,:,i].sum() > 2:
                    for k in range(valid.shape[1]):
                        if valid[j,k,i]:
                            col_x = (culane_col_anchor[k] / 800.0) * 1640
                            fp.write('%.3f %.3f '% ( col_x, loc[j,k,i] ))
                    fp.write('\n')

def generate_lines_local(dataset, out, out_ext, names, output_path, mode='normal', row_anchor = None):
    batch_size, num_grid_row, num_cls, num_lane = out.shape
    max_indices = out.argmax(1).cpu()
    # n , num_cls, num_lanes
    
    valid = out_ext.argmax(1).cpu()
    # n, num_cls, num_lanes
    out = out.cpu()

    if mode == 'normal' or mode == '2row2col':
        if dataset == 'CULane':
            lane_list = [1, 2]
        elif dataset == 'CurveLanes':
            # lane_list = [2, 3, 4, 5, 6, 7]
            lane_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    else:
        lane_list = range(num_lane)

    # ensure lane indices are valid for this model output
    lane_list = _sanitize_lane_list(list(lane_list), num_lane)

    local_width = 1
    for j in range(valid.shape[0]):

        name = names[j]
        line_save_path = os.path.join(output_path, name[:-3] + 'lines.txt')
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir) 
        with open(line_save_path, 'w') as fp:
            # for i in range(num_lane):
            for i in lane_list:
                if valid[j,:,i].sum() > num_cls / 2:
                    for k in range(valid.shape[1]):
                        if valid[j,k,i]:
                            all_ind = torch.tensor(list(range(max(0,max_indices[j,k,i] - local_width), min(out.shape[1]-1, max_indices[j,k,i] + local_width) + 1)))
                            
                            out_tmp = (out[j,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5 

                            if dataset == 'CULane':
                                out_tmp = out_tmp / (out.shape[1]-1) * 1640
                                fp.write('%.3f %.3f '% ( out_tmp , row_anchor[k] * 590))
                            elif dataset == 'CurveLanes':
                                out_tmp = out_tmp / (out.shape[1]-1) * 2560
                                fp.write('%.3f %.3f '% ( out_tmp , row_anchor[k] * 1440))
                            else:
                                raise Exception
                    fp.write('\n')
                elif mode == 'all':
                    fp.write('\n')

def generate_lines_col_local(dataset, out_col,out_col_ext, names, output_path, mode='normal', col_anchor = None):
    batch_size, num_grid_col, num_cls, num_lane = out_col.shape
    max_indices = out_col.argmax(1).cpu()
    # n, num_cls, num_lanes
    valid = out_col_ext.argmax(1).cpu()
    # n, num_cls, num_lanes
    out_col = out_col.cpu()
    local_width = 1

    if mode == 'normal' or mode == '2row2col':
        if dataset == 'CULane':
            lane_list = [0, 3]
        elif dataset == 'CurveLanes':
            # lane_list = [0, 1, 8, 9]
            lane_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    else:
        lane_list = range(num_lane)

    # ensure lane indices are valid for this model output
    lane_list = _sanitize_lane_list(list(lane_list), num_lane)

    for j in range(valid.shape[0]):

        name = names[j]
        line_save_path = os.path.join(output_path, name[:-3] + 'lines.txt')
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(line_save_path, 'a') as fp:
            # for i in range(num_lane):
            for i in lane_list:
                if valid[j,:,i].sum() > num_cls / 4:
                    for k in range(valid.shape[1]):
                        if valid[j,k,i]:
                            all_ind = torch.tensor(list(range(max(0,max_indices[j,k,i] - local_width), min(out_col.shape[1]-1, max_indices[j,k,i] + local_width) + 1)))
                            out_tmp = (out_col[j,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5 
                            if dataset == 'CULane':
                                out_tmp = out_tmp / (out_col.shape[1]-1) * 590
                                fp.write('%.3f %.3f '% ( col_anchor[k] * 1640, out_tmp ))
                            elif dataset == 'CurveLanes':
                                out_tmp = out_tmp / (out_col.shape[1]-1) * 1440
                                fp.write('%.3f %.3f '% ( col_anchor[k] * 2560, out_tmp ))
                            else:
                                raise Exception

                    fp.write('\n')
                elif mode == 'all':
                    fp.write('\n')

def generate_lines_local_curve_combine(dataset, out, out_ext, names, output_path, mode='normal', row_anchor = None):
    batch_size, num_grid_row, num_cls, num_lane = out.shape
    max_indices = out.argmax(1).cpu()
    # n , num_cls, num_lanes
    
    valid = out_ext.argmax(1).cpu()
    # n, num_cls, num_lanes
    out = out.cpu()

    if mode == 'normal' or mode == '2row2col':
        if dataset == 'CULane':
            lane_list = [1, 2]
        elif dataset == 'CurveLanes':
            # lane_list = [2, 3, 4, 5, 6, 7]
            lane_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    else:
        lane_list = range(num_lane)

    # ensure lane indices are valid for this model output
    lane_list = _sanitize_lane_list(list(lane_list), num_lane)

    local_width = 1
    for j in range(valid.shape[0]):

        # import pdb; pdb.set_trace()

        name = names[j]
        line_save_path = os.path.join(output_path, name[:-3] + 'lines_row.txt')
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir) 
        with open(line_save_path, 'w') as fp:
            # for i in range(num_lane):
            for i in lane_list:
                if valid[j,:,i].sum() > num_cls / 4:
                    for k in range(valid.shape[1]):
                        if valid[j,k,i]:
                            all_ind = torch.tensor(list(range(max(0,max_indices[j,k,i] - local_width), min(out.shape[1]-1, max_indices[j,k,i] + local_width) + 1)))
                            out_tmp = (out[j,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5 
                            if dataset == 'CULane':
                                out_tmp = out_tmp / (out.shape[1]-1) * 1640
                                fp.write('%.3f %.3f '% ( out_tmp , row_anchor[k] * 590))
                            elif dataset == 'CurveLanes':
                                out_tmp = out_tmp / (out.shape[1]-1) * 2560
                                fp.write('%.3f %.3f '% ( out_tmp , row_anchor[k] * 1440))
                            else:
                                raise Exception
                    fp.write('\n')
                else:
                    fp.write('\n')

def generate_lines_col_local_curve_combine(dataset, out_col,out_col_ext, names, output_path, mode='normal', col_anchor = None):
    batch_size, num_grid_col, num_cls, num_lane = out_col.shape
    max_indices = out_col.argmax(1).cpu()
    # n, num_cls, num_lanes
    valid = out_col_ext.argmax(1).cpu()
    # n, num_cls, num_lanes
    out_col = out_col.cpu()
    local_width = 1

    if mode == 'normal' or mode == '2row2col':
        if dataset == 'CULane':
            lane_list = [0, 3]
        elif dataset == 'CurveLanes':
            # lane_list = [0, 1, 8, 9]
            lane_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    else:
        lane_list = range(num_lane)

    # ensure lane indices are valid for this model output
    lane_list = _sanitize_lane_list(list(lane_list), num_lane)

    for j in range(valid.shape[0]):
        # import pdb; pdb.set_trace()

        name = names[j]
        line_save_path = os.path.join(output_path, name[:-3] + 'lines_col.txt')
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(line_save_path, 'w') as fp:
            # for i in range(num_lane):
            for i in lane_list:
                if valid[j,:,i].sum() > num_cls / 4:
                    for k in range(valid.shape[1]):
                        if valid[j,k,i]:
                            all_ind = torch.tensor(list(range(max(0,max_indices[j,k,i] - local_width), min(out_col.shape[1]-1, max_indices[j,k,i] + local_width) + 1)))
                            out_tmp = (out_col[j,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5 
                            if dataset == 'CULane':
                                out_tmp = out_tmp / (out_col.shape[1]-1) * 590
                                fp.write('%.3f %.3f '% ( col_anchor[k] * 1640, out_tmp ))
                            elif dataset == 'CurveLanes':
                                out_tmp = out_tmp / (out_col.shape[1]-1) * 1440
                                fp.write('%.3f %.3f '% ( col_anchor[k] * 2560, out_tmp ))
                            else:
                                raise Exception

                    fp.write('\n')
                # elif mode == 'all':
                #     fp.write('\n')
                else:
                    fp.write('\n')

def revise_lines_curve_combine(names, output_path):
    for name in names:
        line_save_path = os.path.join(output_path, name[:-3] + 'lines.txt')
        row_line_save_path = os.path.join(output_path, name[:-3] + 'lines_row.txt')
        col_line_save_path = os.path.join(output_path, name[:-3] + 'lines_col.txt')
        if not os.path.exists(row_line_save_path):
            continue
        if not os.path.exists(col_line_save_path):
            continue
        with open(row_line_save_path, 'r') as fp:
            row_lines = fp.readlines()
        with open(col_line_save_path, 'r') as fp:
            col_lines = fp.readlines()
        flag = True
        for i in range(10):
            x1, y1 = coordinate_parse(row_lines[i])
            x2, y2 = coordinate_parse(col_lines[i])
            x = x1 + x2
            y = y1 + y2
            if x == [] or y == []:
                continue
            x = np.array(x)
            y = np.array(y)

            p_init = np.random.randn(3)
            para_x = leastsq(resudual, p_init, args=(x, y))
            y_temp = func(para_x[0], x)
            y_error = np.mean(np.square(y_temp-y))

            para_y = leastsq(resudual, p_init, args=(y, x))
            x_temp = func(para_y[0], y)
            x_error = np.mean(np.square(x_temp-x))

            if x_error > y_error:
                x_new = np.linspace(min(x), max(x), 36)
                y_new = func(para_x[0], x_new)
            else:
                y_new = np.linspace(min(y), max(y), 41)
                x_new = func(para_y[0], y_new)

            if flag:
                fp = open(line_save_path, 'w')
                flag = False
            else:
                fp = open(line_save_path, 'a')
            for i in range(x_new.shape[0]):
                fp.write('%.3f %.3f '% ( x_new[i], y_new[i] ))
            fp.write('\n')
            fp.close()
        if flag:
            fp = open(line_save_path, 'w')
            fp.close()

def generate_lines_reg(out, out_ext, names, output_path, mode='normal', row_anchor = None):
    batch_size, num_grid_row, num_cls, num_lane = out.shape
    # n , num_cls, num_lanes
    
    valid = out_ext.argmax(1).cpu()
    # n, num_cls, num_lanes
    out = out.cpu().sigmoid()

    if mode == 'normal' or mode == '2row2col':
        lane_list = _sanitize_lane_list([1, 2], num_lane)
    else:
        lane_list = range(num_lane)

    local_width = 1
    for j in range(valid.shape[0]):

        name = names[j]
        line_save_path = os.path.join(output_path, name[:-3] + 'lines.txt')
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(line_save_path, 'w') as fp:
            # for i in range(num_lane):
            for i in lane_list:
                if valid[j,:,i].sum() > num_cls / 2:
                    for k in range(valid.shape[1]):
                        if valid[j,k,i]:
                            # all_ind = torch.tensor(list(range(max(0,max_indices[j,k,i] - local_width), min(out.shape[1]-1, max_indices[j,k,i] + local_width) + 1)))
                            
                            out_tmp = out[j,0,k,i] * 1640

                            fp.write('%.3f %.3f '% ( out_tmp , row_anchor[k] * 590))
                    fp.write('\n')
                elif mode == 'all':
                    fp.write('\n')

def generate_lines_col_reg(out_col,out_col_ext, names, output_path, mode='normal', col_anchor = None):
    batch_size, num_grid_col, num_cls, num_lane = out_col.shape
    # max_indices = out_col.argmax(1).cpu()
    # n, num_cls, num_lanes
    valid = out_col_ext.argmax(1).cpu()
    # n, num_cls, num_lanes
    out_col = out_col.cpu().sigmoid()
    local_width = 1

    if mode == 'normal' or mode == '2row2col':
        lane_list = _sanitize_lane_list([0, 3], num_lane)
    else:
        lane_list = range(num_lane)

    for j in range(valid.shape[0]):

        name = names[j]
        line_save_path = os.path.join(output_path, name[:-3] + 'lines.txt')
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(line_save_path, 'a') as fp:
            # for i in range(num_lane):
            for i in lane_list:
                if valid[j,:,i].sum() > num_cls / 4:
                    for k in range(valid.shape[1]):
                        if valid[j,k,i]:
                            # all_ind = torch.tensor(list(range(max(0,max_indices[j,k,i] - local_width), min(out_col.shape[1]-1, max_indices[j,k,i] + local_width) + 1)))
                            # out_tmp = (out_col[j,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5 
                            out_tmp = out_col[j,0,k,i] * 590
                            fp.write('%.3f %.3f '% ( col_anchor[k] * 1640, out_tmp ))
                    fp.write('\n')
                elif mode == 'all':
                    fp.write('\n')

def coordinate_parse(line):
    if line == '\n':
        return [], []

    items = line.split(' ')[:-1]
    x = [float(items[2*i]) for i in range(len(items)//2)]
    y = [float(items[2*i+1]) for i in range(len(items)//2)]

    return x, y


def func(p, x):
    f = np.poly1d(p)
    return f(x)


def resudual(p, x, y):
    error = y - func(p, x)
    return error


def revise_lines(names, output_path):
    for name in names:
        line_save_path = os.path.join(output_path, name[:-3] + 'lines.txt')
        if not os.path.exists(line_save_path):
            continue
        with open(line_save_path, 'r') as fp:
            lines = fp.readlines()
        flag = True
        for i in range(4):
            x1, y1 = coordinate_parse(lines[i])
            x2, y2 = coordinate_parse(lines[i+4])
            x = x1 + x2
            y = y1 + y2
            if x == [] or y == []:
                continue
            x = np.array(x)
            y = np.array(y)

            p_init = np.random.randn(3)
            para_x = leastsq(resudual, p_init, args=(x, y))
            y_temp = func(para_x[0], x)
            y_error = np.mean(np.square(y_temp-y))

            para_y = leastsq(resudual, p_init, args=(y, x))
            x_temp = func(para_y[0], y)
            x_error = np.mean(np.square(x_temp-x))

            if x_error > y_error:
                x_new = np.linspace(min(x), max(x), 18)
                y_new = func(para_x[0], x_new)
            else:
                y_new = np.linspace(min(y), max(y), 41)
                x_new = func(para_y[0], y_new)

            if flag:
                fp = open(line_save_path, 'w')
                flag = False
            else:
                fp = open(line_save_path, 'a')
            for i in range(x_new.shape[0]):
                fp.write('%.3f %.3f '% ( x_new[i], y_new[i] ))
            fp.write('\n')
            fp.close()
        if flag:
            fp = open(line_save_path, 'w')
            fp.close()
            

def rectify_lines(names, output_path):
    for name in names:
        line_save_path = os.path.join(output_path, name[:-3] + 'lines.txt')
        if not os.path.exists(line_save_path):
            continue
        with open(line_save_path, 'r') as fp:
            lines = fp.readlines()
        flag = True
        for line in lines:
            x, y = coordinate_parse(line)
            if x == [] or y == []:
                continue
            x = np.array(x)
            y = np.array(y)

            p_init = np.random.randn(3)
            para_x = leastsq(resudual, p_init, args=(x, y))
            y_temp = func(para_x[0], x)
            y_error = np.mean(np.square(y_temp-y))

            para_y = leastsq(resudual, p_init, args=(y, x))
            x_temp = func(para_y[0], y)
            x_error = np.mean(np.square(x_temp-x))

            if x_error > y_error:
                x_new = np.linspace(min(x), max(x), 18)
                y_new = func(para_x[0], x_new)
            else:
                y_new = np.linspace(min(y), max(y), 41)
                x_new = func(para_y[0], y_new)

            if flag:
                fp = open(line_save_path, 'w')
                flag = False
            else:
                fp = open(line_save_path, 'a')
            for i in range(x_new.shape[0]):
                fp.write('%.3f %.3f '% ( x_new[i], y_new[i] ))
            fp.write('\n')
            fp.close()
        if flag:
            fp = open(line_save_path, 'w')
            fp.close()


def run_test(dataset, net, data_root, exp_name, work_dir, distributed, crop_ratio, train_width, train_height , batch_size=8, row_anchor = None, col_anchor = None, logger=None):
    # torch.backends.cudnn.benchmark = True
    output_path = os.path.join(work_dir, exp_name)
    if not os.path.exists(output_path) and is_main_process():
        os.mkdir(output_path)
    synchronize()
    loader = get_test_loader(batch_size, data_root, dataset, distributed, crop_ratio, train_width, train_height)
    # import pdb;pdb.set_trace()
    for i, data in enumerate(dist_tqdm(loader)):
        imgs, names = data
        imgs = imgs.cuda()
        with torch.no_grad():
            pred = net(imgs)

        # Precompute a shared display image for this batch when we will log visuals.
        # This ensures `eval/input_with_pred` and `eval/input_with_lines` use the
        # exact same disp_img array (float img_vis and uint8 disp_img_show_shared).
        disp_img_show_shared = None
        img_vis_shared = None
        if logger is not None and is_main_process() and (i % 20 == 0):
            try:
                img_sh = imgs[0].cpu().numpy()
                if img_sh.ndim == 3:
                    img_sh = np.transpose(img_sh, (1,2,0))
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_vis_shared = (img_sh * std[None,None,:]) + mean[None,None,:]
                img_vis_shared = np.clip(img_vis_shared, 0, 1)
                # uint8 RGB prepared once
                disp_img_show_shared = _prepare_disp_img(img_vis_shared)
            except Exception:
                disp_img_show_shared = None
                img_vis_shared = None
        # Optionally log a visualisation of the first image of this batch to TensorBoard
        try:
            if logger is not None and is_main_process() and (i % 20 == 0):
                # prepare image for display (reuse precomputed shared image when available)
                if img_vis_shared is not None and disp_img_show_shared is not None:
                    img_vis = img_vis_shared
                    disp_img_show = disp_img_show_shared
                    H, W = img_vis.shape[:2]
                else:
                    # fallback: compute per-batch
                    img = imgs[0].cpu().numpy()
                    if img.ndim == 3:
                        img = np.transpose(img, (1,2,0))
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img_vis = (img * std[None,None,:]) + mean[None,None,:]
                    img_vis = np.clip(img_vis, 0, 1)
                    H, W = img_vis.shape[:2]

                # compute predicted row/col points from tensors
                pts = []
                if 'loc_row' in pred and 'exist_row' in pred:
                    out = pred['loc_row'].cpu()
                    out_ext = pred['exist_row'].cpu()
                    grid = torch.arange(out.shape[1]) + 0.5
                    grid = grid.view(1,-1,1,1)
                    loc = (out.softmax(1) * grid).sum(1)
                    # map loc from grid to resized width
                    loc = loc / (out.shape[1]-1) * (W - 1)
                    valid = out_ext.argmax(1).cpu()
                    # iterate anchors
                    num_cls = valid.shape[1]
                    num_lane = valid.shape[2]
                    for k in range(num_cls):
                        y = None
                        # row_anchor may be normalized or in pixel space; prefer provided row_anchor
                        if row_anchor is not None:
                            y = row_anchor[k] * (H - 1)
                        else:
                            # fallback to culane_row_anchor mapping (assumed 288px) scaled to H
                            y = (culane_row_anchor[k] / 288.0) * (H - 1)
                        for lane_idx in range(num_lane):
                            if valid[0,k,lane_idx]:
                                x = float(loc[0,k,lane_idx])
                                pts.append(('row', (x, y)))
                # columns (x fixed from anchors, y from loc_col)
                if 'loc_col' in pred and 'exist_col' in pred:
                    outc = pred['loc_col'].cpu()
                    outc_ext = pred['exist_col'].cpu()
                    gridc = torch.arange(outc.shape[1]) + 0.5
                    gridc = gridc.view(1,-1,1,1)
                    locc = (outc.softmax(1) * gridc).sum(1)
                    locc = locc / (outc.shape[1]-1) * (H - 1)
                    validc = outc_ext.argmax(1).cpu()
                    num_cls_c = validc.shape[1]
                    num_lane_c = validc.shape[2]
                    for k in range(num_cls_c):
                        if col_anchor is not None:
                            x_anchor = col_anchor[k] * (W - 1)
                        else:
                            x_anchor = (culane_col_anchor[k] / 800.0) * (W - 1)
                        for lane_idx in range(num_lane_c):
                            if validc[0,k,lane_idx]:
                                y = float(locc[0,k,lane_idx])
                                pts.append(('col', (x_anchor, y)))

                # draw using matplotlib and log
                import matplotlib.pyplot as plt
                fig = plt.figure(figsize=(6,4))
                ax = fig.add_subplot(1,1,1)
                # draw image as background (lowest zorder)
                disp_img = img_vis
                # prepare a robust uint8 RGB image and apply conservative BGR->RGB heuristic
                disp_img_show = _prepare_disp_img(disp_img)
                ax.imshow(disp_img_show, zorder=0, interpolation='nearest')
                ax.axis('off')
                for t,p in pts:
                    x,y = p
                    if t == 'row':
                        ax.plot(x, y, 'ro', markersize=4, zorder=2)
                    else:
                        ax.plot(x, y, 'bx', markersize=4, zorder=2)
                logger.add_figure('eval/input_with_pred', fig, i)
                plt.close(fig)
                # Quick debug summary for diagnosis: print sample pts and simple stats
                try:
                    # only print from main process
                    if is_main_process():
                        col_ys = [p[1] for t,p in pts if t == 'col']
                        row_ys = [p[1] for t,p in pts if t == 'row']
                        eprint(f"[eval vis] sample pts (first 10): {pts[:10]}")
                        if len(col_ys) > 0:
                            col_min, col_max = min(col_ys), max(col_ys)
                            distinct_col = len(set([int(round(v)) for v in col_ys]))
                            eprint(f"[eval vis] col y -> min:{col_min:.1f} max:{col_max:.1f} distinct_px:{distinct_col}")
                        if len(row_ys) > 0:
                            row_min, row_max = min(row_ys), max(row_ys)
                            distinct_row = len(set([int(round(v)) for v in row_ys]))
                            eprint(f"[eval vis] row y -> min:{row_min:.1f} max:{row_max:.1f} distinct_px:{distinct_row}")
                except Exception:
                    pass
        except Exception:
            pass
        # debug: print shapes of prediction tensors on first batch to help diagnose mismatch
        if i == 0 and is_main_process():
            try:
                dr = pred.get('loc_row')
                er = pred.get('exist_row')
                dc = pred.get('loc_col')
                ec = pred.get('exist_col')
                eprint('Pred shapes: loc_row {}, exist_row {}, loc_col {}, exist_col {}'.format(
                    None if dr is None else tuple(dr.shape),
                    None if er is None else tuple(er.shape),
                    None if dc is None else tuple(dc.shape),
                    None if ec is None else tuple(ec.shape)
                ))
            except Exception:
                eprint('Could not read pred shapes for debug')
        
        if dataset == "CULane":
            generate_lines_local(dataset, pred['loc_row'],pred['exist_row'], names, output_path, 'normal', row_anchor=row_anchor)
            generate_lines_col_local(dataset, pred['loc_col'],pred['exist_col'], names, output_path, 'normal', col_anchor=col_anchor)
        elif dataset == 'CurveLanes':
            generate_lines_local_curve_combine(dataset, pred['loc_row'],pred['exist_row'], names, output_path, row_anchor=row_anchor)
            generate_lines_col_local_curve_combine(dataset, pred['loc_col'],pred['exist_col'], names, output_path, col_anchor=col_anchor)
            revise_lines_curve_combine(names, output_path)
        else:
            raise NotImplementedError

        # Additionally, visualise the generated lines.txt (predicted lanes) over the image
        # This shows exactly the files used by the evaluator. Only run in main process and when a
        # TensorBoard logger is provided.
        if logger is not None and is_main_process() and (i % 20 == 0):
            try:
                # reuse the same precomputed display image if present
                if img_vis_shared is not None and disp_img_show_shared is not None:
                    img_vis = img_vis_shared
                    disp_img_show = disp_img_show_shared
                    H, W = img_vis.shape[:2]
                else:
                    # rebuild display image (undo normalize)
                    img = imgs[0].cpu().numpy()
                    if img.ndim == 3:
                        img = np.transpose(img, (1,2,0))
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img_vis = (img * std[None,None,:]) + mean[None,None,:]
                    img_vis = np.clip(img_vis, 0, 1)
                    H, W = img_vis.shape[:2]

                # path to the lines file for the first image in the batch
                line_path = os.path.join(output_path, names[0][:-3] + 'lines.txt')
                if os.path.exists(line_path):
                    with open(line_path, 'r') as lf:
                        lines = lf.readlines()
                    # try to load an optional GT mask corresponding to this image
                    # (some datasets provide pixel-level masks; this is optional)
                    mask = None
                    try:
                        mask = _try_load_mask(data_root, names[0], H, W)
                        if mask is not None:
                            eprint(f"[eval vis] loaded GT mask for {names[0]} (uniq={len(np.unique(mask))})")
                    except Exception:
                        mask = None
                    # Try to load GT lines file that should be co-located with the image
                    # (same path but with extension 'lines.txt'), and draw those polylines
                    # in yellow so you can compare GT (yellow) vs predicted lines (green).
                    import matplotlib.pyplot as plt
                    fig = plt.figure(figsize=(6,4))
                    ax = fig.add_subplot(1,1,1)
                    # ensure the image is drawn underneath the line overlays
                    disp_img = img_vis
                    # prepare a robust uint8 RGB image (do not force channel swap here;
                    # `eval/input_with_pred` uses the same helper without forcing and
                    # that shows the image correctly for the user's dataset)
                    disp_img_show = _prepare_disp_img(disp_img)
                    # quick diagnostics: write out the exact array used for TensorBoard so
                    # you can open it locally and confirm the raw image is correct
                    try:
                        # always write a diagnostic image with batch index and timestamp so
                        # the file is easy to find and inspect locally
                        # sanitize the name so it cannot contain path separators
                        safe_name = names[0][:-3].replace('/', '_').replace('\\', '_')
                        dbg_name = f"debug_img_{safe_name}_batch{i}_{int(time.time())}.png"
                        dbg_path = os.path.join(output_path, dbg_name)
                        # ensure output directory exists (output_path should already exist,
                        # but be defensive if a nested path sneaks in)
                        os.makedirs(os.path.dirname(dbg_path), exist_ok=True)
                        from PIL import Image as PILImage
                        PILImage.fromarray(disp_img_show).save(dbg_path)
                        eprint(f"[eval vis lines] wrote debug image {dbg_path} (dtype={disp_img_show.dtype} shape={disp_img_show.shape} min={disp_img_show.min()} max={disp_img_show.max()})")
                    except Exception as e:
                        eprint(f"[eval vis lines] failed to write debug image: {e}")
                    ax.imshow(disp_img_show, zorder=0, interpolation='nearest')
                    ax.axis('off')
                    try:
                        gt_line_path = os.path.join(data_root, names[0][:-3] + 'lines.txt')
                        gt_count = 0
                        if os.path.exists(gt_line_path):
                            with open(gt_line_path, 'r') as gfl:
                                gt_lines = gfl.readlines()
                            for ln in gt_lines:
                                xs, ys = coordinate_parse(ln)
                                if len(xs) == 0:
                                    continue
                                # scale from original image space (1640x590) to display
                                xs_resized = [ (float(x) / 1640.0) * (W - 1) for x in xs ]
                                ys_resized = [ (float(y) / 590.0) * (H - 1) for y in ys ]
                                ax.plot(xs_resized, ys_resized, '-', color='y', linewidth=2, zorder=3)
                                ax.plot(xs_resized, ys_resized, 'y.', markersize=4, zorder=4)
                                gt_count += 1
                            eprint(f"[eval vis] loaded GT lines for {names[0]} (lines={gt_count})")
                    except Exception:
                        pass
                    for ln in lines:
                        xs, ys = coordinate_parse(ln)
                        if len(xs) == 0:
                            continue
                        # lines.txt coordinates are in original image space (1640x590)
                        xs_resized = [ (x / 1640.0) * (W - 1) for x in xs ]
                        ys_resized = [ (y / 590.0) * (H - 1) for y in ys ]
                        ax.plot(xs_resized, ys_resized, '-g', linewidth=2, label='pred', zorder=3)
                        ax.plot(xs_resized, ys_resized, 'go', markersize=3, zorder=4)
                        # if GT mask exists, optionally draw overlay points for GT lanes
                        # but only when the mask is sparse; dense masks will otherwise
                        # completely cover the image and make the background invisible.
                        if mask is not None:
                            try:
                                ys_gt, xs_gt = np.where(mask > 0)
                                num_mask_pts = len(xs_gt)
                                # threshold: only plot when reasonably sparse
                                SPARSE_MASK_THRESHOLD = 3000
                                if num_mask_pts > 0 and num_mask_pts <= SPARSE_MASK_THRESHOLD:
                                    xs_gt_res = xs_gt.astype(float)
                                    ys_gt_res = ys_gt.astype(float)
                                    # sparse mask: scatter is fine
                                    ax.scatter(xs_gt_res, ys_gt_res, c='c', s=1, alpha=0.6, zorder=1)
                                elif num_mask_pts > SPARSE_MASK_THRESHOLD:
                                    # dense mask: drawing per-pixel scatter will accumulate opacity and
                                    # completely hide the background. Use imshow with a small alpha
                                    # for a faint overlay that preserves the image underneath.
                                    try:
                                        mask_overlay = (mask > 0).astype(float)
                                        ax.imshow(mask_overlay, cmap='copper', alpha=0.12, zorder=1, interpolation='nearest')
                                        eprint(f"[eval vis] GT mask dense ({num_mask_pts} pixels); drawing with imshow low-alpha")
                                    except Exception:
                                        eprint("[eval vis] GT mask dense; failed to draw low-alpha mask")
                            except Exception:
                                pass
                    logger.add_figure('eval/input_with_lines', fig, i)
                    plt.close(fig)
            except Exception:
                pass

        # Diagnostic check: compare predicted points (from tensors) vs coordinates written
        # in the .lines.txt file for the first image in batch. This verifies what the
        # evaluator will read matches the in-memory predictions.
        try:
            if is_main_process() and logger is not None and (i % 20 == 0):
                # rebuild the predicted points list (same logic as earlier)
                pred_pts = []
                if 'loc_row' in pred and 'exist_row' in pred:
                    out = pred['loc_row'].cpu()
                    out_ext = pred['exist_row'].cpu()
                    grid = torch.arange(out.shape[1]) + 0.5
                    grid = grid.view(1,-1,1,1)
                    loc = (out.softmax(1) * grid).sum(1)
                    loc = loc / (out.shape[1]-1) * (W - 1)
                    valid = out_ext.argmax(1).cpu()
                    num_cls = valid.shape[1]
                    num_lane = valid.shape[2]
                    for k in range(num_cls):
                        if row_anchor is not None:
                            y = row_anchor[k] * (H - 1)
                        else:
                            y = (culane_row_anchor[k] / 288.0) * (H - 1)
                        for lane_idx in range(num_lane):
                            if valid[0,k,lane_idx]:
                                x = float(loc[0,k,lane_idx])
                                pred_pts.append((float(x), float(y)))
                if 'loc_col' in pred and 'exist_col' in pred:
                    outc = pred['loc_col'].cpu()
                    outc_ext = pred['exist_col'].cpu()
                    gridc = torch.arange(outc.shape[1]) + 0.5
                    gridc = gridc.view(1,-1,1,1)
                    locc = (outc.softmax(1) * gridc).sum(1)
                    locc = locc / (outc.shape[1]-1) * (H - 1)
                    validc = outc_ext.argmax(1).cpu()
                    num_cls_c = validc.shape[1]
                    num_lane_c = validc.shape[2]
                    for k in range(num_cls_c):
                        if col_anchor is not None:
                            x_anchor = col_anchor[k] * (W - 1)
                        else:
                            x_anchor = (culane_col_anchor[k] / 800.0) * (W - 1)
                        for lane_idx in range(num_lane_c):
                            if validc[0,k,lane_idx]:
                                y = float(locc[0,k,lane_idx])
                                pred_pts.append((float(x_anchor), float(y)))

                # Read points from file (scaled to display W,H)
                line_path = os.path.join(output_path, names[0][:-3] + 'lines.txt')
                file_pts = []
                if os.path.exists(line_path):
                    with open(line_path, 'r') as lf:
                        for ln in lf.readlines():
                            xs, ys = coordinate_parse(ln)
                            for x,y in zip(xs, ys):
                                # file coordinates are in original 1640x590 space
                                xr = (float(x) / 1640.0) * (W - 1)
                                yr = (float(y) / 590.0) * (H - 1)
                                file_pts.append((xr, yr))

                # compare sets: for each file point find nearest pred point
                import math, json
                diag = {'image': names[0], 'num_pred_pts': len(pred_pts), 'num_file_pts': len(file_pts), 'pairs': []}
                if len(file_pts) > 0 and len(pred_pts) > 0:
                    pred_arr = np.array(pred_pts)
                    for (fx,fy) in file_pts:
                        dists = np.sqrt(np.sum((pred_arr - np.array([fx,fy]))**2, axis=1))
                        min_d = float(dists.min())
                        argmin = int(dists.argmin())
                        diag['pairs'].append({'file_pt': [fx,fy], 'best_pred_idx': argmin, 'pred_pt': pred_pts[argmin], 'dist': min_d})
                    dists_all = [p['dist'] for p in diag['pairs']]
                    diag['max_dist'] = max(dists_all)
                    diag['mean_dist'] = float(np.mean(dists_all))
                else:
                    diag['max_dist'] = None
                    diag['mean_dist'] = None

                # print short summary and save JSON
                eprint(f"[eval diag] image={names[0]} pred_pts={len(pred_pts)} file_pts={len(file_pts)} max_dist={diag['max_dist']} mean_dist={diag['mean_dist']}")
                try:
                    diag_path = os.path.join(output_path, f"diag_{names[0][:-3]}.json")
                    with open(diag_path, 'w') as df:
                        json.dump(_sanitize_for_json(diag), df, indent=2)
                except Exception:
                    pass
        except Exception:
            pass


def generate_lines_local_tta(loc_row, loc_row_left, loc_row_right, exist_row, exist_row_left, exist_row_right, names, output_path, row_anchor):

    local_width = 1

    max_indices = loc_row.argmax(1).cpu()
    valid = exist_row.argmax(1).cpu()
    loc_row = loc_row.cpu()

    max_indices_left = loc_row_left.argmax(1).cpu()
    valid_left = exist_row_left.argmax(1).cpu()
    loc_row_left = loc_row_left.cpu()

    max_indices_right = loc_row_right.argmax(1).cpu()
    valid_right = exist_row_right.argmax(1).cpu()
    loc_row_right = loc_row_right.cpu()

    batch_size, num_grid, num_cls, num_lane = loc_row.shape

    min_lane_length = num_cls / 2

    for batch_idx in range(batch_size):

        name = names[batch_idx]
        line_save_path = os.path.join(output_path, name.replace('jpg', 'lines.txt'))
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(line_save_path, 'w') as fp:
            # for lane_idx in range(num_lane):
            for lane_idx in [1,2]:
                if valid[batch_idx,:,lane_idx].sum() >= min_lane_length:
                    pt_all = []
                    for cls_idx in range(num_cls):
                        cnt = 0
                        out_tmp_all = 0
                        if valid[batch_idx,cls_idx,lane_idx]:
                            all_ind = torch.tensor(list(range(max(0,max_indices[batch_idx,cls_idx,lane_idx] - local_width), min(num_grid-1, max_indices[batch_idx,cls_idx,lane_idx] + local_width) + 1)))
                            out_tmp = (loc_row[batch_idx,all_ind,cls_idx,lane_idx].softmax(0) * all_ind.float()).sum() + 0.5 
                            out_tmp = out_tmp / (num_grid-1) * 1640
                            cnt += 1
                            out_tmp_all = out_tmp_all + out_tmp

                        if valid_left[batch_idx,cls_idx,lane_idx]:
                            all_ind_left = torch.tensor(list(range(max(0,max_indices_left[batch_idx,cls_idx,lane_idx] - local_width), min(num_grid-1, max_indices_left[batch_idx,cls_idx,lane_idx] + local_width) + 1)))
                        
                            out_tmp_left = (loc_row_left[batch_idx,all_ind_left,cls_idx,lane_idx].softmax(0) * all_ind_left.float()).sum() + 0.5 
                            out_tmp_left = out_tmp_left / (num_grid-1) * 1640 + 1640./25
                            cnt += 1
                            out_tmp_all = out_tmp_all + out_tmp_left

                        if valid_right[batch_idx,cls_idx,lane_idx]:
                            all_ind_right = torch.tensor(list(range(max(0,max_indices_right[batch_idx,cls_idx,lane_idx] - local_width), min(num_grid-1, max_indices_right[batch_idx,cls_idx,lane_idx] + local_width) + 1)))
                        
                            out_tmp_right = (loc_row_right[batch_idx,all_ind_right,cls_idx,lane_idx].softmax(0) * all_ind_right.float()).sum() + 0.5 
                            out_tmp_right = out_tmp_right / (num_grid-1) * 1640 - 1640./25
                            cnt += 1
                            out_tmp_all = out_tmp_all + out_tmp_right


                        if cnt >= 2:
                            pt_all.append(( out_tmp_all/cnt , row_anchor[cls_idx] * 590))
                    if len(pt_all) < min_lane_length:
                            continue
                    for pt in pt_all:
                        fp.write('%.3f %.3f '% pt)
                    fp.write('\n')

def generate_lines_col_local_tta(loc_col, loc_col_up, loc_col_down, exist_col, exist_col_up, exist_col_down, names, output_path, col_anchor):
    local_width = 1
    
    max_indices = loc_col.argmax(1).cpu()
    valid = exist_col.argmax(1).cpu()
    loc_col = loc_col.cpu()

    max_indices_up = loc_col_up.argmax(1).cpu()
    valid_up = exist_col_up.argmax(1).cpu()
    loc_col_up = loc_col_up.cpu()

    max_indices_down = loc_col_down.argmax(1).cpu()
    valid_down = exist_col_down.argmax(1).cpu()
    loc_col_down = loc_col_down.cpu()

    batch_size, num_grid, num_cls, num_lane = loc_col.shape

    min_lane_length = num_cls / 4

    for batch_idx in range(batch_size):

        name = names[batch_idx]
        line_save_path = os.path.join(output_path, name.replace('jpg','lines.txt'))
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(line_save_path, 'a') as fp:
            # for lane_idx in range(num_lane):
            for lane_idx in [0,3]:
                if valid[batch_idx,:,lane_idx].sum() >= min_lane_length:
                    pt_all = []
                    for cls_idx in range(num_cls):
                        cnt = 0
                        out_tmp_all = 0
                        if valid[batch_idx,cls_idx,lane_idx]:
                            all_ind = torch.tensor(list(range(max(0,max_indices[batch_idx,cls_idx,lane_idx] - local_width), min(num_grid-1, max_indices[batch_idx,cls_idx,lane_idx] + local_width) + 1)))
                            out_tmp = (loc_col[batch_idx,all_ind,cls_idx,lane_idx].softmax(0) * all_ind.float()).sum() + 0.5 
                            out_tmp = out_tmp / (num_grid-1) * 590
                            cnt += 1
                            out_tmp_all = out_tmp_all + out_tmp

                        if valid_up[batch_idx,cls_idx,lane_idx]:
                            all_ind_up = torch.tensor(list(range(max(0,max_indices_up[batch_idx,cls_idx,lane_idx] - local_width), min(num_grid-1, max_indices_up[batch_idx,cls_idx,lane_idx] + local_width) + 1)))
                            out_tmp_up = (loc_col_up[batch_idx,all_ind_up,cls_idx,lane_idx].softmax(0) * all_ind_up.float()).sum() + 0.5 
                            out_tmp_up = out_tmp_up / (num_grid-1) * 590 + 32./534*590
                            cnt += 1
                            out_tmp_all = out_tmp_all + out_tmp_up
                        if valid_down[batch_idx,cls_idx,lane_idx]:
                            all_ind_down = torch.tensor(list(range(max(0,max_indices_down[batch_idx,cls_idx,lane_idx] - local_width), min(num_grid-1, max_indices_down[batch_idx,cls_idx,lane_idx] + local_width) + 1)))
                            out_tmp_down = (loc_col_down[batch_idx,all_ind_down,cls_idx,lane_idx].softmax(0) * all_ind_down.float()).sum() + 0.5 
                            out_tmp_down = out_tmp_down / (num_grid-1) * 590 - 32./534*590     
                            cnt += 1
                            out_tmp_all = out_tmp_all + out_tmp_down

                        if cnt >= 2:
                            pt_all.append(( col_anchor[cls_idx] * 1640, out_tmp_all/cnt ))
                    if len(pt_all) < min_lane_length:
                        continue
                    for pt in pt_all:
                        fp.write('%.3f %.3f '% pt)
                    fp.write('\n')

def run_test_tta(dataset, net, data_root, exp_name, work_dir,distributed, crop_ratio, train_width, train_height, batch_size=8, row_anchor = None, col_anchor = None):
    output_path = os.path.join(work_dir, exp_name)
    if not os.path.exists(output_path) and is_main_process():
        os.mkdir(output_path)
    synchronize()
    loader = get_test_loader(batch_size, data_root, dataset, distributed, crop_ratio, train_width, train_height)
    # import pdb;pdb.set_trace()
    for i, data in enumerate(dist_tqdm(loader)):
        imgs, names = data
        imgs = imgs.cuda()
        with torch.no_grad():
            if hasattr(net, 'module'):
                pred = net.module.forward_tta(imgs)
            else:
                pred = net.forward_tta(imgs)

            loc_row, loc_row_left, loc_row_right, _, _ = torch.chunk(pred['loc_row'], 5)
            loc_col, _, _, loc_col_up, loc_col_down = torch.chunk(pred['loc_col'], 5)

            exist_row, exist_row_left, exist_row_right, _, _ = torch.chunk(pred['exist_row'], 5)
            exist_col, _, _, exist_col_up, exist_col_down = torch.chunk(pred['exist_col'], 5)


        generate_lines_local_tta(loc_row, loc_row_left, loc_row_right, exist_row, exist_row_left, exist_row_right, names, output_path, row_anchor)
        generate_lines_col_local_tta(loc_col, loc_col_up, loc_col_down, exist_col, exist_col_up, exist_col_down, names, output_path, col_anchor)

def generate_tusimple_lines(row_out, row_ext, col_out, col_ext, row_anchor = None, col_anchor = None, mode = '2row2col'):
    tusimple_h_sample = np.linspace(160, 710, 56)
    row_num_grid, row_num_cls, row_num_lane = row_out.shape
    row_max_indices = row_out.argmax(0).cpu()
    # num_cls, num_lanes
    row_valid = row_ext.argmax(0).cpu()
    # num_cls, num_lanes
    row_out = row_out.cpu()

    col_num_grid, col_num_cls, col_num_lane = col_out.shape
    col_max_indices = col_out.argmax(0).cpu()
    # num_cls, num_lanes
    col_valid = col_ext.argmax(0).cpu()
    # num_cls, num_lanes
    col_out = col_out.cpu()

    # mode = '2row2col'

    if mode == 'normal' or mode == '2row2col':
        row_lane_list = [1, 2]
        col_lane_list = [0, 3]
    elif mode == '4row':
        row_lane_list = range(row_num_lane)
        col_lane_list = []
    elif mode == '4col':
        row_lane_list = []
        col_lane_list = range(col_num_lane)
    else:
        raise NotImplementedError

    local_width_row = 14
    local_width_col = 14
    min_lanepts_row = 3
    min_lanepts_col = 3
    
    # local_width = 2
    all_lanes = []

    for row_lane_idx in row_lane_list:
        if row_valid[ :, row_lane_idx].sum() > min_lanepts_row:
            cur_lane = []
            for row_cls_idx in range(row_num_cls):

                if row_valid[ row_cls_idx, row_lane_idx]:
                    all_ind = torch.tensor(list(
                        range(
                            max(0,row_max_indices[ row_cls_idx, row_lane_idx] - local_width_row), 
                            min(row_num_grid-1, row_max_indices[ row_cls_idx, row_lane_idx] + local_width_row) + 1)
                            )
                            )
                    coord = (row_out[all_ind,row_cls_idx,row_lane_idx].softmax(0) * all_ind.float()).sum() + 0.5
                    coord_x = coord / (row_num_grid - 1) * 1280
                    coord_y = row_anchor[row_cls_idx] * 720
                    cur_lane.append(int(coord_x))
                else:
                    cur_lane.append(-2)
                    # cur_lane.append((coord_x, coord_y))
            # cur_lane = np.array(cur_lane)
            # p = np.polyfit(cur_lane[:,1], cur_lane[:,0], deg = 2)
            # top_lim = min(cur_lane[:,1])
            # # all_lane_interps.append((p, top_lim))
            # lanes_on_tusimple = np.polyval(p, tusimple_h_sample)
            # lanes_on_tusimple = np.round(lanes_on_tusimple)
            # lanes_on_tusimple = lanes_on_tusimple.astype(int)
            # lanes_on_tusimple[lanes_on_tusimple < 0] = -2
            # lanes_on_tusimple[lanes_on_tusimple > 1280] = -2
            # lanes_on_tusimple[tusimple_h_sample < top_lim] = -2
            # all_lanes.append(lanes_on_tusimple.tolist())
            all_lanes.append(cur_lane)
        else:
            # all_lanes.append([-2]*56)
            pass

    for col_lane_idx in col_lane_list:
        if col_valid[ :, col_lane_idx].sum() > min_lanepts_col:
            cur_lane = []
            for col_cls_idx in range(col_num_cls):
                if col_valid[ col_cls_idx, col_lane_idx]:
                    all_ind = torch.tensor(list(
                        range(
                            max(0,col_max_indices[ col_cls_idx, col_lane_idx] - local_width_col), 
                            min(col_num_grid-1, col_max_indices[ col_cls_idx, col_lane_idx] + local_width_col) + 1)
                            )
                            )
                    coord = (col_out[all_ind,col_cls_idx,col_lane_idx].softmax(0) * all_ind.float()).sum() + 0.5
                    coord_y = coord / (col_num_grid - 1) * 720
                    coord_x = col_anchor[col_cls_idx] * 1280
                    cur_lane.append((coord_x, coord_y))    
            cur_lane = np.array(cur_lane)
            top_lim = min(cur_lane[:,1])
            bot_lim = max(cur_lane[:,1])
            
            p = np.polyfit(cur_lane[:,1], cur_lane[:,0], deg = 2)
            lanes_on_tusimple = np.polyval(p, tusimple_h_sample)

            # cur_lane_x = cur_lane[:,0]
            # cur_lane_y = cur_lane[:,1]
            # cur_lane_x_sorted = [x for _, x in sorted(zip(cur_lane_y, cur_lane_x))]
            # cur_lane_y_sorted = sorted(cur_lane_y)
            # p = InterpolatedUnivariateSpline(cur_lane_y_sorted, cur_lane_x_sorted, k=min(3, len(cur_lane_x_sorted) - 1))
            # lanes_on_tusimple = p(tusimple_h_sample)

            lanes_on_tusimple = np.round(lanes_on_tusimple)
            lanes_on_tusimple = lanes_on_tusimple.astype(int)
            lanes_on_tusimple[lanes_on_tusimple < 0] = -2
            lanes_on_tusimple[lanes_on_tusimple > 1280] = -2
            lanes_on_tusimple[tusimple_h_sample < top_lim] = -2
            lanes_on_tusimple[tusimple_h_sample > bot_lim] = -2
            all_lanes.append(lanes_on_tusimple.tolist())
        else:
            # all_lanes.append([-2]*56)
            pass
    # for (p, top_lim) in all_lane_interps:
    #     lanes_on_tusimple = np.polyval(p, tusimple_h_sample)
    #     lanes_on_tusimple = np.round(lanes_on_tusimple)
    #     lanes_on_tusimple = lanes_on_tusimple.astype(int)
    #     lanes_on_tusimple[lanes_on_tusimple < 0] = -2
    #     lanes_on_tusimple[lanes_on_tusimple > 1280] = -2
    #     lanes_on_tusimple[tusimple_h_sample < top_lim] = -2
    #     all_lanes.append(lanes_on_tusimple.tolist())
    return all_lanes
    
def run_test_tusimple(net,data_root,work_dir,exp_name, distributed, crop_ratio, train_width, train_height, batch_size = 8, row_anchor = None, col_anchor = None):
    output_path = os.path.join(work_dir,exp_name+'.%d.txt'% get_rank())
    fp = open(output_path,'w')
    loader = get_test_loader(batch_size,data_root,'Tusimple', distributed, crop_ratio, train_width, train_height)
    for data in dist_tqdm(loader):
        imgs,names = data
        imgs = imgs.cuda()
        with torch.no_grad():
            pred = net(imgs)
        for b_idx,name in enumerate(names):
            tmp_dict = {}
            tmp_dict['lanes'] = generate_tusimple_lines(pred['loc_row'][b_idx], pred['exist_row'][b_idx], pred['loc_col'][b_idx], pred['exist_col'][b_idx], row_anchor = row_anchor, col_anchor = col_anchor, mode = '4row')
            tmp_dict['h_samples'] = [160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260,
             270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 
             430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 
             590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710]
            tmp_dict['raw_file'] = name
            tmp_dict['run_time'] = 10
            json_str = json.dumps(tmp_dict)

            fp.write(json_str+'\n')
    fp.close()

def combine_tusimple_test(work_dir,exp_name):
    size = get_world_size()
    all_res = []
    for i in range(size):
        output_path = os.path.join(work_dir,exp_name+'.%d.txt'% i)
        with open(output_path, 'r') as fp:
            res = fp.readlines()
        all_res.extend(res)
    names = set()
    all_res_no_dup = []
    for i, res in enumerate(all_res):
        pos = res.find('clips')
        name = res[pos:].split('\"')[0]
        if name not in names:
            names.add(name)
            all_res_no_dup.append(res)

    output_path = os.path.join(work_dir,exp_name+'.txt')
    with open(output_path, 'w') as fp:
        fp.writelines(all_res_no_dup)
    

def eval_lane(net, cfg, ep = None, logger = None):
    # Allow per-run override of noisy evaluation debug via cfg.eval_debug.
    # If present, this will set the module-level EVAL_DEBUG flag used by eprint().
    global EVAL_DEBUG
    if hasattr(cfg, 'eval_debug'):
        try:
            EVAL_DEBUG = bool(cfg.eval_debug)
        except Exception:
            pass
    net.eval()
    if cfg.dataset == 'CurveLanes':
        if not cfg.tta:
            run_test(cfg.dataset, net, cfg.data_root, 'curvelanes_eval_tmp', cfg.test_work_dir, cfg.distributed, cfg.crop_ratio, cfg.train_width, cfg.train_height, row_anchor = cfg.row_anchor, col_anchor = cfg.col_anchor, logger=logger)
        else:
            run_test_tta(cfg.dataset, net, cfg.data_root, 'curvelanes_eval_tmp', cfg.test_work_dir, cfg.distributed,  cfg.crop_ratio, cfg.train_width, cfg.train_height, row_anchor = cfg.row_anchor, col_anchor = cfg.col_anchor)
        synchronize()   # wait for all results
        if is_main_process():
            res = call_curvelane_eval(cfg.data_root, 'curvelanes_eval_tmp', cfg.test_work_dir)
            if res is None:
                dist_print('call_curvelane_eval returned None (evaluator failed). Setting F=0 and continuing.')
                if logger is not None and is_main_process():
                    logger.add_scalar('CuEval/total', 0.0, global_step = ep)
                synchronize()
                if is_main_process():
                    return 0.0
                else:
                    return None
            TP,FP,FN = 0,0,0
            for k, v in res.items():
                # defensively read Fmeasure: evaluator output may omit or return 'nan'
                fmeasure_raw = v.get('Fmeasure') if isinstance(v, dict) else None
                try:
                    if fmeasure_raw is None or 'nan' in str(fmeasure_raw):
                        val = 0.0
                    else:
                        val = float(fmeasure_raw)
                except Exception:
                    val = 0.0
                val_tp,val_fp,val_fn = int(v.get('tp', 0)),int(v.get('fp', 0)),int(v.get('fn', 0))
                TP += val_tp
                FP += val_fp
                FN += val_fn
                eprint(k, val)
                if logger is not None:
                    if k == 'res_cross':
                        logger.add_scalar('CuEval_cls/'+k,val_fp,global_step = ep)
                        continue
                    logger.add_scalar('CuEval_cls/'+k,val,global_step = ep)
            if TP + FP == 0:
                P = 0
                print("nearly no results!")
            else:
                P = TP * 1.0/(TP + FP)
            if TP + FN == 0:
                R = 0
                print("nearly no results!")
            else:
                R = TP * 1.0/(TP + FN)
            if (P+R) == 0:
                F = 0
            else:
                F = 2*P*R/(P + R)
            eprint(F)
            if logger is not None:
                logger.add_scalar('CuEval/total',F,global_step = ep)
                logger.add_scalar('CuEval/P',P,global_step = ep)
                logger.add_scalar('CuEval/R',R,global_step = ep)
                # Append per-epoch history (one line per epoch) so we keep an audit trail
                try:
                    _append_eval_history(cfg, ep, {'P': P, 'R': R, 'F': F, 'per_class': res}, dataset_name='CurveLanes', logger=logger)
                except Exception:
                    pass
              
        synchronize()
        if is_main_process():
            return F
        else:
            return None
    elif cfg.dataset == 'CULane':
        if not cfg.tta:
            run_test(cfg.dataset, net, cfg.data_root, 'culane_eval_tmp', cfg.test_work_dir, cfg.distributed, cfg.crop_ratio, cfg.train_width, cfg.train_height, row_anchor = cfg.row_anchor, col_anchor = cfg.col_anchor, logger=logger)
        else:
            run_test_tta(cfg.dataset, net, cfg.data_root, 'culane_eval_tmp', cfg.test_work_dir, cfg.distributed, cfg.crop_ratio, cfg.train_width, cfg.train_height, row_anchor = cfg.row_anchor, col_anchor = cfg.col_anchor)
        synchronize()    # wait for all results
        if is_main_process():
            res_both = call_culane_eval(cfg.data_root, 'culane_eval_tmp', cfg.test_work_dir)
            # res_both is a dict {'0.3': res_03, '0.5': res_05}
            # Log both results (0.3 and 0.5). Keep compatibility: return F for 0.5.
            for iou_key in ['0.3', '0.5']:
                res = res_both.get(iou_key) if isinstance(res_both, dict) else None
                if res is None:
                    dist_print(f'call_culane_eval returned None for IOU={iou_key} (evaluator failed). Skipping this IOU result.')
                    continue
                TP,FP,FN = 0,0,0
                for k, v in res.items():
                    # defensively read Fmeasure: evaluator output may omit or return 'nan'
                    fmeasure_raw = v.get('Fmeasure') if isinstance(v, dict) else None
                    try:
                        if fmeasure_raw is None or 'nan' in str(fmeasure_raw):
                            val = 0.0
                        else:
                            val = float(fmeasure_raw)
                    except Exception:
                        val = 0.0
                    val_tp,val_fp,val_fn = int(v.get('tp', 0)),int(v.get('fp', 0)),int(v.get('fn', 0))
                    TP += val_tp
                    FP += val_fp
                    FN += val_fn
                    eprint(f"IOU={iou_key} ", k, val)
                    if logger is not None:
                        if k == 'res_cross':
                            logger.add_scalar(f'CuEval_cls/{iou_key}/'+k,val_fp,global_step = ep)
                            continue
                        logger.add_scalar(f'CuEval_cls/{iou_key}/'+k,val,global_step = ep)
                if TP + FP == 0:
                    P = 0
                    print("nearly no results!")
                else:
                    P = TP * 1.0/(TP + FP)
                if TP + FN == 0:
                    R = 0
                    print("nearly no results!")
                else:
                    R = TP * 1.0/(TP + FN)
                if (P+R) == 0:
                    F = 0
                else:
                    F = 2*P*R/(P + R)
                dist_print(f"IOU={iou_key} total F:", F)
                if logger is not None:
                    logger.add_scalar(f'CuEval/{iou_key}/total',F,global_step = ep)
                    logger.add_scalar(f'CuEval/{iou_key}/P',P,global_step = ep)
                    logger.add_scalar(f'CuEval/{iou_key}/R',R,global_step = ep)
            # return the 0.5-F score to keep behavior consistent (F currently holds 0.5 result)
            # persist epoch history (both IOU results plus summary F for 0.5)
            try:
                _append_eval_history(cfg, ep, {'iou_results': res_both, 'F_0.5': F}, dataset_name='CULane', logger=logger)
            except Exception:
                pass
            return F
              
        synchronize()
        if is_main_process():
            return F
        else:
            return None
    elif cfg.dataset == 'Tusimple':
        exp_name = 'tusimple_eval_tmp'
        run_test_tusimple(net, cfg.data_root, cfg.test_work_dir, exp_name, cfg.distributed, cfg.crop_ratio, cfg.train_width, cfg.train_height, row_anchor = cfg.row_anchor, col_anchor = cfg.col_anchor)
        synchronize()  # wait for all results
        if is_main_process():
            combine_tusimple_test(cfg.test_work_dir,exp_name)
            res = LaneEval.bench_one_submit(os.path.join(cfg.test_work_dir,exp_name + '.txt'),os.path.join(cfg.data_root,'test_label.json'))
            res = json.loads(res)
            for r in res:
                dist_print(r['name'], r['value'])
                if logger is not None:
                    logger.add_scalar('TuEval/'+r['name'],r['value'],global_step = ep)
        synchronize()
        if is_main_process():
            for r in res:
                if r['name'] == 'F1':
                    # append epoch history for Tusimple
                    try:
                        metrics = {it['name']: it['value'] for it in res}
                        _append_eval_history(cfg, ep, {'metrics': metrics}, dataset_name='Tusimple', logger=logger)
                    except Exception:
                        pass
                    return r['value']
        else:
            return None


def read_helper(path):
    lines = open(path, 'r').readlines()[1:]
    lines = ' '.join(lines)
    values = lines.split(' ')[1::2]
    keys = lines.split(' ')[0::2]
    keys = [key[:-1] for key in keys]
    res = {k : v for k,v in zip(keys,values)}
    return res


def _sanitize_for_json(obj):
    """Recursively convert numpy types and sanitize NaN/inf to None so json.dump won't fail.
    Returns a JSON-serializable structure with Python primitives.
    """
    import math
    # primitives
    if obj is None:
        return None
    if isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (int,)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    # numpy scalar types
    try:
        import numpy as _np
        if isinstance(obj, _np.floating):
            v = float(obj)
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        if isinstance(obj, _np.integer):
            return int(obj)
    except Exception:
        pass
    # containers
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    # fallback: try to convert to float/int/str, else return str(obj)
    try:
        if hasattr(obj, 'tolist'):
            return _sanitize_for_json(obj.tolist())
    except Exception:
        pass
    try:
        return float(obj)
    except Exception:
        try:
            return int(obj)
        except Exception:
            return str(obj)


def _append_eval_history(cfg, ep, results_dict, dataset_name=None, logger=None):
    """Append a JSON line with evaluation results to a persistent history file.
    - cfg: config object; uses cfg.test_work_dir as destination directory
    - ep: epoch number (may be None)
    - results_dict: dict with results (will be sanitized for JSON)
    - dataset_name: optional string for easy filtering
    - logger: optional tensorboard logger (will receive a summary scalar 'Eval/F' when present)
    """
    try:
        out_dir = getattr(cfg, 'test_work_dir', None) or '.'
        if not os.path.exists(out_dir) and is_main_process():
            os.makedirs(out_dir, exist_ok=True)
        hist_path = os.path.join(out_dir, 'eval_history.jsonl')
        entry = {
            'timestamp': time.time(),
            'epoch': ep if ep is not None else -1,
            'dataset': dataset_name,
            'results': results_dict
        }
        with open(hist_path, 'a') as hf:
            json.dump(_sanitize_for_json(entry), hf)
            hf.write('\n')
        if is_main_process():
            dist_print(f"Appended eval results to: {hist_path}")
    except Exception as e:
        dist_print(f"Failed to append eval history: {e}")

    # Additionally, add a lightweight TensorBoard summary if provided
    try:
        if logger is not None:
            # prefer explicit keys
            if isinstance(results_dict, dict):
                if 'F' in results_dict:
                    logger.add_scalar('Eval/F', float(results_dict['F']), global_step=ep)
                elif 'F_0.5' in results_dict:
                    logger.add_scalar('Eval/F', float(results_dict['F_0.5']), global_step=ep)
                elif 'metrics' in results_dict and isinstance(results_dict['metrics'], dict) and 'F1' in results_dict['metrics']:
                    logger.add_scalar('Eval/F', float(results_dict['metrics']['F1']), global_step=ep)
    except Exception:
        pass

def call_culane_eval(data_dir, exp_name,output_path):
    # helper that runs the CULane evaluator with a specified iou and returns results
    def _call_with_iou(iou_val):
        if data_dir[-1] != '/':
            dd = data_dir + '/'
        else:
            dd = data_dir
        detect_dir=os.path.join(output_path,exp_name) + '/'

        w_lane=30
        iou=iou_val
        im_w=1640
        im_h=590
        frame=1
        split_dir = os.path.join(dd, 'list', 'test_split')
        eval_cmd = './evaluation/culane/evaluate'
        if platform.system() == 'Windows':
            eval_cmd = eval_cmd.replace('/', os.sep)

        if os.path.exists(split_dir):
            list0 = os.path.join(dd,'list/test_split/test0_normal.txt')
            list1 = os.path.join(dd,'list/test_split/test1_crowd.txt')
            list2 = os.path.join(dd,'list/test_split/test2_hlight.txt')
            list3 = os.path.join(dd,'list/test_split/test3_shadow.txt')
            list4 = os.path.join(dd,'list/test_split/test4_noline.txt')
            list5 = os.path.join(dd,'list/test_split/test5_arrow.txt')
            list6 = os.path.join(dd,'list/test_split/test6_curve.txt')
            list7 = os.path.join(dd,'list/test_split/test7_cross.txt')
            list8 = os.path.join(dd,'list/test_split/test8_night.txt')
            if not os.path.exists(os.path.join(output_path,'txt')):
                os.mkdir(os.path.join(output_path,'txt'))
            out0 = os.path.join(output_path,'txt','out0_normal.txt')
            out1=os.path.join(output_path,'txt','out1_crowd.txt')
            out2=os.path.join(output_path,'txt','out2_hlight.txt')
            out3=os.path.join(output_path,'txt','out3_shadow.txt')
            out4=os.path.join(output_path,'txt','out4_noline.txt')
            out5=os.path.join(output_path,'txt','out5_arrow.txt')
            out6=os.path.join(output_path,'txt','out6_curve.txt')
            out7=os.path.join(output_path,'txt','out7_cross.txt')
            out8=os.path.join(output_path,'txt','out8_night.txt')

            os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,dd,detect_dir,dd,list0,w_lane,iou,im_w,im_h,frame,out0))
            os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,dd,detect_dir,dd,list1,w_lane,iou,im_w,im_h,frame,out1))
            os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,dd,detect_dir,dd,list2,w_lane,iou,im_w,im_h,frame,out2))
            os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,dd,detect_dir,dd,list3,w_lane,iou,im_w,im_h,frame,out3))
            os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,dd,detect_dir,dd,list4,w_lane,iou,im_w,im_h,frame,out4))
            os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,dd,detect_dir,dd,list5,w_lane,iou,im_w,im_h,frame,out5))
            os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,dd,detect_dir,dd,list6,w_lane,iou,im_w,im_h,frame,out6))
            os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,dd,detect_dir,dd,list7,w_lane,iou,im_w,im_h,frame,out7))
            os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,dd,detect_dir,dd,list8,w_lane,iou,im_w,im_h,frame,out8))
            res_all = {}
            res_all['res_normal'] = read_helper(out0)
            res_all['res_crowd']= read_helper(out1)
            res_all['res_night']= read_helper(out8)
            res_all['res_noline'] = read_helper(out4)
            res_all['res_shadow'] = read_helper(out3)
            res_all['res_arrow']= read_helper(out5)
            res_all['res_hlight'] = read_helper(out2)
            res_all['res_curve']= read_helper(out6)
            res_all['res_cross']= read_helper(out7)
            return res_all
        else:
            # Fallback: if test_split doesn't exist, use the single test list (list/test.txt)
            # Run the evaluator once for the requested `iou_val` and return a per-split
            # mapping where every split key maps to the single-result read from the
            # evaluator output. This avoids executing the evaluator multiple times
            # (the previous implementation could run the evaluator repeatedly).
            list_file = os.path.join(dd, 'list', 'test.txt')
            if not os.path.exists(os.path.join(output_path, 'txt')):
                os.mkdir(os.path.join(output_path, 'txt'))
            out_single = os.path.join(output_path, 'txt', f'out_single_{int(iou_val*10)}.txt')
            cmd = '%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s' % (
                eval_cmd, dd, detect_dir, dd, list_file, w_lane, iou_val, im_w, im_h, frame, out_single)
            os.system(cmd)
            try:
                res = read_helper(out_single)
            except Exception:
                # Evaluator failed to produce output we can parse
                return None
            # Populate a per-split dict where each expected split key maps to the same result
            res_all = {k: res for k in ['res_normal', 'res_crowd', 'res_night', 'res_noline', 'res_shadow', 'res_arrow', 'res_hlight', 'res_curve', 'res_cross']}
            return res_all
    # call the inner helper for the IOU thresholds we want and return both results
    try:
        res_03 = _call_with_iou(0.3)
    except Exception as e:
        dist_print(f"call_culane_eval: evaluator failed for IOU=0.3: {e}")
        res_03 = None
    try:
        res_05 = _call_with_iou(0.5)
    except Exception as e:
        dist_print(f"call_culane_eval: evaluator failed for IOU=0.5: {e}")
        res_05 = None

    results = {}
    results['0.3'] = res_03
    results['0.5'] = res_05
    # Persist a combined summary JSON for easier inspection (best-effort)
    try:
        summary_path = os.path.join(output_path, f"{exp_name}_eval_results.json")
        with open(summary_path, 'w') as sf:
            json.dump(_sanitize_for_json(results), sf, indent=2)
        if is_main_process():
            dist_print(f"Wrote evaluation summary to: {summary_path}")
    except Exception as e:
        dist_print(f"Failed to write evaluation summary: {e}")
    return results


def _try_load_mask(data_root, img_name, H=None, W=None):
    """Attempt to locate and load a mask corresponding to img_name under data_root.
    Returns a 2D numpy array or None.
    """
    dirname = os.path.dirname(img_name)
    basename = os.path.basename(img_name)
    name_no_ext = os.path.splitext(basename)[0]
    candidates = [
        os.path.join(data_root, dirname, name_no_ext + '.png'),
        os.path.join(data_root, dirname, name_no_ext + '.jpg'),
        os.path.join(data_root, dirname, name_no_ext + '_mask.png'),
        os.path.join(data_root, dirname, name_no_ext + '_label.png'),
        os.path.join(data_root, 'mask', name_no_ext + '.png'),
        os.path.join(data_root, 'label', name_no_ext + '.png'),
        os.path.join(data_root, basename.replace('.jpg', '.png')),
        os.path.join(data_root, basename.replace('.jpg', '_mask.png')),
    ]
    for p in candidates:
        try:
            if not os.path.exists(p):
                continue
            if p.endswith('.npy'):
                arr = np.load(p)
            else:
                pil = Image.open(p).convert('L')
                if H is not None and W is not None and pil.size != (W, H):
                    pil = pil.resize((W, H), Image.NEAREST)
                arr = np.asarray(pil)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            return arr
        except Exception:
            continue
    return None