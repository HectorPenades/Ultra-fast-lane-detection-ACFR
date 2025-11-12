import torch, os, datetime


from utils.dist_utils import dist_print, dist_tqdm, synchronize
from utils.factory import get_metric_dict, get_loss_dict, get_optimizer, get_scheduler
from utils.metrics import update_metrics, reset_metrics

from utils.common import calc_loss, get_model, get_train_loader, inference, merge_config, save_model, cp_projects
from utils.common import get_work_dir, get_logger

import time
from evaluation.eval_wrapper import eval_lane
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import sys


def _robust_load_weights(path, net, desc=None):
    """Try several common checkpoint formats and print diagnostic info.
    Returns True if any load succeeded (even with strict=False), else False.
    """
    from utils.dist_utils import dist_print
    try:
        dist_print(f"Attempting to load checkpoint for {desc or 'weights'}: {path}")
        ckpt = torch.load(path, map_location='cpu')
    except Exception as e:
        dist_print(f"Failed to torch.load('{path}'): {e}")
        return False

    # If ckpt is a dict, examine keys
    if isinstance(ckpt, dict):
        dist_print('Checkpoint top-level keys:', list(ckpt.keys())[:20])

        # Common layout: {'model': {...}, 'optimizer': {...}, ...}
        candidate_keys = ['model', 'state_dict', 'state_dicts', 'state']
        for k in candidate_keys:
            if k in ckpt:
                model_dict = ckpt[k]
                try:
                    net.load_state_dict(model_dict)
                    dist_print(f"Loaded weights from key '{k}' with strict=True")
                    return True
                except Exception as e:
                    # try stripping 'module.' prefixes
                    try:
                        stripped = { (nk.replace('module.', '') if nk.startswith('module.') else nk): v for nk, v in model_dict.items() }
                        net.load_state_dict(stripped, strict=False)
                        dist_print(f"Loaded weights from key '{k}' with strict=False after stripping 'module.' prefixes")
                        return True
                    except Exception as e2:
                        dist_print(f"Could not load from key '{k}': {e}; fallback attempt also failed: {e2}")
                        # continue to next candidate

        # If dict but none of the above keys worked, maybe the dict itself is a state_dict
        try:
            net.load_state_dict(ckpt)
            dist_print('Loaded checkpoint dict directly as state_dict (strict=True)')
            return True
        except Exception as e:
            try:
                stripped = { (nk.replace('module.', '') if nk.startswith('module.') else nk): v for nk, v in ckpt.items() }
                net.load_state_dict(stripped, strict=False)
                dist_print('Loaded checkpoint dict directly as state_dict with strict=False after stripping module.')
                return True
            except Exception as e2:
                dist_print('Failed to load checkpoint dict as state_dict:', e, e2)
                return False

    else:
        dist_print('Checkpoint file did not contain a dict; cannot load model weights directly.')
        return False


def train(net, data_loader, loss_dict, optimizer, scheduler,logger, epoch, metric_dict, dataset, vis_interval=None):
    net.train()
    # use the passed data_loader variable (could be TrainCollect or DataLoader)
    progress_bar = dist_tqdm(data_loader)
    for b_idx, data_label in enumerate(progress_bar):
        global_step = epoch * len(data_loader) + b_idx

        results = inference(net, data_label, dataset)

        loss = calc_loss(loss_dict, results, logger, global_step, epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(global_step)


        if global_step % 20 == 0:
            reset_metrics(metric_dict)
            update_metrics(metric_dict, results)
            for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
                logger.add_scalar('metric/' + me_name, me_op.get(), global_step=global_step)
            logger.add_scalar('meta/lr', optimizer.param_groups[0]['lr'], global_step=global_step)

            if hasattr(progress_bar,'set_postfix'):
                kwargs = {me_name: '%.3f' % me_op.get() for me_name, me_op in zip(metric_dict['name'], metric_dict['op'])}
                new_kwargs = {}
                for k,v in kwargs.items():
                    if 'lane' in k:
                        continue
                    new_kwargs[k] = v
                progress_bar.set_postfix(loss = '%.3f' % float(loss), 
                                        **new_kwargs)
        
        # Visualization: save input+label overlay to TensorBoard every N steps
        try:
            # determine visualization interval (default 200 steps)
            try:
                vi = int(vis_interval) if vis_interval is not None else 200
            except Exception:
                vi = 200
            if logger is not None and (global_step % vi == 0):
                fig = plt.figure(figsize=(6,4))
                ax = fig.add_subplot(1,1,1)

                img_np = None
                seg_np = None
                pts = None

                # DALI path: data_label is dict; pytorch path: tuple (imgs, labels)
                # Prefer using the labels that the model actually received (from `results`) because
                # these are post-transform / post-augmentation. Fall back to data_label when needed.
                if isinstance(data_label, dict):
                    if 'images' in data_label:
                        images = data_label['images']
                        img = images[0]
                        if isinstance(img, np.ndarray):
                            img_np = img
                        else:
                            img_np = img.cpu().numpy()
                            if img_np.ndim == 3:
                                img_np = np.transpose(img_np, (1,2,0))
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        img_np = (img_np * std[None,None,:]) + mean[None,None,:]
                        img_np = np.clip(img_np, 0, 1)
                        img_np = (img_np * 255).astype(np.uint8)

                    if 'seg_images' in data_label:
                        seg = data_label['seg_images'][0]
                        if not isinstance(seg, np.ndarray):
                            seg_np = seg.cpu().numpy()
                            if seg_np.ndim == 3:
                                seg_np = seg_np[0]
                        else:
                            seg_np = seg
                        seg_np = np.round(seg_np).astype(np.uint8)

                    # Prefer the transformed labels available in `results` (they were constructed from data_label
                    # after any pipeline transforms and interpolation). This avoids accidentally plotting raw/untransformed
                    # points if they are present in the source.
                    if isinstance(results, dict) and 'labels_row_float' in results:
                        lr_raw = results['labels_row_float']
                        try:
                            lr = lr_raw[0].cpu().numpy() if hasattr(lr_raw[0], 'cpu') else np.array(lr_raw[0])
                        except Exception:
                            lr = np.array(lr_raw[0])
                        pts = ('row_float', lr)
                        label_source = 'results'
                    elif 'labels_row_float' in data_label:
                        lr = data_label['labels_row_float'][0]
                        try:
                            lr = lr.cpu().numpy() if hasattr(lr, 'cpu') else np.array(lr)
                        except Exception:
                            lr = np.array(lr)
                        pts = ('row_float', lr)
                        label_source = 'data_label'

                else:
                    imgs, cls = data_label
                    img = imgs[0]
                    img_np = img.cpu().numpy()
                    if img_np.ndim == 3:
                        img_np = np.transpose(img_np, (1,2,0))
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img_np = (img_np * std[None,None,:]) + mean[None,None,:]
                    img_np = np.clip(img_np, 0, 1)
                    img_np = (img_np * 255).astype(np.uint8)

                    # Prefer labels from `results` (post-transform) if available; else use raw cls from dataloader
                    if isinstance(results, dict) and 'cls_label' in results:
                        cls_raw = results['cls_label']
                        try:
                            cls_np = cls_raw[0].cpu().numpy() if hasattr(cls_raw[0], 'cpu') else np.array(cls_raw[0])
                        except Exception:
                            cls_np = np.array(cls_raw[0])
                        label_source = 'results'
                    else:
                        try:
                            cls_np = cls[0].numpy()
                        except Exception:
                            cls_np = np.array(cls[0])
                    pts = ('cls_grid', cls_np)
                    label_source = label_source if 'label_source' in locals() else 'data_label'

                if img_np is not None:
                    ax.imshow(img_np)
                    ax.axis('off')

                    if seg_np is not None:
                        cmap = np.zeros((seg_np.shape[0], seg_np.shape[1], 4), dtype=np.uint8)
                        cmap[seg_np == 1] = [255,0,0,128]
                        cmap[seg_np == 2] = [0,0,255,128]
                        ax.imshow(cmap)

                    if pts is not None:
                        typ, data = pts
                        H, W = img_np.shape[:2]
                        if typ == 'row_float':
                            # arr: rows x lanes (normalized x coords 0..1 or -1)
                            arr = data
                            if arr.ndim == 3:
                                arr = arr[0]
                            if arr.shape[0] < arr.shape[1]:
                                arr = arr.T
                            num_rows = arr.shape[0]
                            y_coords = np.linspace(0, H-1, num_rows)
                            # Choose deterministic colors and limit to at most 2 lanes to avoid visual clutter
                            lane_colors = ['cyan', 'magenta']
                            lane_styles = ['-', '--']  # first lane solid, second dashed
                            max_plot_lanes = 2
                            counts = []
                            for lane_idx in range(min(arr.shape[1], max_plot_lanes)):
                                xs = arr[:,lane_idx]
                                valid = (xs >= 0) & (xs <= 1)
                                counts.append(int(np.sum(valid)))
                            try:
                                dist_print(f'Vis: label_source={label_source}, row_float valid counts per lane: {counts}')
                            except Exception:
                                print(f'Vis: label_source={label_source}, row_float valid counts per lane: {counts}')
                            plotted_handles = []
                            for lane_idx in range(min(arr.shape[1], max_plot_lanes)):
                                xs = arr[:,lane_idx]
                                valid = (xs >= 0) & (xs <= 1)
                                if not np.any(valid):
                                    continue
                                pos = np.where(valid)[0]
                                xs_vals = (xs[pos] * W)
                                ys_vals = y_coords[pos]
                                # split into contiguous runs so we don't connect across missing points
                                runs = np.split(np.arange(len(pos)), np.where(np.diff(pos) != 1)[0] + 1)
                                for run in runs:
                                    if run.size == 0:
                                        continue
                                    h, = ax.plot(xs_vals[run], ys_vals[run], lane_styles[lane_idx % len(lane_styles)], color=lane_colors[lane_idx % len(lane_colors)], linewidth=2)
                                plotted_handles.append(h)
                            # add legend indicating lane indices and source
                            if len(plotted_handles) > 0:
                                legend_lines = []
                                legend_labels = []
                                for i, h in enumerate(plotted_handles):
                                    legend_lines.append(Line2D([0], [0], color=lane_colors[i % len(lane_colors)], linestyle=lane_styles[i % len(lane_styles)], linewidth=2))
                                    legend_labels.append(f'lane {i} ({"results" if label_source=="results" else "data"})')
                                ax.legend(legend_lines, legend_labels, loc='lower right', fontsize='small')
                        elif typ == 'cls_grid':
                            cls_np = data
                            n_rows, n_lanes = cls_np.shape
                            y_coords = np.linspace(0, H-1, n_rows)
                            valid_vals = cls_np[cls_np < 10000]
                            if valid_vals.size == 0:
                                num_cols = 200
                            else:
                                num_cols = int(np.max(valid_vals)) + 1
                            # Use same deterministic lane colors and limit to two lanes
                            lane_colors = ['cyan', 'magenta']
                            lane_styles = ['-', '--']
                            max_plot_lanes = 2
                            counts = []
                            for lane in range(min(n_lanes, max_plot_lanes)):
                                cnt = 0
                                for ri in range(n_rows):
                                    c = cls_np[ri, lane]
                                    if c < 0 or c >= num_cols:
                                        continue
                                    cnt += 1
                                counts.append(cnt)
                            try:
                                dist_print(f'Vis: label_source={label_source}, cls_grid valid counts per lane: {counts}')
                            except Exception:
                                print(f'Vis: label_source={label_source}, cls_grid valid counts per lane: {counts}')
                            plotted_handles = []
                            for lane in range(min(n_lanes, max_plot_lanes)):
                                pos_idxs = []
                                xs_list = []
                                ys_list = []
                                for ri in range(n_rows):
                                    c = cls_np[ri, lane]
                                    if c < 0 or c >= num_cols:
                                        continue
                                    x = c * (W - 1) / max(1, (num_cols - 1))
                                    y = y_coords[ri]
                                    pos_idxs.append(ri)
                                    xs_list.append(x)
                                    ys_list.append(y)
                                if len(pos_idxs) == 0:
                                    continue
                                pos = np.array(pos_idxs)
                                xs_arr = np.array(xs_list)
                                ys_arr = np.array(ys_list)
                                runs = np.split(np.arange(len(pos)), np.where(np.diff(pos) != 1)[0] + 1)
                                for run in runs:
                                    if run.size == 0:
                                        continue
                                    h, = ax.plot(xs_arr[run], ys_arr[run], lane_styles[lane % len(lane_styles)], color=lane_colors[lane % len(lane_colors)], linewidth=2)
                                plotted_handles.append(h)
                            if len(plotted_handles) > 0:
                                legend_lines = []
                                legend_labels = []
                                for i, h in enumerate(plotted_handles):
                                    legend_lines.append(Line2D([0], [0], color=lane_colors[i % len(lane_colors)], linestyle=lane_styles[i % len(lane_styles)], linewidth=2))
                                    legend_labels.append(f'lane {i} ({"results" if label_source=="results" else "data"})')
                                ax.legend(legend_lines, legend_labels, loc='lower right', fontsize='small')

                    logger.add_figure('train/input_with_label', fig, global_step)

                    # Also save PNG to disk for quick inspection
                    try:
                        log_dir = getattr(logger, 'log_dir', None)
                        if log_dir is None:
                            # SummaryWriter may store as .log_dir
                            log_dir = getattr(logger, 'file_writer', None)
                        if log_dir is None:
                            base_dir = os.getcwd()
                        else:
                            # if logger.log_dir is a property string
                            if isinstance(log_dir, str):
                                base_dir = log_dir
                            else:
                                base_dir = os.getcwd()

                        vis_dir = os.path.join(base_dir, 'vis')
                        os.makedirs(vis_dir, exist_ok=True)
                        png_path = os.path.join(vis_dir, 'input_%06d.png' % global_step)
                        fig.savefig(png_path, bbox_inches='tight')
                        # Optionally open immediately for fast debugging if env var set
                        if os.environ.get('UFLD_DEBUG_OPEN', '0') == '1':
                            try:
                                import subprocess
                                subprocess.Popen(['xdg-open', png_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            except Exception:
                                pass
                    except Exception:
                        pass

                    plt.close(fig)
        except Exception as e:
            try:
                dist_print('Visualization error:', e)
            except Exception:
                pass

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    if args.local_rank == 0:
        work_dir = get_work_dir(cfg)

    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

        if args.local_rank == 0:
            with open('.work_dir_tmp_file.txt', 'w') as f:
                f.write(work_dir)
        else:
            while not os.path.exists('.work_dir_tmp_file.txt'):
                time.sleep(0.1)
            with open('.work_dir_tmp_file.txt', 'r') as f:
                work_dir = f.read().strip()

    synchronize()
    cfg.test_work_dir = work_dir
    cfg.distributed = distributed

    if args.local_rank == 0:
        os.system('rm .work_dir_tmp_file.txt')
    
    dist_print(datetime.datetime.now().strftime('[%Y/%m/%d %H:%M:%S]') + ' start training...')
    dist_print(cfg)
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide', '34fca']

    train_loader = get_train_loader(cfg)
    dist_print('Use augmentations for training:', getattr(cfg, 'use_augmentations', True))
    net = get_model(cfg)

    # Optionally freeze parameters matching a prefix before creating optimizer
    freeze_active = False
    bn_frozen = False
    if getattr(cfg, 'freeze_backbone', False) or getattr(args, 'freeze_backbone', False):
        prefix = getattr(cfg, 'freeze_prefix', None) or getattr(args, 'freeze_prefix', 'model.')
        dist_print(f'Freezing parameters with prefix: {prefix}')
        base_net = net.module if hasattr(net, 'module') else net
        for name, p in base_net.named_parameters():
            if name.startswith(prefix):
                p.requires_grad = False
        freeze_active = True
        # Diagnostic prints: which names were frozen and parameter counts
        try:
            frozen_names = [name for name, p in base_net.named_parameters() if not p.requires_grad]
            total_params = sum(p.numel() for _, p in base_net.named_parameters())
            trainable_params = sum(p.numel() for _, p in base_net.named_parameters() if p.requires_grad)
            dist_print(f'Frozen parameter count: {len(frozen_names)} (sample up to 20):', frozen_names[:20])
            dist_print(f'Total params: {total_params}, Trainable params after freeze: {trainable_params}')
            # Optionally set BatchNorm layers to eval to avoid updating running stats
            freeze_bn = getattr(cfg, 'freeze_bn', False) or getattr(args, 'freeze_bn', False)
            if freeze_bn:
                bn_frozen = True
                num_bn = 0
                for m in base_net.modules():
                    # detect BatchNorm base class
                    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                        m.eval()
                        num_bn += 1
                dist_print(f'BatchNorm modules set to eval() while frozen: {num_bn}')
        except Exception:
            pass

    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids = [args.local_rank])
    optimizer = get_optimizer(net, cfg)

    try:
        base_net = net.module if hasattr(net, 'module') else net
        total_params = sum(p.numel() for _, p in base_net.named_parameters())
        trainable_params = sum(p.numel() for _, p in base_net.named_parameters() if p.requires_grad)
        dist_print(f'Optimizer created. Total params: {total_params}, Trainable params: {trainable_params}')
    except Exception:
        pass

    if cfg.finetune is not None:
        dist_print('finetune from ', cfg.finetune)
        ok = _robust_load_weights(cfg.finetune, net, desc='finetune')
        if not ok:
            dist_print('Warning: finetune weights failed to load or were partial:', cfg.finetune)
    if cfg.resume is not None:
        dist_print('==> Resume model from ' + cfg.resume)
        try:
            resume_dict = torch.load(cfg.resume, map_location='cpu')
        except Exception as e:
            dist_print('Failed to load resume checkpoint:', e)
            resume_dict = None
        if isinstance(resume_dict, dict):
            # try to load model portion robustly
            loaded = False
            if 'model' in resume_dict:
                try:
                    net.load_state_dict(resume_dict['model'])
                    loaded = True
                    dist_print('Loaded resume model from key "model"')
                except Exception:
                    dist_print('Could not strictly load resume["model"]; trying _robust_load_weights fallback')
            if not loaded:
                ok = _robust_load_weights(cfg.resume, net, desc='resume')
                if not ok:
                    dist_print('Warning: resume model failed to load via robust loader')

            if 'optimizer' in resume_dict.keys():
                try:
                    optimizer.load_state_dict(resume_dict['optimizer'])
                    dist_print('Loaded optimizer state from resume checkpoint')
                except Exception as e:
                    dist_print('Failed to load optimizer state from resume checkpoint:', e)

            # determine resume epoch if present in checkpoint
            if 'epoch' in resume_dict:
                try:
                    resume_epoch = int(resume_dict['epoch']) + 1
                except Exception:
                    resume_epoch = 0
            else:
                # fallback: try to parse filename, else start from 0
                try:
                    resume_epoch = int(os.path.split(cfg.resume)[1][2:5]) + 1
                except Exception:
                    dist_print('Could not parse epoch from resume filename; starting from 0')
                    resume_epoch = 0
        else:
            resume_epoch = 0
    else:
        resume_epoch = 0

    # If evaluation-only mode requested, optionally load provided test weights
    if args.eval_only:
        # Priority for evaluation weights: command-line --test_model > cfg.test_model
        test_path = args.test_model if hasattr(args, 'test_model') and args.test_model is not None else getattr(cfg, 'test_model', None)
        if test_path is not None and str(test_path).strip() != '':
            dist_print('Loading weights for evaluation from', test_path)
            ok = _robust_load_weights(test_path, net, desc='test_model')
            if not ok:
                dist_print('Warning: test_model weights failed to load or were partial:', test_path)
        else:
            # No explicit test_model provided. If finetune was specified earlier it was already loaded.
            if getattr(cfg, 'finetune', None) is not None:
                dist_print('No --test_model provided; using weights loaded from --finetune (if any).')
            else:
                dist_print('No --test_model or cfg.finetune provided; evaluating with current network weights (may be random/uninitialized)')

    scheduler = get_scheduler(optimizer, cfg, len(train_loader))
    dist_print(len(train_loader))
    metric_dict = get_metric_dict(cfg)
    loss_dict = get_loss_dict(cfg)
    logger = get_logger(work_dir, cfg)
    # If eval-only was requested, run evaluation and exit
    if args.eval_only:
        dist_print('Running evaluation-only mode...')
        res = eval_lane(net, cfg, ep = 0, logger = logger)
        dist_print('Evaluation result:', res)
        logger.close()
        sys.exit(0)
    # cp_projects(cfg.auto_backup, work_dir)
    max_res = 0
    res = None
    for epoch in range(resume_epoch, cfg.epoch):

        # If we had frozen parameters for initial epochs, check whether to unfreeze now
        try:
            freeze_epochs = int(getattr(cfg, 'freeze_epochs', 0) or getattr(args, 'freeze_epochs', 0))
        except Exception:
            freeze_epochs = 0
        if freeze_active and freeze_epochs > 0:
            if epoch >= resume_epoch + freeze_epochs:
                dist_print(f'Unfreezing parameters after {freeze_epochs} epochs (epoch={epoch})')
                base_net = net.module if hasattr(net, 'module') else net
                prefix = getattr(cfg, 'freeze_prefix', None) or getattr(args, 'freeze_prefix', 'model.')
                for name, p in base_net.named_parameters():
                    if name.startswith(prefix):
                        p.requires_grad = True
                # recreate optimizer and scheduler so new params are included
                optimizer = get_optimizer(net, cfg)
                scheduler = get_scheduler(optimizer, cfg, len(train_loader))
                # If we had set BatchNorm to eval, restore them to train()
                try:
                    if bn_frozen:
                        num_restored = 0
                        for m in base_net.modules():
                            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                                m.train()
                                num_restored += 1
                        dist_print(f'Restored BatchNorm modules to train(): {num_restored}')
                        bn_frozen = False
                except Exception:
                    pass
                freeze_active = False

        vis_int = getattr(cfg, 'vis_interval', None)
        train(net, train_loader, loss_dict, optimizer, scheduler, logger, epoch, metric_dict, cfg.dataset, vis_interval=vis_int)
        train_loader.reset()


        res = eval_lane(net, cfg, ep = epoch, logger = logger)

        if res is not None and res > max_res:
            max_res = res
            save_model(net, optimizer, epoch, work_dir, distributed)
        logger.add_scalar('CuEval/X',max_res,global_step = epoch)

    logger.close()
