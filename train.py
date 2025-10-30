import torch, os, datetime


from utils.dist_utils import dist_print, dist_tqdm, synchronize
from utils.factory import get_metric_dict, get_loss_dict, get_optimizer, get_scheduler
from utils.metrics import update_metrics, reset_metrics

from utils.common import calc_loss, get_model, get_train_loader, inference, merge_config, save_model, cp_projects
from utils.common import get_work_dir, get_logger

import time
from evaluation.eval_wrapper import eval_lane
import matplotlib.pyplot as plt
import numpy as np
import sys

def train(net, data_loader, loss_dict, optimizer, scheduler,logger, epoch, metric_dict, dataset):
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
            if logger is not None and (global_step % 200 == 0):
                fig = plt.figure(figsize=(6,4))
                ax = fig.add_subplot(1,1,1)

                img_np = None
                seg_np = None
                pts = None

                # DALI path: data_label is dict; pytorch path: tuple (imgs, labels)
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

                    if 'labels_row_float' in data_label:
                        lr = data_label['labels_row_float'][0].cpu().numpy()
                        pts = ('row_float', lr)

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

                    cls_np = cls[0].numpy()
                    pts = ('cls_grid', cls_np)

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
                            arr = data
                            if arr.ndim == 3:
                                arr = arr[0]
                            if arr.shape[0] < arr.shape[1]:
                                arr = arr.T
                            num_rows = arr.shape[0]
                            y_coords = np.linspace(0, H-1, num_rows)
                            for lane_idx in range(arr.shape[1]):
                                xs = arr[:,lane_idx]
                                valid = (xs >= 0) & (xs <= 1)
                                ax.plot(xs[valid] * W, y_coords[valid], '.', markersize=4)
                        elif typ == 'cls_grid':
                            cls_np = data
                            n_rows, n_lanes = cls_np.shape
                            y_coords = np.linspace(0, H-1, n_rows)
                            valid_vals = cls_np[cls_np < 10000]
                            if valid_vals.size == 0:
                                num_cols = 200
                            else:
                                num_cols = int(np.max(valid_vals)) + 1
                            for lane in range(n_lanes):
                                for ri in range(n_rows):
                                    c = cls_np[ri, lane]
                                    if c < 0 or c >= num_cols:
                                        continue
                                    x = c * (W - 1) / max(1, (num_cols - 1))
                                    y = y_coords[ri]
                                    ax.plot(x, y, 'y.', markersize=3)

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
    net = get_model(cfg)

    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids = [args.local_rank])
    optimizer = get_optimizer(net, cfg)

    if cfg.finetune is not None:
        dist_print('finetune from ', cfg.finetune)
        state_all = torch.load(cfg.finetune)['model']
        state_clip = {}  # only use backbone parameters
        for k,v in state_all.items():
            if 'model' in k:
                state_clip[k] = v
        net.load_state_dict(state_clip, strict=False)
    if cfg.resume is not None:
        dist_print('==> Resume model from ' + cfg.resume)
        resume_dict = torch.load(cfg.resume, map_location='cpu')
        net.load_state_dict(resume_dict['model'])
        if 'optimizer' in resume_dict.keys():
            optimizer.load_state_dict(resume_dict['optimizer'])
        resume_epoch = int(os.path.split(cfg.resume)[1][2:5]) + 1
    else:
        resume_epoch = 0

    # If evaluation-only mode requested, optionally load provided test weights
    if args.eval_only:
        test_path = args.test_model if hasattr(args, 'test_model') and args.test_model is not None else getattr(cfg, 'test_model', None)
        if test_path is not None:
            dist_print('Loading weights for evaluation from', test_path)
            try:
                ckpt = torch.load(test_path, map_location='cpu')
                if isinstance(ckpt, dict) and 'model' in ckpt:
                    net.load_state_dict(ckpt['model'])
                else:
                    try:
                        net.load_state_dict(ckpt)
                    except Exception:
                        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                            net.load_state_dict(ckpt['state_dict'])
                        else:
                            dist_print('Warning: could not load model weights from', test_path)
            except Exception as e:
                dist_print('Error loading test model:', e)

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

        train(net, train_loader, loss_dict, optimizer, scheduler,logger, epoch, metric_dict, cfg.dataset)
        train_loader.reset()

        res = eval_lane(net, cfg, ep = epoch, logger = logger)

        if res is not None and res > max_res:
            max_res = res
            save_model(net, optimizer, epoch, work_dir, distributed)
        logger.add_scalar('CuEval/X',max_res,global_step = epoch)

    logger.close()
