# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
import sys
sys.path.append(os.getcwd())

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import (build_ddp, build_dp, compat_cfg, get_device,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)


import mmdet_custom

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
import json
import copy
from scipy import stats
import cv2
import matplotlib.pyplot as plt
from mmdet.datasets import build_dataset, get_loading_pipeline
import seaborn as sns
from scipy.optimize import linear_sum_assignment
import torch.nn as nn
import torchvision

import torch.nn.functional as F
from scipy.optimize import minimize



def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--tag1',
        default='',
        help='tag1')
    parser.add_argument(
        '--tag2',
        default='',
        help='tag2')
    parser.add_argument(
        '--dropout_uncertainty',
        type=float,
        default=0.01,
        help='tag')
    parser.add_argument(
        '--drop_layers',
        nargs='+', 
        type=int,
        default=[])
    parser.add_argument('--alpha', default=0.1, type=float, help='alpha')
    parser.add_argument('--slope_con', default=-10, type=int, help='k_C')
    parser.add_argument('--slope_rel', default=20, type=int, help='k_R')
    parser.add_argument('--conf_th', default=0.4, type=int, help='c')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args

def sigmoid(score, center=0.6, slope=10, base=0.1):
    return base + (1-base) / (1 + np.exp(-slope * (score - center)))

def apply_iou_filter(scores: torch.Tensor,
              iou_values_all: torch.Tensor,
              iou_threshold: float,
              init_box: torch.Tensor, args) -> torch.Tensor:
    
    num_boxes = scores.size(0)
    sorted_indices = scores.sort(descending=True).indices
    sorted_scores = scores[sorted_indices]
    keep = torch.ones(num_boxes, dtype=torch.bool)


    consistency_iou = []
    consistency_closeness = []
    reliability = []

    for i in range(num_boxes):
        if not keep[sorted_indices[i]]:
            continue
        previous_keep = keep.clone()
        current_idx = sorted_indices[i]
        iou_with_remaining = iou_values_all[current_idx] > iou_threshold
        iou_with_remaining[current_idx] = False  
        
        removed_box = init_box[iou_with_remaining]
        remove_score = scores[iou_with_remaining] 
        
        ref_bbox = init_box[current_idx]
      
        
        reliability.append((args.alpha + (1-args.alpha) / (1 + np.exp(-args.slope_rel * (remove_score - args.conf_th)))).sum().item())
       
        if removed_box.numel() != 0: 
          
            x1_min = torch.min(removed_box[:, 0])
            y1_min = torch.min(removed_box[:, 1])
            x2_max = torch.max(removed_box[:, 2])
            y2_max = torch.max(removed_box[:, 3])
          
            pseudo_bbox = torch.tensor([x1_min, y1_min, x2_max, y2_max])

            center1_x = (pseudo_bbox[0] + pseudo_bbox[2]) / 2  
            center1_y = (pseudo_bbox[1] + pseudo_bbox[3]) / 2  
            
            center2_x = (ref_bbox[0] + ref_bbox[2]) / 2 
            center2_y = (ref_bbox[1] + ref_bbox[3]) / 2  
            
    
            distance = torch.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
            _, _, r1, r2 = ref_bbox
            ref_dist  = torch.sqrt((r1 - center2_x)**2 + (r2 - center2_y)**2) / 2

            pseudo_iou = torchvision.ops.box_iou(init_box[current_idx].unsqueeze(0) , pseudo_bbox.unsqueeze(0))
            consistency_iou.append(pseudo_iou.item())
            consistency_closeness.append((1-distance/ref_dist).item())
        else:
            consistency_iou.append(0)
            consistency_closeness.append(0)
           
            
        keep &= ~iou_with_remaining
     
    return keep, torch.tensor(consistency_iou), torch.tensor(consistency_closeness), torch.tensor(reliability)

    
def main():
    args = parse_args()
    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    elif 'init_cfg' in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None

    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]
    cfg.device = get_device()

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    # model with perturbe operation
    # import pdb; pdb.set_trace()
    

    if not distributed:
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
        model.module.backbone.dropout_uncertainty = 0
        model.module.backbone.drop_layers = []
        model.module.backbone.drop_nn = nn.Dropout(p=0)
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                args.show_score_thr)
        results_cls_per_dataset, results_bbox_per_dataset, results_cls_per_img, results_bbox_per_img = outputs[1:5]
        results_cls_per_dataset_ori, results_cls_per_img_ori, iou_alls, init_boxes = outputs[5:]
        results_cls_per_dataset, results_bbox_per_dataset = np.array(results_cls_per_dataset), np.array(results_bbox_per_dataset)
    else:
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False)
        outputs = multi_gpu_test(
            model, data_loader, args.tmpdir, args.gpu_collect
            or cfg.evaluation.get('gpu_collect', False))
    

    areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
    areaRngLbl = ['all', 'small', 'medium', 'large']
    num_flag = [0 for area_idx in areaRng]

    result_consistency = [0 for area_idx in areaRng]
    result_reliability = [0 for area_idx in areaRng]
 
    cls_areaRng = [[] for area_idx in areaRng]
    bboxes_areaRng = []
    entropy_areaRng = [[] for area_idx in areaRng]
  
    for area_idx, area in enumerate(areaRng):
        score_consistency = []
        score_reliability = []
        for img_idx in range(len(results_bbox_per_img)):
           
            score_all = torch.tensor(results_cls_per_img_ori[img_idx])
            iou_all = iou_alls[img_idx]
            init_box = torch.tensor(np.array(init_boxes[img_idx]))
           
            if len(iou_all) > 0 :
                iou_all = torch.tensor(np.array(iou_all))
                first_filter = score_all>0.05
                score_all = score_all[first_filter.bool()]
                keep, consistency_iou, consistency_closeness, reliability = apply_iou_filter(score_all, iou_all, 0.5, init_box, args)
                conf, _ = torch.sort(score_all[keep], descending=True)
                
                # consistency
                iou_closeness = 0.5*consistency_iou + 0.5*consistency_closeness
                conf_down = sigmoid(conf, args.conf_th, args.slope_con, 0)
                low_count = iou_closeness * conf_down
                if low_count.sum()/keep.sum() > 0:
                    score_consistency.append(low_count.sum()/keep.sum())

                # reliability
                conf_up = conf > args.conf_th
                under = (sigmoid(score_all[~keep], args.conf_th, args.slope_rel, args.alpha)).sum()
                ratio = reliability[conf_up].sum()/under
                if under.item() > 0 :
                    score_reliability.append(ratio.item())
               
        result_consistency[area_idx] = np.mean(score_consistency).item()
        result_reliability[area_idx] = np.mean(score_reliability).item()
    
   
    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            if len(outputs) >= 4:
                mmcv.dump(outputs[0], args.out)
            else:
                mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            if len(outputs) >= 4:
                dataset.format_results(outputs[0], **kwargs)
            else:
                dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule', 'dynamic_intervals'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            
            if len(outputs) >= 4:
                metric = dataset.evaluate(outputs[0],    **eval_kwargs)  
                min_recall = 0
            else:
                metric = dataset.evaluate(outputs, **eval_kwargs) 

            print(metric)
            metric_dict = dict(config=args.config, metric=metric)
            if args.work_dir is not None and rank == 0:
                mmcv.dump(metric_dict, json_file)
      
        num_flag = [0 for area_idx in areaRng]
        if len(outputs) >= 4:
            num_cls_large_k = [[] for a in areaRng]
            mean_cls_large_k = [[] for a in areaRng]
            results_cls_entropy_small_k = [[] for a in areaRng]

            for area_idx, area in enumerate(areaRng):
                for score_thr in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
                    cls_areaRng[area_idx] = np.array(cls_areaRng[area_idx])
                    entropy_areaRng[area_idx] = np.array(entropy_areaRng[area_idx])
                    num_cls_large_k[area_idx].append(len(cls_areaRng[area_idx][cls_areaRng[area_idx]>score_thr]))
                    mean_cls_large_k[area_idx].append(np.mean(cls_areaRng[area_idx][cls_areaRng[area_idx]>score_thr]).tolist())
                    results_cls_entropy_small_k[area_idx].append(np.mean(entropy_areaRng[area_idx][cls_areaRng[area_idx]<score_thr]).tolist())
            
            # Change the directory according to 
            # (1) the task, (vehicle detection or pedestrian detection)
            # and (2) the model architecture 
            NAME_DIR  = "./result/car_ori/PCR/r50_retina/" + str(args.tag1)
            if not os.path.exists(NAME_DIR):
                os.makedirs(NAME_DIR)

            NAME = NAME_DIR + "/" + str(args.tag2) + ".json"

            if not os.path.exists(NAME):
                with open(NAME, "w", encoding="utf-8") as f:
                    json.dump({}, f)
            
            with open(NAME, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data == {}:
                    data.update({0: [[metric['bbox_mAP'], metric['bbox_mAP_50'], metric['bbox_mAP_75'], metric['bbox_mAP_s'], metric['bbox_mAP_m'], metric['bbox_mAP_l']], \
                        result_consistency, result_reliability]})
                else:
                    key=list(map(float, list(data.keys())))
                    key = max(key)
                    key += 1
                    data.update({key: [[metric['bbox_mAP'], metric['bbox_mAP_50'], metric['bbox_mAP_75'], metric['bbox_mAP_s'], metric['bbox_mAP_m'], metric['bbox_mAP_l']],  \
                        result_consistency, result_reliability]})
            with open(NAME, "w", encoding="utf-8") as f:
                json.dump(data, f)

if __name__ == '__main__':
    main()
