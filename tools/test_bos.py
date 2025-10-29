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
    parser.add_argument('--task', default='car', type=str, help='task (e.g., car, pedestrian)')
    parser.add_argument('--detector', default='retinanet', type=str, help='task (e.g., retinanet, faster_rcnn)')
    parser.add_argument('--backbone', default='r50', type=str, help='backbone (e.g., r50, swin)')
    
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


def fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        return x.float().clamp(min, max).half()

    return x.clamp(min, max)


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.
    FP16 Contributed by https://github.com/open-mmlab/mmdetection/pull/4889
    Note:
        Assume bboxes1 is M x 4, bboxes2 is N x 4, when mode is 'iou',
        there are some new generated variable when calculating IOU
        using bbox_overlaps function:
        1) is_aligned is False
            area1: M x 1
            area2: N x 1
            lt: M x N x 2
            rb: M x N x 2
            wh: M x N x 2
            overlap: M x N x 1
            union: M x N x 1
            ious: M x N x 1
            Total memory:
                S = (9 x N x M + N + M) * 4 Byte,
            When using FP16, we can reduce:
                R = (9 x N x M + N + M) * 4 / 2 Byte
                R large than (N + M) * 4 * 2 is always true when N and M >= 1.
                Obviously, N + M <= N * M < 3 * N * M, when N >=2 and M >=2,
                           N + 1 < 3 * N, when N or M is 1.
            Given M = 40 (ground truth), N = 400000 (three anchor boxes
            in per grid, FPN, R-CNNs),
                R = 275 MB (one times)
            A special case (dense detection), M = 512 (ground truth),
                R = 3516 MB = 3.43 GB
            When the batch size is B, reduce:
                B x R
            Therefore, CUDA memory runs out frequently.
            Experiments on GeForce RTX 2080Ti (11019 MiB):
            |   dtype   |   M   |   N   |   Use    |   Real   |   Ideal   |
            |:----:|:----:|:----:|:----:|:----:|:----:|
            |   FP32   |   512 | 400000 | 8020 MiB |   --   |   --   |
            |   FP16   |   512 | 400000 |   4504 MiB | 3516 MiB | 3516 MiB |
            |   FP32   |   40 | 400000 |   1540 MiB |   --   |   --   |
            |   FP16   |   40 | 400000 |   1264 MiB |   276MiB   | 275 MiB |
        2) is_aligned is True
            area1: N x 1
            area2: N x 1
            lt: N x 2
            rb: N x 2
            wh: N x 2
            overlap: N x 1
            union: N x 1
            ious: N x 1
            Total memory:
                S = 11 x N * 4 Byte
            When using FP16, we can reduce:
                R = 11 x N * 4 / 2 Byte
        So do the 'giou' (large than 'iou').
        Time-wise, FP16 is generally faster than FP32.
        When gpu_assign_thr is not -1, it takes more time on cpu
        but not reduce memory.
        There, we can reduce half the memory and keep the speed.
    If ``is_aligned`` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.
    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned`` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.
    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)
    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )
    Example:
        >>> empty = torch.empty(0, 4)
        >>> nonempty = torch.FloatTensor([[0, 0, 10, 9]])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2],
                       bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:],
                       bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


def bbox_xyxy_to_cxcywh(bbox):
    """Convert bbox coordinates from (x1, y1, x2, y2) to (cx, cy, w, h).
    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.
    Returns:
        Tensor: Converted bboxes.
    """
    x1, y1, x2, y2 = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)]
    return torch.cat(bbox_new, dim=-1)


def bbox_cxcywh_to_xyxy(bbox):
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2).
    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.
    Returns:
        Tensor: Converted bboxes.
    """
    cx, cy, w, h = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.cat(bbox_new, dim=-1)


class ClassificationCost:
    """ClsSoftmaxCost.
     Args:
         weight (int | float, optional): loss_weight
     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import \
         ... ClassificationCost
         >>> import torch
         >>> self = ClassificationCost()
         >>> cls_pred = torch.rand(4, 3)
         >>> gt_labels = torch.tensor([0, 1, 2])
         >>> factor = torch.tensor([10, 8, 10, 8])
         >>> self(cls_pred, gt_labels)
         tensor([[-0.3430, -0.3525, -0.3045],
                [-0.3077, -0.2931, -0.3992],
                [-0.3664, -0.3455, -0.2881],
                [-0.3343, -0.2701, -0.3956]])
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
        Returns:
            torch.Tensor: cls_cost value with weight
        """
        # Following the official DETR repo, contrary to the loss that
        # NLL is used, we approximate it in 1 - cls_score[gt_label].
        # The 1 is a constant that doesn't change the matching,
        # so it can be omitted.
        cls_score = cls_pred.softmax(-1)
        cls_cost = -cls_score[:, gt_labels]
        return cls_cost * self.weight


class BBoxL1Cost:
    """BBoxL1Cost.
     Args:
         weight (int | float, optional): loss_weight
         box_format (str, optional): 'xyxy' for DETR, 'xywh' for Sparse_RCNN
     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import BBoxL1Cost
         >>> import torch
         >>> self = BBoxL1Cost()
         >>> bbox_pred = torch.rand(1, 4)
         >>> gt_bboxes= torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
         >>> factor = torch.tensor([10, 8, 10, 8])
         >>> self(bbox_pred, gt_bboxes, factor)
         tensor([[1.6172, 1.6422]])
    """

    def __init__(self, weight=1., box_format='xyxy'):
        self.weight = weight
        assert box_format in ['xyxy', 'xywh']
        self.box_format = box_format

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                (num_query, 4).
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape (num_gt, 4).
        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        if self.box_format == 'xywh':
            gt_bboxes = bbox_xyxy_to_cxcywh(gt_bboxes)
        elif self.box_format == 'xyxy':
            bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight


class IoUCost:
    """IoUCost.
     Args:
         iou_mode (str, optional): iou mode such as 'iou' | 'giou'
         weight (int | float, optional): loss weight
     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import IoUCost
         >>> import torch
         >>> self = IoUCost()
         >>> bboxes = torch.FloatTensor([[1,1, 2, 2], [2, 2, 3, 4]])
         >>> gt_bboxes = torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
         >>> self(bboxes, gt_bboxes)
         tensor([[-0.1250,  0.1667],
                [ 0.1667, -0.5000]])
    """

    def __init__(self, iou_mode='iou', weight=1.): 
        self.weight = weight
        self.iou_mode = iou_mode

    def __call__(self, bboxes, gt_bboxes):
        """
        Args:
            bboxes (Tensor): Predicted boxes with unnormalized coordinates
                (x1, y1, x2, y2). Shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
        Returns:
            torch.Tensor: iou_cost value with weight
        """
        overlaps = bbox_overlaps(bboxes, gt_bboxes, mode=self.iou_mode, is_aligned=False)

        # The 1 is a constant that doesn't change the matching, so omitted.
        iou_cost = -overlaps
        return iou_cost * self.weight


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
        # model with perturbe operation
        model.module.backbone.dropout_uncertainty = args.dropout_uncertainty
        model.module.backbone.drop_layers = args.drop_layers
        model.module.backbone.drop_nn = nn.Dropout(p=args.dropout_uncertainty)
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                args.show_score_thr)
        results_cls_per_dataset_perturbe, results_bbox_per_dataset_perturbe, results_cls_per_img_perturbe, results_bbox_per_img_perturbe = outputs[1:5]
        results_cls_per_dataset_perturbe, results_bbox_per_dataset_perturbe = np.array(results_cls_per_dataset_perturbe), np.array(results_bbox_per_dataset_perturbe) 

        # model without perturbe operation
        model.module.backbone.dropout_uncertainty = 0
        model.module.backbone.drop_layers = []
        model.module.backbone.drop_nn = nn.Dropout(p=0)
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                args.show_score_thr)
        results_cls_per_dataset, results_bbox_per_dataset, results_cls_per_img, results_bbox_per_img = outputs[1:5]
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
    
    reg_loss = BBoxL1Cost()
    iou_loss = IoUCost(iou_mode='giou')
    cls_loss = ClassificationCost()
    
    areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
    areaRngLbl = ['all', 'small', 'medium', 'large']
    num_flag = [0 for area_idx in areaRng]

    iou_matched = []
    cls_matched = []
    iou_cost_perturbe = []
    cls_perturbe = []
    iou_perturbe_matched = []
    cls_perturbe_matched = []
    cost_cls_matched = []

    least_cost = [0 for area_idx in areaRng]
    least_reg_cost_final = [0 for area_idx in areaRng]
    least_iou_cost_final = [0 for area_idx in areaRng]
    least_cls_cost_final = [0 for area_idx in areaRng]

    cls_areaRng = [[] for area_idx in areaRng]
    bboxes_areaRng = []
    entropy_areaRng = [[] for area_idx in areaRng]
        
    for area_idx, area in enumerate(areaRng):
        for img_idx in range(len(results_bbox_per_img)):
            bboxes_raw = results_bbox_per_img[img_idx]
            bboxes_perturbe_raw = results_bbox_per_img_perturbe[img_idx]
            cls_raw = results_cls_per_img[img_idx]
            cls_perturbe_raw = results_cls_per_img_perturbe[img_idx]

            bboxes = []
            bboxes_perturbe = []
            cls = []
            cls_perturbe = []
            
            for b_idx, b in enumerate(bboxes_raw):
                x1, y1, x2, y2 = b
                w = x2 - x1
                h = y2 - y1
                area_b = w * h
                if area_b > area[0] and area_b < area[1]:
                    bboxes.append(b)
                    cls.append(cls_raw[b_idx])
                    cls_areaRng[area_idx].append(cls_raw[b_idx])
                    entropy_areaRng[area_idx].append(stats.entropy([cls_raw[b_idx], 1 - cls_raw[b_idx]], base=2))
            
            for b_idx, b in enumerate(bboxes_perturbe_raw):
                x1, y1, x2, y2 = b
                w = x2 - x1
                h = y2 - y1
                area_b = w * h
                if area_b > area[0] and area_b < area[1]:
                    bboxes_perturbe.append(b)
                    cls_perturbe.append(cls_perturbe_raw[b_idx])

            if area_idx == 0:
                assert len(cls) == len(cls_raw)
                assert len(cls_perturbe) == len(cls_perturbe_raw)

            bboxes = torch.Tensor(bboxes)
            cls = torch.Tensor([cls])
            bboxes_perturbe = torch.Tensor(bboxes_perturbe)
            cls_perturbe = torch.Tensor([cls_perturbe])

            if len(bboxes.shape) < 2 or len(bboxes_perturbe.shape) < 2:
                continue
                    
            sample, _ = bboxes.shape
            sample_perturbe, _ = bboxes_perturbe.shape
            max_match = min(sample, sample_perturbe)

            num_flag[area_idx] += 1
            img_h, img_w, _ =  dataset.prepare_test_img(img_idx)['img_metas'][0].data['img_shape']
            
            factor = bboxes.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
            normalize_bboxes =  bboxes / factor
            normalize_bbox_perturbe =  bboxes_perturbe / factor
            normalize_bbox_perturbe = bbox_xyxy_to_cxcywh(normalize_bbox_perturbe)
            reg_cost = reg_loss(normalize_bbox_perturbe, normalize_bboxes)
            iou_cost = iou_loss(bboxes_perturbe, bboxes)
            
            reg_cost_final = reg_cost
            reg_matched_row_inds, reg_matched_col_inds = linear_sum_assignment(reg_cost_final)
            
            try:
                least_reg_cost_final[area_idx] += reg_cost_final[reg_matched_row_inds, reg_matched_col_inds].sum().numpy().tolist() / max_match
            except:
                import pdb; pdb.set_trace()

            cls_perturbe = torch.transpose(cls_perturbe, 0, 1)
            cls_cost =  cls_perturbe - cls
            cls_cost_final = cls_cost
            cls_matched_row_inds, cls_matched_col_inds = linear_sum_assignment(cls_cost_final)
            least_cls_cost_final[area_idx] += cls_cost_final[cls_matched_row_inds, cls_matched_col_inds].sum().numpy().tolist() / max_match

            cost_cls_matched.extend(cls_cost_final[cls_matched_row_inds, cls_matched_col_inds].numpy().tolist())
            cls2cls_matched = cls[0][cls_matched_col_inds]
            bboxes2cls_matched = bboxes[cls_matched_col_inds]
            cls_pertube2cls_matched = cls_perturbe[cls_matched_row_inds][0]

            cls_matched.extend(cls2cls_matched.numpy().tolist())
            cls_perturbe_matched.extend(cls_pertube2cls_matched.numpy().tolist())

            iou_cost_final = iou_cost
            iou_matched_row_inds, iou_matched_col_inds = linear_sum_assignment(iou_cost_final)
            least_iou_cost_final[area_idx] += iou_cost_final[iou_matched_row_inds, iou_matched_col_inds].sum().numpy().tolist() / max_match
            iou_cost_perturbe.extend(iou_cost_final[iou_matched_row_inds, iou_matched_col_inds].numpy().tolist())

            cls_iou_matched = cls[0][iou_matched_col_inds]
            cls_perturbe_iou_matched = cls_perturbe[iou_matched_row_inds][0]
            bboxes_iou_matched = bboxes[iou_matched_col_inds]

            iou_matched.extend(cls_iou_matched.numpy().tolist())
            iou_perturbe_matched.extend(cls_perturbe_iou_matched.numpy().tolist())

            try:
                cost = 2 * iou_cost + 5 * reg_cost + cls_cost
            except:
                import pdb; pdb.set_trace()
            matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
            least_cost[area_idx] += cost[matched_row_inds, matched_col_inds].sum().numpy().tolist() / max_match
        
        if num_flag[area_idx] != 0:
            least_cost[area_idx] = least_cost[area_idx] / (num_flag[area_idx])
            least_reg_cost_final[area_idx] = least_reg_cost_final[area_idx] / (num_flag[area_idx])
            least_iou_cost_final[area_idx] = least_iou_cost_final[area_idx] / (num_flag[area_idx])
            least_cls_cost_final[area_idx] = least_cls_cost_final[area_idx] / (num_flag[area_idx])
            
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
                metric = dataset.evaluate(outputs[0], **eval_kwargs)  # scores: (10, 101, 1, 4, 3)  precision_score_zero: (10, 1, 4, 3) 
                min_recall = 0
            else:
                metric = dataset.evaluate(outputs, **eval_kwargs)  # scores: (10, 101, 1, 4, 3)
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

            NAME_DIR  = f"./result/{args.task}_inc/BoS/{args.detector}_{args.backbone}/" + str(args.tag1)
            if not os.path.exists(NAME_DIR):
                os.makedirs(NAME_DIR)

            NAME = NAME_DIR + "/" + str(args.tag2) + ".json"

            if not os.path.exists(NAME):
                with open(NAME, "w", encoding="utf-8") as f:
                    json.dump({}, f)
            
            with open(NAME, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data == {}:
                    data.update({0: [[metric['bbox_mAP'], metric['bbox_mAP_50'], metric['bbox_mAP_75'], metric['bbox_mAP_s'], metric['bbox_mAP_m'], metric['bbox_mAP_l']], least_cost, \
                        least_iou_cost_final, least_reg_cost_final, least_cls_cost_final, num_cls_large_k, mean_cls_large_k, results_cls_entropy_small_k]})
                else:
                    key=list(map(float, list(data.keys())))
                    key = max(key)
                    key += 1
                    data.update({key: [[metric['bbox_mAP'], metric['bbox_mAP_50'], metric['bbox_mAP_75'], metric['bbox_mAP_s'], metric['bbox_mAP_m'], metric['bbox_mAP_l']], least_cost, \
                        least_iou_cost_final, least_reg_cost_final, least_cls_cost_final, num_cls_large_k, mean_cls_large_k, results_cls_entropy_small_k]})
            with open(NAME, "w", encoding="utf-8") as f:
                json.dump(data, f)

if __name__ == '__main__':
    main()
