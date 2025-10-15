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
sys.path.append('/data/seungju/BoS')  # 모듈이 있는 상위 디렉토리 경로 추가

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

from ensemble_boxes_wbf_raw import weighted_boxes_fusion_raw

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

def apply_iou_filter(scores: torch.Tensor,
              iou_values_all: torch.Tensor,
              iou_threshold: float,
              init_box: torch.Tensor) -> torch.Tensor:
    """
    Applies Non-Maximum Suppression (NMS) to filter bounding boxes based on their IoU values and confidence scores.

    Args:
        boxes (torch.Tensor): Tensor of shape (N, 4) containing bounding boxes.
        scores (torch.Tensor): Tensor of shape (N,) containing confidence scores for each bounding box.
        iou_values_all (torch.Tensor): Tensor of shape (N, N) containing pairwise IoU values between all bounding boxes.
        iou_threshold (float): IoU threshold to use for filtering boxes.

    Returns:
        torch.Tensor: Indices of bounding boxes that are kept after applying NMS.
    """
    # Number of boxes
    num_boxes = scores.size(0)

     # Sort boxes by scores in descending order
    sorted_indices = scores.sort(descending=True).indices
    sorted_scores = scores[sorted_indices]
    
    # Keep track of which boxes should be kept
    keep = torch.ones(num_boxes, dtype=torch.bool)

    remove_bbox = []
    over_under = 0
    over_under2 = 0 
    # Apply NMS
    j = 0
    no = 0
    modi = 0
    # iou = []
    pseudo_ious = []
    pseudo_ious2 = []
    yes = 0
    for i in range(num_boxes):
        if not keep[sorted_indices[i]]:
            continue
        previous_keep = keep.clone()
        current_idx = sorted_indices[i]
        predicted_score = scores[current_idx]
        # print(predicted_score)
        iou_with_remaining = iou_values_all[current_idx] > iou_threshold
        iou_with_remaining[current_idx] = False  # Do not suppress the current box itself
        predicted_bbox = init_box[current_idx]
        # print(iou_with_remaining.sum())
        # print(keep.sum())
        ########################### predicted idx 제외############ 결과는 # iou_with_remaining이랑 동일 
        # idx_remaining = iou_with_remaining.clone()   # predicted idx 는 제외하기
        # # except_pre = sorted_indices[:i+1]
        # # print(sorted_indices[:i+1])
        # # print(current_idx)
        # # print(keep[sorted_indices[:i+1]])
        # # print(sorted_indices[:i+1])
        # # print(sorted_indices[:i+1][keep[sorted_indices[:i+1]]])
        # except_pre = sorted_indices[:i+1][keep[sorted_indices[:i+1]]]  # predicted idx 는 제외하기
        # # exit()
        # idx_remaining[except_pre] = False    # iou_with_remaining이랑 동일 
        # # print(keep[sorted_indices[:i+1]])
        ###########################################
        removed_box = init_box[iou_with_remaining]
        # print(removed_box.shape)
        remove_score = scores[iou_with_remaining] 
        # removed_box = removed_box[remove_score >= 0.1] # res_revise_upper0.1
        # print(removed_box.shape)
        # exit()
        
        # print()
        # exit()
        ref_bbox = init_box[current_idx]
        # 각 bbox의 좌상단 좌표 (x1, y1)
        # print(removed_box.shape)
        # removed_box = removed_box[remove_score > 0.1]
        # print(removed_box.shape)
        # exit()
        if removed_box.numel() != 0:  # # removed_box가 비어 있지 않을 때 처리
            # if predicted_score.item() > torch.max(remove_score).item():
            #     pseudo_ious2.append(True)
            # else:
            #     pseudo_ious2.append(False)
            x1_min = torch.min(removed_box[:, 0])
            y1_min = torch.min(removed_box[:, 1])
            
            # 각 bbox의 우하단 좌표 (x2, y2)
            x2_max = torch.max(removed_box[:, 2])
            y2_max = torch.max(removed_box[:, 3])
            
            # 하나의 외접 bbox 반환
            pseudo_bbox = torch.tensor([x1_min, y1_min, x2_max, y2_max])
        
            # exit()
            # pseudo_iou = torchvision.ops.box_iou(init_box[current_idx].unsqueeze(0) , pseudo_bbox.unsqueeze(0))
            # pseudo_ious.append(pseudo_iou.item())
            center1_x = (pseudo_bbox[0] + pseudo_bbox[2]) / 2  # (x1 + x2) / 2
            center1_y = (pseudo_bbox[1] + pseudo_bbox[3]) / 2  # (y1 + y2) / 2
            
            center2_x = (ref_bbox[0] + ref_bbox[2]) / 2  # (x1 + x2) / 2
            center2_y = (ref_bbox[1] + ref_bbox[3]) / 2  # (y1 + y2) / 2
            
            # 두 중심점 사이의 유클리드 거리 구하기
            distance = torch.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
            _, _, r1, r2 = ref_bbox
            ref_dist  = torch.sqrt((r1 - center2_x)**2 + (r2 - center2_y)**2) / 2

            
            # print(distance/ref_dist)
            # print(1- distance/ref_dist)

            # print(ref_dist/10)
            # exit()
            
            # pseudo_iou = torchvision.ops.box_iou(init_box[current_idx].unsqueeze(0) , removed_box)
            # print(pseudo_iou)
            # print(remove_score)
            # print(pseudo_iou*remove_score)
            # print(torch.sum(pseudo_iou*remove_score))
            # exit()
            # center_x = (predicted_bbox[0] + predicted_bbox[2]) / 2  # (x1 + x2) / 2
            # center_y = (predicted_bbox[1] + predicted_bbox[3]) / 2  # (y1 + y2) / 2

            # center1_x = (pseudo_bbox[0] + pseudo_bbox[2]) / 2  # (x1 + x2) / 2
            # center1_y = (pseudo_bbox[1] + pseudo_bbox[3]) / 2  # (y1 + y2) / 2
            
            # center2_x = (removed_box[:, 0] + removed_box[:, 2]) / 2  # (x1 + x2) / 2
            # center2_y = (removed_box[:, 1] + removed_box[:, 3]) / 2  # (y1 + y2) / 2
            
            # # 두 중심점 사이의 유클리드 거리 구하기
            # distance_pre_merged = torch.sqrt((center_x - center1_x)**2 + (center_y - center1_y)**2)
            # distance_pre_adjacent = torch.sqrt((center_x - center2_x)**2 + (center_y - center2_y)**2)

            # merged_box_area = torchvision.ops.box_area(pseudo_bbox.unsqueeze(0))
            # removed_box_area = torchvision.ops.box_area(removed_box)
            # # exit()
            # predicted_box_area = torchvision.ops.box_area(predicted_bbox.unsqueeze(0))
            # area_ratio_p_m = merged_box_area/predicted_box_area
            # area_ratio_p_a = removed_box_area/predicted_box_area
            # ratio_merged = area_ratio_p_m/distance_pre_merged
            # ratio_adjacent = torch.max(area_ratio_p_a/distance_pre_adjacent)   # 이것도 뭔가 한 박스 당 구하는 것말고 전체 값들끼리의 조합 중에 max를 골라도 될 듯듯
            # # print(ratio_merged)
            # # print(ratio_merged/ratio_adjacent)
            # pseudo_ious.append(area_ratio_p_m.item())

            #####################
            # pseudo_bbox = pseudo_bbox.unsqueeze(0)
            # ref_bbox = ref_bbox.unsqueeze(0)
            # x1 = torch.max(pseudo_bbox[:, None, 0], ref_bbox[:, 0])
            # y1 = torch.max(pseudo_bbox[:, None, 1], ref_bbox[:, 1])
            # x2 = torch.min(pseudo_bbox[:, None, 2], ref_bbox[:, 2])
            # y2 = torch.min(pseudo_bbox[:, None, 3], ref_bbox[:, 3])
            # inter_w = (x2 - x1).clamp(min=0)
            # inter_h = (y2 - y1).clamp(min=0)
            
            # # 교집합 면적
            # intersection = inter_w * inter_h
            
            # # box1의 면적
            # # pseudo_box_area = torchvision.ops.box_area(pseudo_bbox) # 원래
            # ref_box_area = torchvision.ops.box_area(ref_bbox)
            
            # # 교집합 면적 / box1의 면적
            # inter_over_pseudo_box = intersection / ref_box_area[:, None]
            # pseudo_bbox = pseudo_bbox.squeeze(0)
            # ref_bbox = ref_bbox.squeeze(0)
            # pseudo_ious.append(inter_over_pseudo_box.item())  # area_ratio_p_m.item() 얘랑 성능 똑같이 나옴옴
            #########################
            # if ratio_merged.item() > ratio_adjacent.item():
            # # # if area_ratio_p_m.item() > torch.max(area_ratio_p_a).item():
            # #     # yes += 1

            #     x1_stan, y1_stan, x2_stan, y2_stan = init_box[current_idx] 
            #     width = x2_stan-x1_stan
            #     height = y2_stan-y1_stan
            #     x1, y1, x2, y2 = pseudo_bbox
            #     # 중심 좌표 계산
            #     cx = (x1 + x2) / 2
            #     cy = (y1 + y2) / 2
            #     # width = (x2 - x1) * 0.8
            #     # height = (y2 - y1) * 0.8
            #     half_width = width / 2
            #     half_height = height / 2
            #     new_x1 = cx - half_width
            #     new_y1 = cy - half_height
            #     new_x2 = cx + half_width
            #     new_y2 = cy + half_height

            #     # 새로운 bbox
            #     pseudo_bbox = torch.tensor([new_x1, new_y1, new_x2, new_y2])
                # print("OOOOOOOOOOOO")
                # print(distance)
                # print(ref_dist)
                # print(ref_dist/distance)
                # print(distance/ref_dist)
                # pseudo_iou = torchvision.ops.box_iou(init_box[current_idx].unsqueeze(0) , pseudo_bbox.unsqueeze(0))
                # print(pseudo_iou)
            # else:
                
            #     print("XXXXXXXXXXXX")
            #     print(distance)
            #     print(ref_dist)
            #     print(ref_dist/distance)
            #     print(distance/ref_dist)
            #     pseudo_iou = torchvision.ops.box_iou(init_box[current_idx].unsqueeze(0) , pseudo_bbox.unsqueeze(0))
            #     print(pseudo_iou)
            # removed_iou = torchvision.ops.box_iou(removed_box, removed_box)
            # iou_mean = torch.mean(removed_iou)
            # pseudo_bbox = pseudo_bbox.unsqueeze(0)
            # ref_bbox = ref_bbox.unsqueeze(0)
            # x1 = torch.max(pseudo_bbox[:, None, 0], ref_bbox[:, 0])
            # y1 = torch.max(pseudo_bbox[:, None, 1], ref_bbox[:, 1])
            # x2 = torch.min(pseudo_bbox[:, None, 2], ref_bbox[:, 2])
            # y2 = torch.min(pseudo_bbox[:, None, 3], ref_bbox[:, 3])
            # inter_w = (x2 - x1).clamp(min=0)
            # inter_h = (y2 - y1).clamp(min=0)
            
            # # 교집합 면적
            # intersection = inter_w * inter_h
            
            # # box1의 면적
            # # pseudo_box_area = torchvision.ops.box_area(pseudo_bbox) # 원래
            # ref_box_area = torchvision.ops.box_area(ref_bbox)
            
            # # 교집합 면적 / box1의 면적
            # inter_over_pseudo_box = intersection / ref_box_area[:, None]
            # pseudo_bbox = pseudo_bbox.squeeze(0)
            # # iou.append(iou_mean.item())
            # removed_iou = torchvision.ops.box_iou(removed_box, removed_box)
            # diag_mask = torch.eye(removed_iou.size(0), dtype=torch.bool)

            # # 대각 성분 제외한 값들만 필터링
            # non_diag_values = removed_iou[~diag_mask]

            # # 평균 계산
            # iou_mean = torch.mean(non_diag_values)
            # ref_iou = iou_values_all[current_idx][iou_with_remaining]
            # iou_mean2 = torch.mean(ref_iou)
            # print(iou_mean)
            # iou_mean = torch.mean(removed_iou)
            # print(iou_mean)
            # exit()
            # iou.append(iou_mean.item())
            # pseudo_iou = torchvision.ops.box_iou(init_box[current_idx].unsqueeze(0) , pseudo_bbox.unsqueeze(0))
            # print(pseudo_iou)
            # if iou_mean.item() < 0.65:   # 0.65이 베스트 0.7
            #     yes += 1
            # # if (iou_mean.item() < 0.65) & (inter_over_pseudo_box.item() != 1):
            # # if (iou_mean.item() < 0.7) & (iou_mean2.item() < 0.7) & (inter_over_pseudo_box.item() >= 2/3):
            #     # print(inter_over_pseudo_box.item())
            #     # print(iou_mean.item())
            #     yes += 1
            #     # exit()
            #     # modi += 1
            #     # print(remove_score)
            #     # modi_removed_box = removed_box[remove_score > 0.1]
            #     # modi_removed_iou = torchvision.ops.box_iou(modi_removed_box, modi_removed_box)
            #     # modi_iou_mean = torch.mean(modi_removed_iou)
            #     # print(modi_iou_mean)
            #     ######### resize modi############33
            #     # ref_bbox = ref_bbox.squeeze(0)
            #     # x1, y1, x2, y2 = ref_bbox
            #     # target_width = y2 - y1
            #     # # target_width = x2 - x1
            #     # x_min, y_min, x_max, y_max = pseudo_bbox

            #     # # 중심 좌표 계산
            #     # center_x = (x_min + x_max) / 2
            #     # center_y = (y_min + y_max) / 2

            #     # # 원본 bbox의 가로와 세로 길이 계산
            #     # original_width = x_max - x_min
            #     # original_height = y_max - y_min

            #     # # 비율 유지하면서 높이를 조정
            #     # scale_factor = target_width / original_width
            #     # new_width = target_width
            #     # new_height = original_height * scale_factor

            #     # # 비율 유지한 새로운 bbox 좌표 계산
            #     # new_x_min = center_x - new_width / 2
            #     # new_y_min = center_y - new_height / 2
            #     # new_x_max = center_x + new_width / 2
            #     # new_y_max = center_y + new_height / 2

            #     # # 결과 bbox 반환
            #     # pseudo_bbox = torch.tensor([new_x_min, new_y_min, new_x_max, new_y_max])
            #     ######ori#####
            #     x1_stan, y1_stan, x2_stan, y2_stan = init_box[current_idx] 
            #     width = x2_stan-x1_stan
            #     height = y2_stan-y1_stan
            #     x1, y1, x2, y2 = pseudo_bbox
            #     # 중심 좌표 계산
            #     cx = (x1 + x2) / 2
            #     cy = (y1 + y2) / 2
            #     # width = (x2 - x1) * 0.8
            #     # height = (y2 - y1) * 0.8
            #     half_width = width / 2
            #     half_height = height / 2
            #     new_x1 = cx - half_width
            #     new_y1 = cy - half_height
            #     new_x2 = cx + half_width
            #     new_y2 = cy + half_height

            #     # 새로운 bbox
            #     pseudo_bbox = torch.tensor([new_x1, new_y1, new_x2, new_y2])
            #     # print("################################################################3")
            #     # pseudo_iou = torchvision.ops.box_iou(init_box[current_idx].unsqueeze(0) , pseudo_bbox.unsqueeze(0))
            #     # print(pseudo_iou)
        
            pseudo_iou = torchvision.ops.box_iou(init_box[current_idx].unsqueeze(0) , pseudo_bbox.unsqueeze(0))
            # print(pseudo_iou)
            # exit()
            # pseudo_ious.append(torch.mean(pseudo_iou*remove_score).item()) 
            # pseudo_iou1 = 0.7 * (1-distance/ref_dist) + 0.3 * pseudo_iou
            pseudo_ious.append(pseudo_iou.item())
            # pseudo_iou2 = 0.5 * (1-distance/ref_dist) + 0.5 * pseudo_iou
            pseudo_ious2.append((1-distance/ref_dist).item())
            # print(pseudo_iou.item())    
        else:
            no += 1

            pseudo_ious.append(0)
            pseudo_ious2.append(0)
            # pseudo_ious.append(100)
            # pseudo_ious2.append(True)  # 이거
            # pseudo_ious2.append(False)
            # pseudo_ious.append(1)
            
        keep &= ~iou_with_remaining
        # remove_bbox.append(keep^previous_keep)  # "&":두 값이 모두 True일 때만 결과가 True가 됨

        remove_idx = keep^previous_keep
        # remove_score = sorted_scores[remove_idx.bool()] #이렇게 하면 안됨
        remove_score = scores[remove_idx.bool()]  #이렇게 해야 함
        #*************************over_under = remove_score>=(sorted_scores[i])   # 이런 경우는 없음 
        j += 1
    # print(j)
    # exit()
    # Return indices of boxes that are kept
    # print(iou)
    # print(len(pseudo_ious))
    # score = torch.tensor(pseudo_ious) <= 0.5
    # print(len(pseudo_ious))
    # print(score.sum())
    # print(no)
    # exit()
    kept_indices = sorted_indices[keep.bool()]
    # print("num", num)
    # print("크기조절:", modi)  # 40 생각보다 크기 조절이 많이 필요함 / thr 0.6인 경우 4 / thr 0.5인 경우 0
    # print(yes)
    # print(keep.sum())
    # print(scores[keep])
    # print(pseudo_ious2)
    # print(len(pseudo_ious2))
    # print(np.sum(pseudo_ious2))
    # print(sorted_indices)
    # print(pseudo_ious)
    # exit()

    # return kept_indices, remove_bbox
    return keep, torch.tensor(pseudo_ious), torch.tensor(pseudo_ious2)


def apply_iou_filter_no_sort(scores: torch.Tensor,
              iou_values_all: torch.Tensor,
              iou_threshold: float) -> torch.Tensor:
    """
    Applies Non-Maximum Suppression (NMS) to filter bounding boxes based on their IoU values and confidence scores.

    Args:
        boxes (torch.Tensor): Tensor of shape (N, 4) containing bounding boxes.
        scores (torch.Tensor): Tensor of shape (N,) containing confidence scores for each bounding box.
        iou_values_all (torch.Tensor): Tensor of shape (N, N) containing pairwise IoU values between all bounding boxes.
        iou_threshold (float): IoU threshold to use for filtering boxes.

    Returns:
        torch.Tensor: Indices of bounding boxes that are kept after applying NMS.
    """
    # Number of boxes
    num_boxes = scores.size(0)

     # Sort boxes by scores in descending order
    sorted_indices = scores.sort(descending=True).indices
    sorted_scores = scores[sorted_indices]

    # Keep track of which boxes should be kept
    keep = torch.ones(num_boxes, dtype=torch.bool)

    remove_bbox = []
    num = 0
    # Apply NMS
    for i in range(num_boxes):
        if not keep[i]:
            continue
        previous_keep = keep.clone()
        
        iou_with_remaining = iou_values_all[i] > iou_threshold
        iou_with_remaining[i] = False  # Do not suppress the current box itself
        keep &= ~iou_with_remaining
        remove_bbox.append(keep^previous_keep)  # "&":두 값이 모두 True일 때만 결과가 True가 됨
        
        ## find over&underconfidenct

        ######### method4 
        # print()
        # print()
        # print("################################################")
        # print(iou_values_all[current_idx])
        remove_idx = keep^previous_keep
        # remove_score = sorted_scores[remove_idx.bool()] 
        remove_score = scores[remove_idx.bool()] 
        # score_filter = remove_score <= 
        over_under = remove_score>=(sorted_scores[i])   # 이런 경우는 없음 
        # print(over_under)
        # exit()
        # over_under = remove_score>=(sorted_scores[i]+0.4)
        over_under = remove_score>=(sorted_scores[i])
        if over_under.float().sum() > 0 :
            num += over_under.float().sum()/len(remove_score)

        ######### method1
        # remove_idx = keep^previous_keep
        # remove_score = sorted_scores[remove_idx.bool()]
        # over_under = remove_score[remove_score>=sorted_scores[i]]
        # over_under = remove_score>=sorted_scores[i]
        # num += over_under.sum()


        
    # Return indices of boxes that are kept
    kept_indices = sorted_indices[keep.bool()]
    # print("num", num)
    # exit()
    # return kept_indices, remove_bbox
    return keep, num

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
    
class BetaCalibration:   # car처럼 단일 클래스에 대한거임 
    def __init__(self, use_cuda=False, method='mle'):
        """
        Beta Calibration 클래스 (단일 클래스)
        
        Args:
            use_cuda (bool): CUDA를 사용할지 여부.
            method (str): 매개변수 추정 방법 ('mle').
        """
        self.use_cuda = use_cuda
        self.method = method

    def fit(self, probs: torch.Tensor, labels: torch.Tensor):
        """
        베타 분포의 매개변수를 피팅합니다.
        
        Args:
            probs (torch.Tensor): 모델의 예측 확률 (또는 detection에서의 confidence score), shape (N,).
            labels (torch.Tensor): 실제 클래스 레이블, shape (N,).
        """
        if self.method == 'mle':
            self.alpha, self.beta = self._mle_fit(probs, labels)
        else:
            raise NotImplementedError(f"지원하지 않는 방법입니다: {self.method}")

    def _mle_fit(self, probs: torch.Tensor, labels: torch.Tensor):
        """
        MLE를 사용하여 alpha와 beta 매개변수를 추정합니다.
        
        Args:
            probs (torch.Tensor): 모델의 예측 확률 또는 confidence score.
            labels (torch.Tensor): 실제 레이블.
        
        Returns:
            (float, float): 추정된 alpha와 beta 값.
        """
        # # 성공 및 실패 횟수 계산
        # success_count = torch.sum(labels).item()
        # failure_count = len(labels) - success_count

        # 초기 alpha와 beta 값 설정
        initial_params = [1.0, 1.0]

        # MLE를 사용하여 alpha와 beta 최적화
        def negative_log_likelihood(params):
            alpha, beta = params
            likelihood = (
                (alpha - 1) * torch.log(probs + 1e-10) + (beta - 1) * torch.log(1 - probs + 1e-10)
            )
            return -torch.sum(likelihood).item()

        # scipy의 minimize 함수를 사용하여 최적화
        result = minimize(negative_log_likelihood, initial_params, bounds=[(1e-3, None), (1e-3, None)])
        alpha_mle, beta_mle = result.x
        
        return alpha_mle, beta_mle

    def predict(self, probs: torch.Tensor) -> torch.Tensor:
        """
        예측 확률을 보정합니다 (Detection에서는 confidence score를 보정합니다).
        
        Args:
            probs (torch.Tensor): Detection task의 confidence score 또는 예측 확률, shape (N,).
        
        Returns:
            torch.Tensor: 보정된 확률 또는 보정된 confidence score, shape (N,).
        """
        if self.use_cuda:
            probs = probs.cuda()

        # Beta 보정된 확률 계산
        calibrated_probs = (probs ** self.alpha) / ((probs ** self.alpha) + ((1 - probs) ** self.beta))
        
        return calibrated_probs

def main():
    args = parse_args()
    # exit()
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
        # scores_per_dataset, scores_ori_per_dataset, scores_per_img, scores_ori_per_img = outputs[5:]
        results_cls_per_dataset_ori, results_cls_per_img_ori, iou_alls, iou_exs, init_boxes = outputs[5:]
        
        # bbox = outputs[5:]
        # re = outputs[-1]
        # print(len(bbox))
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
    iou_loss = IoUCost()
    cls_loss = ClassificationCost()
    beta_cal = BetaCalibration(use_cuda=True, method='mle')
    

    # print(outputs)
    # print(len(outputs[0]))  # 250
    # print(len(outputs)) # 9
    # print(outputs[0][0])  # bbox정보가 담겨 있음 
    # exit()
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
    result_measures = [0 for area_idx in areaRng]
    result_over_under = [0 for area_idx in areaRng]
    result_over_under_2 = [0 for area_idx in areaRng]
    result_over_under_3 = [0 for area_idx in areaRng]
    result_over_under_4 = [0 for area_idx in areaRng]
    result_over_under_5 = [0 for area_idx in areaRng]
    result_over_under_6 = [0 for area_idx in areaRng]
    result_over_under_7 = [0 for area_idx in areaRng]
    result_over_under_8 = [0 for area_idx in areaRng]
    result_over_under_9 = [0 for area_idx in areaRng]
    result_over_under_10 = [0 for area_idx in areaRng]
    result_over_under_11 = [0 for area_idx in areaRng]
    result_over_under_12 = [0 for area_idx in areaRng]
    

    cls_areaRng = [[] for area_idx in areaRng]
    bboxes_areaRng = []
    entropy_areaRng = [[] for area_idx in areaRng]
    # print("a",len(results_bbox_per_dataset)) # 생성된 box의 수  27370
    # print(len(outputs[0]))  # 250
    # print(len(results_cls_per_dataset))  # 27370
    # print("b", len(results_bbox_per_img)) # 이미지 수   250
    for area_idx, area in enumerate(areaRng):
        # measure_len = len(results_bbox_per_img)
        measures = []
        over_under = []
        over_under_2 = []
        over_under_3 = []
        over_under_4 = []
        over_under_5 = []
        over_under_6 = []
        over_under_7 = []
        over_under_8 = []
        over_under_9 = []
        over_under_10 = []
        over_under_11 = []
        over_under_12 = []

        for img_idx in range(len(results_bbox_per_img)):
            # print(len(iou_alls))  # 250
            # print(len(iou_exs))  # 250
            # print(len(results_bbox_per_img))  # 250
            # print(len(init_boxes))  # 250
        
            ########### checking ###############
            '''
            print(outputs[0][img_idx])  # 모든 bbox + score  # torch.Size([1, 100, 5])
            print(outputs[1][img_idx])  # score
            print(outputs[2][img_idx])  # bbox
            print(outputs[3][img_idx])  # 모든 score (outputs[0][0]보다 더 많음 7개)
            print(outputs[4][img_idx])  # 모든 bbox (outputs[0][0]보다 더 많음 7개)
            print(outputs[5][img_idx])  # 어떤 값
            print(outputs[6][img_idx])  # 모든 어떤 값
            print(outputs[7][img_idx][0])  # 모름
            print(outputs[8][img_idx])  # iou 
            print(outputs[8][img_idx][0].shape)  # 107
            print(outputs[7][img_idx][0].shape)  #(832,)

            
            
            bboxes_raw = torch.tensor(results_bbox_per_img[img_idx])
            # print(out[:,0,:])   # score 높은 순으로 되어있음   ,  score_ex들에 대한 bbox랑 score가 있음 
            # print(bboxes_raw[0])  #### 이 둘이 같음  # 근데 사이즈 다름 

            # print(out[:,99,:])
            # print(bboxes_raw[99])  #### 이 둘이가 같음  # 근데 사이즈 다름 
            # print(out.shape)  # torch.Size([1, 100, 5])
            print("###################################")
            iou_bbox = torchvision.ops.box_iou(bboxes_raw, bboxes_raw)   # outputs[8][img_idx]랑 같음 
            print(iou_bbox.shape)  # torch.Size([107, 107])
            print(iou_bbox[0])

            out = torch.tensor(outputs[0][img_idx])
            iou_out = torchvision.ops.box_iou(out[:,:,:4].squeeze(0), out[:,:,:4].squeeze(0))
            print(iou_out.shape)  # torch.Size([100, 100])   # 앞선 iou_bbox랑 같은데 길이가 다름 
            print(iou_out[0])
            '''
            ########################################################
            # ex_mean = np.mean(results_cls_per_img[img_idx])
           

            # # print(len(results_cls_per_img_ori[img_idx]))  # 832
            # ori = sorted(results_cls_per_img_ori[img_idx], reverse=True)
            # # print(np.mean(results_cls_per_img[img_idx]))
            # ori_mean = np.mean(results_cls_per_img_ori[img_idx])
            # measure = ori_mean / ex_mean
            # # print(measure)
            # measures.append(measure)
            # print("############################################") ddd
            score_all = torch.tensor(results_cls_per_img_ori[img_idx])
            score_ex = torch.tensor(results_cls_per_img[img_idx])
            iou_all = iou_alls[img_idx]
            iou_ex = iou_exs[img_idx]
            iou_out_ex = torch.tensor(np.array(outputs[8][img_idx]))   # iou_ex랑 똑같음 
            # score_out_ex = torch.tensor(np.array(outputs[3][img_idx]))  # score_ex랑 똑같음 
            init_box = torch.tensor(np.array(init_boxes[img_idx]))
            # print(score_all.shape)
            # print(init_box.shape)
            if len(iou_all) > 0 :
                iou_all = torch.tensor(np.array(iou_all))
                first_filter = score_all>0.05
                score_all = score_all[first_filter.bool()]
                iou_all = iou_all[first_filter.bool(),:][:, first_filter.bool()]
                init_box = init_box[first_filter.bool(),:]
                keep, pseudo_ious, pseudo_ious2 = apply_iou_filter(score_all, iou_all, 0.5, init_box)
                conf = torch.tensor(results_cls_per_img[img_idx])
                if conf.shape != pseudo_ious.shape:
                    conf, _ = torch.sort(score_all[keep], descending=True)
                    print("nnnnnnnnnnnnnnnnnnnnnnnnnnn")  # coco 49 에서만 4번 있음

                
                # print(pseudo_ious)
                # print(conf)
                # exit()

                # conf  = 1 / (1 + np.exp(50 * (conf - 0.5)))
                # pseudo_ious = 1 / (1 + np.exp(-50 * (pseudo_ious - 0.5)))
                # count_soft = conf * pseudo_ious


                # conf1  = 1 / (1 + np.exp(100 * (conf - 0.5)))
                # pseudo_ious1 = 1 / (1 + np.exp(-100 * (pseudo_ious - 0.7)))

                # count_soft = conf1 * pseudo_ious1
                # if count_soft.sum()/keep.sum() > 0:
                #     over_under.append(count_soft.sum()/keep.sum())


                # count_soft = (conf1**0.6) * (pseudo_ious1**0.4)
                # if count_soft.sum()/keep.sum() > 0:
                #     over_under_2.append(count_soft.sum()/keep.sum())

                # count_soft = (conf1**0.4) * (pseudo_ious1**0.6)
                # if count_soft.sum()/keep.sum() > 0:
                #     over_under_3.append(count_soft.sum()/keep.sum())

                # conf2  = 1 / (1 + np.exp(100 * (conf - 0.7)))
                # pseudo_ious2 = 1 / (1 + np.exp(-100 * (pseudo_ious - 0.5)))

                # count_soft = conf2 * pseudo_ious2
                # if count_soft.sum()/keep.sum() > 0:
                #     over_under_4.append(count_soft.sum()/keep.sum())


                # count_soft = (conf2**0.6) * (pseudo_ious2**0.4)
                # if count_soft.sum()/keep.sum() > 0:
                #     over_under_5.append(count_soft.sum()/keep.sum())

                # count_soft = (conf2**0.4) * (pseudo_ious2**0.6)
                # if count_soft.sum()/keep.sum() > 0:
                #     over_under_6.append(count_soft.sum()/keep.sum())

                # conf3  = 1 / (1 + np.exp(100 * (conf - 0.7)))
                # pseudo_ious3 = 1 / (1 + np.exp(-100 * (pseudo_ious - 0.7)))

                # count_soft = conf3 * pseudo_ious3
                # if count_soft.sum()/keep.sum() > 0:
                #     over_under_7.append(count_soft.sum()/keep.sum())


                # count_soft = (conf3**0.6) * (pseudo_ious3**0.4)
                # if count_soft.sum()/keep.sum() > 0:
                #     over_under_8.append(count_soft.sum()/keep.sum())

                # count_soft = (conf3**0.4) * (pseudo_ious3**0.6)
                # if count_soft.sum()/keep.sum() > 0:
                #     over_under_9.append(count_soft.sum()/keep.sum())

                # res_ours1_soft_vary_center_k100부터는 over_under_10~12 추가 , k=100, 75
                # res_ours1_revise_no_resize 일 때때
                # com_iou = 0.5*pseudo_ious + 0.5*pseudo_ious2
                # com_iou2 = 0.3*pseudo_ious + 0.7*pseudo_ious2
                # com_iou3 = 0.7*pseudo_ious + 0.3*pseudo_ious2
                # com_iou4 = 0.4*pseudo_ious + 0.6*pseudo_ious2
                # com_iou5 = 0.5*pseudo_ious + 0.4*pseudo_ious2
                conf_up = conf > 0.5
                conf_down = conf <= 0.5
                com_iou = pseudo_ious
                com_iou8 = pseudo_ious[conf_up]
                com_iou9 = pseudo_ious[conf_down]
                pseudo_ious2 = 1 / (1 + np.exp(-100 * (pseudo_ious2 - 0.5)))
                com_iou2 = pseudo_ious2
                com_iou3 = 0.99*pseudo_ious + 0.01*pseudo_ious2
                com_iou4 = pseudo_ious + 0.2*pseudo_ious2
                com_iou5 = pseudo_ious + 0.1*pseudo_ious2
                com_iou6 = pseudo_ious + 0.05*pseudo_ious2
                com_iou7 = pseudo_ious + 0.15*pseudo_ious2
                conf  = 1 / (1 + np.exp(100 * (conf - 0.5)))
                com_iou = 1 / (1 + np.exp(-100 * (com_iou - 0.5)))
                # com_iou2 = 1 / (1 + np.exp(-100 * (com_iou2 - 0.5)))
                com_iou3 = 1 / (1 + np.exp(-100 * (com_iou3 - 0.5)))
                com_iou4 = 1 / (1 + np.exp(-100 * (com_iou4 - 0.5)))
                com_iou5 = 1 / (1 + np.exp(-100 * (com_iou5 - 0.5)))
                com_iou6 = 1 / (1 + np.exp(-100 * (com_iou6 - 0.5)))
                com_iou7 = 1 / (1 + np.exp(-100 * (com_iou7 - 0.5)))
                # count_soft = conf * com_iou
                # count_soft2 = conf * com_iou2
                # count_soft3 = conf * com_iou3
                # count_soft4 = conf * com_iou4
                # count_soft5 = conf * com_iou5
                # count_soft6 = conf * com_iou6
                # count_soft7 = conf * com_iou7

                count_soft = (conf+com_iou)/2
                count_soft2 = (conf+com_iou2)/2
                count_soft3 = (conf+com_iou3)/2
                count_soft4 = (conf+com_iou4)/2
                count_soft5 = (conf+com_iou5)/2
                count_soft6 = (conf+com_iou6)/2
                count_soft7 = (conf+com_iou7)/2

                # count_soft = conf * pseudo_ious
                if count_soft.sum()/keep.sum() > 0:
                    over_under.append(count_soft.sum()/keep.sum())

                if count_soft2.sum()/keep.sum() > 0:
                    over_under_2.append(count_soft2.sum()/keep.sum())

                if count_soft3.sum()/keep.sum() > 0:
                    over_under_3.append(count_soft3.sum()/keep.sum())

                if count_soft4.sum()/keep.sum() > 0:
                    over_under_4.append(count_soft4.sum()/keep.sum())

                if count_soft5.sum()/keep.sum() > 0:
                    over_under_5.append(count_soft5.sum()/keep.sum())

                if count_soft6.sum()/keep.sum() > 0:
                    over_under_6.append(count_soft6.sum()/keep.sum())
                if count_soft7.sum()/keep.sum() > 0:
                    over_under_7.append(count_soft7.sum()/keep.sum())
                ################################final은 여기까지만 
                com_iou5 = pseudo_ious + 0.1*pseudo_ious2
                com_iou5 = 1 / (1 + np.exp(-100 * (com_iou5 - 0.7)))
                count_soft5 = (conf+com_iou5)/2
                if count_soft5.sum()/keep.sum() > 0:
                    over_under_8.append(count_soft5.sum()/keep.sum())

                com_iou5 = pseudo_ious + 0.1*pseudo_ious2
                com_iou5 = 1 / (1 + np.exp(-100 * (com_iou5 - 0.4)))
                count_soft5 = (conf+com_iou5)/2
                if count_soft5.sum()/keep.sum() > 0:
                    over_under_9.append(count_soft5.sum()/keep.sum())

                com_iou5 = pseudo_ious + 0.1*pseudo_ious2
                com_iou5 = 1 / (1 + np.exp(-50 * (com_iou5 - 0.5)))
                count_soft5 = (conf+com_iou5)/2
                if count_soft5.sum()/keep.sum() > 0:
                    over_under_10.append(count_soft5.sum()/keep.sum())
                # if torch.mean(count_soft4) > 0:  # keep으로 나누는 것보다 더 안 좋음 
                #     over_under_8.append(torch.mean(count_soft4))
                # count_soft = (conf**0.6) * (pseudo_ious**0.4)
                # if count_soft.sum()/keep.sum() > 0:
                #     over_under_11.append(count_soft.sum()/keep.sum())

                # count_soft = (conf**0.4) * (pseudo_ious**0.6)
                # if count_soft.sum()/keep.sum() > 0:
                #     over_under_12.append(count_soft.sum()/keep.sum())

                # conf_up = conf > 0.5
                # conf_down = conf <= 0.5
                # iou_up = pseudo_ious > 0.5
                # iou_down = pseudo_ious <= 0.5

                # iou_up = pseudo_ious > 0.7
                # iou_down = pseudo_ious <= 0.7
                # conf_up = conf > 0.5
                # conf_down = conf <= 0.5
                # iou_up = pseudo_ious > 0.4
                # iou_down = pseudo_ious <= 0.4
                # print(pseudo_ious)
                # exit()

                # score_up = conf_up & iou_up
                # score_up_down = conf_up & iou_down
                # score_down_up = conf_down & iou_up
                # score_down = conf_down & iou_down

                # iou_up2 = pseudo_ious2 > 0.5
                # iou_down2 = pseudo_ious2 <= 0.5

                # score_up2 = conf_up & iou_up2
                # score_up_down2 = conf_up & iou_down2
                # score_down_up2 = conf_down & iou_up2
                # score_down2 = conf_down & iou_down2

                # score_up = conf_up & pseudo_ious2
                # score_up_down = conf_up & ~pseudo_ious2
                # score_down_up = conf_down & pseudo_ious2
                # score_down = conf_down & ~pseudo_ious2
                
                # print(score_up.sum())  #12 -> 3
                # print(score_up_down.sum()) #2 -> 0
                # print(score_down_up.sum()) #51 -> 73
                # print(score_down.sum()) #19 -> 31
                # print(keep.sum()) #84 -> 107
                # print(score_down_up.sum()/keep.sum())
                # print(score_down.sum()/keep.sum())
                # exit()

        
                # if (score_down_up.sum()/keep.sum()) > 0 :
                #     over_under.append(score_down_up.sum()/keep.sum())
                # if (score_down_up.sum()+score_up.sum())/keep.sum() > 0:
                #     over_under_2.append((score_down_up.sum()+score_up.sum())/keep.sum())
                # if (score_down_up.sum()+score_down.sum())/keep.sum() > 0:
                #     over_under_3.append((score_down_up.sum()+score_down.sum())/keep.sum())
                # if (score_down.sum()/keep.sum()) > 0 :
                #     over_under_4.append(score_down.sum()/keep.sum())
                # if (score_up.sum()/keep.sum()) > 0 :
                #     over_under_5.append(score_up.sum()/keep.sum())
                # if (score_up_down.sum()/keep.sum()) > 0 :
                #     over_under_6.append(score_up_down.sum()/keep.sum())
                # if (pseudo_ious2.sum()/keep.sum()) > 0 :
                #     over_under_7.append(pseudo_ious2.sum()/keep.sum())
                   
                # if (score_down_up2.sum()/keep.sum()) > 0 :
                #     over_under_4.append(score_down_up2.sum()/keep.sum())
                # if (score_down_up2.sum()+score_up2.sum())/keep.sum() > 0:
                #     over_under_5.append((score_down_up2.sum()+score_up2.sum())/keep.sum())
                # if (score_down_up2.sum()+score_down2.sum())/keep.sum() > 0:
                #     over_under_6.append((score_down_up2.sum()+score_down2.sum())/keep.sum())                  
            #####################################################
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

            bboxes = torch.Tensor(np.array(bboxes))
            cls = torch.Tensor([cls])
            bboxes_perturbe = torch.Tensor(np.array(bboxes_perturbe))
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
            # print(iou_cost)
            # exit()
            reg_cost_final = reg_cost
            # print(reg_cost_final)
            # exit()
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
        # exit()   
        least_cost[area_idx] = least_cost[area_idx] / (num_flag[area_idx])
        least_reg_cost_final[area_idx] = least_reg_cost_final[area_idx] / (num_flag[area_idx])  # BS
        least_iou_cost_final[area_idx] = least_iou_cost_final[area_idx] / (num_flag[area_idx])
        least_cls_cost_final[area_idx] = least_cls_cost_final[area_idx] / (num_flag[area_idx])
        # result_measures[area_idx] = np.mean(measures).item()
        result_over_under[area_idx] = np.mean(over_under).item()
        result_over_under_2[area_idx] = np.mean(over_under_2).item()
        result_over_under_3[area_idx] = np.mean(over_under_3).item()
        result_over_under_4[area_idx] = np.mean(over_under_4).item()
        result_over_under_5[area_idx] = np.mean(over_under_5).item()
        result_over_under_6[area_idx] = np.mean(over_under_6).item()
        result_over_under_7[area_idx] = np.mean(over_under_7).item()
        result_over_under_8[area_idx] = np.mean(over_under_8).item()
        result_over_under_9[area_idx] = np.mean(over_under_9).item()
        result_over_under_10[area_idx] = np.mean(over_under_10).item()
        result_over_under_11[area_idx] = np.mean(over_under_11).item()
        result_over_under_12[area_idx] = np.mean(over_under_12).item()
        break
    # exit()
    print()
    # print(result_over_under)
    # print(result_over_under_2)
    # print(result_over_under_3)
    # print(result_over_under_4)
    # print(result_over_under_5)
    # exit()
    # print(result_over_under_4)
    # print(result_over_under_7)
    # print(result_over_under_10)
    # print(result_over_under_11)
    # exit()
    # print(result_over_under_3) # 3_2랑 차이 많이 남
    # print(result_over_under_3_2)
    # print()
    #     print(least_iou_cost_final[area_idx])
        # print(result_measures[area_idx])
    #     exit()
    # exit()        
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
                metric = dataset.evaluate(outputs[0],    **eval_kwargs)  # scores: (10, 101, 1, 4, 3)  precision_score_zero: (10, 1, 4, 3) 
                min_recall = 0
            else:
                metric = dataset.evaluate(outputs, **eval_kwargs)  # scores: (10, 101, 1, 4, 3)

           



            print(metric)
            metric_dict = dict(config=args.config, metric=metric)
            if args.work_dir is not None and rank == 0:
                mmcv.dump(metric_dict, json_file)
        # print("done")
        # exit()
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

            NAME_DIR  = "./01.17/01.16/res_ours1_revise_no_resize_CCC_S0.1/" + str(args.tag1)
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
                        least_iou_cost_final,  result_over_under, result_over_under_2, result_over_under_3, result_over_under_4, result_over_under_5, result_over_under_6, result_over_under_7, result_over_under_8, result_over_under_9, result_over_under_10, result_over_under_11, result_over_under_12, least_reg_cost_final, least_cls_cost_final, num_cls_large_k, mean_cls_large_k, results_cls_entropy_small_k]})
                else:
                    key=list(map(float, list(data.keys())))
                    key = max(key)
                    key += 1
                    data.update({key: [[metric['bbox_mAP'], metric['bbox_mAP_50'], metric['bbox_mAP_75'], metric['bbox_mAP_s'], metric['bbox_mAP_m'], metric['bbox_mAP_l']], least_cost, \
                        least_iou_cost_final,  result_over_under , result_over_under_2, result_over_under_3, result_over_under_4, result_over_under_5, result_over_under_6, result_over_under_7, result_over_under_8, result_over_under_9, result_over_under_10, result_over_under_11, result_over_under_12, least_reg_cost_final, least_cls_cost_final, num_cls_large_k, mean_cls_large_k, results_cls_entropy_small_k]})
            with open(NAME, "w", encoding="utf-8") as f:
                json.dump(data, f)

if __name__ == '__main__':
    main()
