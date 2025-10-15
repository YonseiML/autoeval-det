import numpy as np
import torch.nn as nn
import torch

from mmdet.models.dense_heads.anchor_head import AnchorHead
from mmcv.cnn import bias_init_with_prob, ConvModule, normal_init

# added by WSK
from mmdet.core import (multi_apply, multiclass_nms, bbox_overlaps, images_to_levels)

from mmdet_custom.core import (anchor_target, delta2bbox, 
                        weighted_cross_entropy,
                        weighted_smoothl1, weighted_binary_cross_entropy,
                        weighted_sigmoid_focal_loss)
from mmdet_custom.core.loss import weighted_iou_regression_loss
from mmdet_custom.core.anchor.anchor_target import expand_binary_labels

from mmdet.models.builder import HEADS, build_loss
from mmcv.ops import DeformConv2d, MaskedConv2d 

class CFG(nn.Module):
    def __init__(self):
        self.allowed_border = None

class FeatureAlignment(nn.Module):
    """Feature Adaption Module.
    Feature Adaption Module is implemented based on DCN v1.
    It uses anchor shape prediction rather than feature map to
    predict offsets of deformable conv layer.
    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels in the output feature map.
        kernel_size (int): Deformable conv kernel size.
        deformable_groups (int): Deformable conv group size.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 deformable_groups=4):
        super(FeatureAlignment, self).__init__()
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(
            4, deformable_groups * offset_channels, 1, bias=False)
        self.conv_adaption = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deformable_groups=deformable_groups)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        normal_init(self.conv_offset, std=0.1)
        normal_init(self.conv_adaption, std=0.01)

    def forward(self, x, shape):
        offset = self.conv_offset(shape.detach())
        x = self.relu(self.conv_adaption(x, offset))
        return x


@HEADS.register_module
class IoUawareRetinaHead(AnchorHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='retina_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(IoUawareRetinaHead, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(self.feat_channels, self.num_anchors*4, 3, padding=1)

        # added by WSK
        # analyze the effect of the shared conv layers between regression head and
        # IoU prediction head. The number of conv layers to extract features for
        # IoU prediction have to be kept to be 4.
        self.shared_conv = 4
        if self.shared_conv < 4:
            self.iou_convs = nn.ModuleList()
            for i in range(4-self.shared_conv):
                chn = self.in_channels if (self.shared_conv==0 and i == 0) else self.feat_channels
                self.iou_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg)
                )

        # feature alignment for IoU prediction
        self.use_feature_alignment = False
        self.deformable_groups = 4
        if self.use_feature_alignment:
            self.feature_alignment = FeatureAlignment(
                self.feat_channels,
                self.feat_channels,
                kernel_size=3,
                deformable_groups = self.deformable_groups)
            self.retina_iou = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        else:
            self.retina_iou = nn.Conv2d(self.feat_channels, self.num_anchors, 3, padding=1)


    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)
        # added by WSK
        if self.shared_conv < 4:
            for m in self.iou_convs:
                normal_init(m.conv, std=0.01)
        normal_init(self.retina_iou, std=0.01)

        # added by WSK
        if self.use_feature_alignment:
            self.feature_alignment.init_weights()


    def forward_single(self, x):
        """
        process one level of FPN
        :param x: one feature level of FPN. tensor of size (batch, self.feat_channels, width_i, height_i)
        :return:
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)

        reg_feat_list = []
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
            reg_feat_list.append(reg_feat)
        cls_score = self.retina_cls(cls_feat) # (batch, A*num_class, width_i, height_i)
        bbox_pred = self.retina_reg(reg_feat) # (batch, A*4, width_i, height_i)

        #added by WSK
        # concatenation of regression prediction and feature map for the input of
        # IoU prediction head
        # bbox_pred_clone = bbox_pred.clone()
        # bbox_pred_clone = bbox_pred_clone.detach()
        # reg_feat = torch.cat([reg_feat_list[-1],bbox_pred_clone], 1)

        # analyze the effect of the shared conv layers between regression head and
        # IoU prediction head.
        if self.shared_conv == 0:
            iou_feat = x
        else:
            iou_feat = reg_feat_list[self.shared_conv - 1]
        if self.shared_conv < 4:
            for iou_conv in self.iou_convs:
                iou_feat = iou_conv(iou_feat)
        # iou_pred = self.retina_iou(iou_feat) # (batch, A, width_i, height_i)

        # feature alignment for iou prediction
        if self.use_feature_alignment:
            bbox_pred_list = torch.split(bbox_pred, 4, dim=1)
            iou_pred_list = []
            for i in range(len(bbox_pred_list)):
                iou_feat_aligned = self.feature_alignment(iou_feat, bbox_pred_list[i])
                iou_pred_single_anchor = self.retina_iou(iou_feat_aligned) # (batch, 1, width_i, height_i)
                iou_pred_list.append(iou_pred_single_anchor)
            iou_pred = torch.cat(iou_pred_list, 1) # (batch, A, width_i, height_i)
        else:
            iou_pred = self.retina_iou(iou_feat)  # (batch, A, width_i, height_i)

        return cls_score, bbox_pred, iou_pred

    def loss_single(self, cls_score, bbox_pred, iou_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor wight
                shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)

        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)

        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)

        iou = bbox_overlaps(bbox_targets, bbox_pred, is_aligned=True)  # (batch*width_i*height_i*A)
        iou_pred = iou_pred.permute(0, 2, 3, 1).reshape(-1)  # (batch*width_i*height_i*A)
        bbox_weight_list = torch.split(bbox_weights, 1, -1)
        bbox_weight = bbox_weight_list[0]
        bbox_weight = torch.squeeze(bbox_weight)  # (batch*A*width_i*height_i)
        weight_iou = 1.0
        loss_iou = weight_iou * weighted_iou_regression_loss(iou_pred, iou, bbox_weight, avg_factor=num_total_samples)

        return loss_cls, loss_bbox, loss_iou


    # @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'iou_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             iou_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        # from IPython import embed; embed()
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=None,
            gt_labels_list=None,
            label_channels=1,
            unmap_outputs=True,
            return_sampling_results=False)

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets

        # cls_reg_targets = anchor_target(
        #     anchor_list,
        #     valid_flag_list,
        #     gt_bboxes,
        #     img_metas,
        #     self.target_means,
        #     self.target_stds,
        #     cfg,
        #     gt_bboxes_ignore_list=gt_bboxes_ignore,
        #     gt_labels_list=gt_labels,
        #     label_channels=label_channels,
        #     sampling=self.sampling)
        # if cls_reg_targets is None:
        #     return None
        # (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg,
        #  level_anchor_list) = cls_reg_targets

        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        losses_cls, losses_bbox, losses_iou = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            iou_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)

        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, losses_iou=losses_iou)


    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   iou_preds,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each level in the
                feature pyramid, has shape
                (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each
                level in the feature pyramid, has shape
                (N, num_anchors * 4, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.

        Example:
            >>> import mmcv
            >>> self = AnchorHead(
            >>>     num_classes=9,
            >>>     in_channels=1,
            >>>     anchor_generator=dict(
            >>>         type='AnchorGenerator',
            >>>         scales=[8],
            >>>         ratios=[0.5, 1.0, 2.0],
            >>>         strides=[4,]))
            >>> img_metas = [{'img_shape': (32, 32, 3), 'scale_factor': 1}]
            >>> cfg = mmcv.Config(dict(
            >>>     score_thr=0.00,
            >>>     nms=dict(type='nms', iou_thr=1.0),
            >>>     max_per_img=10))
            >>> feat = torch.rand(1, 1, 3, 3)
            >>> cls_score, bbox_pred = self.forward_single(feat)
            >>> # note the input lists are over different levels, not images
            >>> cls_scores, bbox_preds = [cls_score], [bbox_pred]
            >>> result_list = self.get_bboxes(cls_scores, bbox_preds,
            >>>                               img_metas, cfg)
            >>> det_bboxes, det_labels = result_list[0]
            >>> assert len(result_list) == 1
            >>> assert det_bboxes.shape[1] == 5
            >>> assert len(det_bboxes) == len(det_labels) == cfg.max_per_img
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)

        mlvl_cls_scores = [cls_scores[i].detach() for i in range(num_levels)]
        mlvl_bbox_preds = [bbox_preds[i].detach() for i in range(num_levels)]
        mlvl_iou_preds = [iou_preds[i].detach() for i in range(num_levels)]

        if torch.onnx.is_in_onnx_export():
            assert len(
                img_metas
            ) == 1, 'Only support one input image while in exporting to ONNX'
            img_shapes = img_metas[0]['img_shape_for_onnx']
        else:
            img_shapes = [
                img_metas[i]['img_shape']
                for i in range(cls_scores[0].shape[0])
            ]
        scale_factors = [
            img_metas[i]['scale_factor'] for i in range(cls_scores[0].shape[0])
        ]

        if with_nms:
            # some heads don't support with_nms argument
            result_list = self._get_bboxes(mlvl_cls_scores, mlvl_bbox_preds, mlvl_iou_preds,
                                           mlvl_anchors, img_shapes,
                                           scale_factors, cfg, rescale)
        else:
            result_list = self._get_bboxes(mlvl_cls_scores, mlvl_bbox_preds, mlvl_iou_preds,
                                           mlvl_anchors, img_shapes,
                                           scale_factors, cfg, rescale,
                                           with_nms)
        return result_list


    def _get_bboxes(self,
                    mlvl_cls_scores,
                    mlvl_bbox_preds,
                    mlvl_iou_preds,
                    mlvl_anchors,
                    img_shapes,
                    scale_factors,
                    cfg,
                    rescale=False,
                    with_nms=True):
        """Transform outputs for a batch item into bbox predictions.

        Args:
            mlvl_cls_scores (list[Tensor]): Each element in the list is
                the scores of bboxes of single level in the feature pyramid,
                has shape (N, num_anchors * num_classes, H, W).
            mlvl_bbox_preds (list[Tensor]):  Each element in the list is the
                bboxes predictions of single level in the feature pyramid,
                has shape (N, num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Each element in the list is
                the anchors of single level in feature pyramid, has shape
                (num_anchors, 4).
            img_shapes (list[tuple[int]]): Each tuple in the list represent
                the shape(height, width, 3) of single image in the batch.
            scale_factors (list[ndarray]): Scale factor of the batch
                image arange as list[(w_scale, h_scale, w_scale, h_scale)].
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(mlvl_cls_scores) == len(mlvl_bbox_preds) == len(
            mlvl_anchors)
        batch_size = mlvl_cls_scores[0].shape[0]
        # convert to tensor to keep tracing
        nms_pre_tensor = torch.tensor(
            cfg.get('nms_pre', -1),
            device=mlvl_cls_scores[0].device,
            dtype=torch.long)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_ious = []
        for cls_score, bbox_pred, iou_pred, anchors in zip(mlvl_cls_scores,
                                                 mlvl_bbox_preds,
                                                 mlvl_iou_preds,
                                                 mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(0, 2, 3,
                                          1).reshape(batch_size, -1,
                                                     self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(0, 2, 3,
                                          1).reshape(batch_size, -1, 4)
            iou_pred = iou_pred.permute(0, 2, 3,
                                          1).reshape(batch_size, -1)
            iou_pred = iou_pred.sigmoid()
            anchors = anchors.expand_as(bbox_pred)

            # compute the IoU between the regressed anchors and the ground truth boxes
            bboxes_pred_decode = self.bbox_coder.decode(anchors, bbox_pred)

            # bboxes_pred_decode = delta2bbox(anchors, bbox_pred, self.target_means,
            #                                 self.target_stds, img_shape)  # (width_i*height_i*A, 4)

            # iou_truth = iou_pred.new_zeros(iou_pred.size())

            # multiply classification score with the class-agnostic IoU to compute the final
            # detection confidence
            iou_expanded = iou_pred.view(-1, 1).expand(-1, scores.size(-1))
            # iou_expanded = iou_truth.view(-1, 1).expand(-1, scores.size(-1))
            # scores = scores * iou_expanded
            alpha = 0.5
            scores = scores.pow(alpha) * iou_expanded.pow(1 - alpha)

            # Always keep topk op for dynamic input in onnx
            from mmdet.core.export import get_k_for_topk
            nms_pre = get_k_for_topk(nms_pre_tensor, bbox_pred.shape[1])
            if nms_pre > 0:
                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(-1)
                else:
                    # remind that we set FG labels to [0, num_class-1]
                    # since mmdet v2.0
                    # BG cat_id: num_class
                    max_scores, _ = scores[..., :-1].max(-1)

                _, topk_inds = max_scores.topk(nms_pre)
                batch_inds = torch.arange(batch_size).view(
                    -1, 1).expand_as(topk_inds)
                anchors = anchors[batch_inds, topk_inds, :]
                bbox_pred = bbox_pred[batch_inds, topk_inds, :]
                scores = scores[batch_inds, topk_inds, :]

            bboxes = self.bbox_coder.decode(
                anchors, bbox_pred, max_shape=img_shapes)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
        if rescale:
            batch_mlvl_bboxes /= batch_mlvl_bboxes.new_tensor(
                scale_factors).unsqueeze(1)
        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)

        # Replace multiclass_nms with ONNX::NonMaxSuppression in deployment
        if torch.onnx.is_in_onnx_export() and with_nms:
            from mmdet.core.export import add_dummy_nms_for_onnx
            # ignore background class
            if not self.use_sigmoid_cls:
                num_classes = batch_mlvl_scores.shape[2] - 1
                batch_mlvl_scores = batch_mlvl_scores[..., :num_classes]
            max_output_boxes_per_class = cfg.nms.get(
                'max_output_boxes_per_class', 200)
            iou_threshold = cfg.nms.get('iou_threshold', 0.5)
            score_threshold = cfg.score_thr
            nms_pre = cfg.get('deploy_nms_pre', -1)
            return add_dummy_nms_for_onnx(batch_mlvl_bboxes, batch_mlvl_scores,
                                          max_output_boxes_per_class,
                                          iou_threshold, score_threshold,
                                          nms_pre, cfg.max_per_img)
        if self.use_sigmoid_cls:
            # Add a dummy background class to the backend when using sigmoid
            # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
            # BG cat_id: num_class
            padding = batch_mlvl_scores.new_zeros(batch_size,
                                                  batch_mlvl_scores.shape[1],
                                                  1)
            batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)

        if with_nms:
            det_results = []
            for (mlvl_bboxes, mlvl_scores) in zip(batch_mlvl_bboxes,
                                                  batch_mlvl_scores):
                det_bbox, det_label = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                     cfg.score_thr, cfg.nms,
                                                     cfg.max_per_img)
                det_results.append(tuple([det_bbox, det_label]))
        else:
            det_results = [
                tuple(mlvl_bs)
                for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores)
            ]
        return det_results
