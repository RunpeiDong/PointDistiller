# Copyright (c) OpenMMLab. All rights reserved.
# --------------------------------------------------------
# PointDistiller
# Copyright (c) 2022-2023 Runpei Dong & Linfeng Zhang
# Licensed under The MIT License [see LICENSE for details]
# Written by OpenMMLab
# Hacked by Runpei Dong & Linfeng Zhang
# --------------------------------------------------------
import torch

from mmdet.models import DETECTORS
from .two_stage import TwoStage3DDetector

from mmcv.cnn import ConvModule
from mmdet3d.utils.kd_utils import *

@DETECTORS.register_module()
class PointRCNN(TwoStage3DDetector):
    r"""PointRCNN detector.

    Please refer to the `PointRCNN <https://arxiv.org/abs/1812.04244>`_

    Args:
        backbone (dict): Config dict of detector's backbone.
        neck (dict, optional): Config dict of neck. Defaults to None.
        rpn_head (dict, optional): Config of RPN head. Defaults to None.
        roi_head (dict, optional): Config of ROI head. Defaults to None.
        train_cfg (dict, optional): Train configs. Defaults to None.
        test_cfg (dict, optional): Test configs. Defaults to None.
        pretrained (str, optional): Model pretrained path. Defaults to None.
        init_cfg (dict, optional): Config of initialization. Defaults to None.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(PointRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        
        self.kd_cfg = train_cfg.get("kd_cfg", None)
        
        # KD layer configurations
        if self.kd_cfg is not None:
            import math
            self.compression_ratio = self.kd_cfg['compression_ratio']
            
            self.stu_channel_reduction = int(math.sqrt(self.compression_ratio)) if self.compression_ratio != 8 else 2.8
            student_channel = int(128 // self.stu_channel_reduction)
            
            self.teacher_gcn_adapt = ConvModule(in_channels=128*2,
                                                out_channels=student_channel,
                                                kernel_size=(1, 1),
                                                stride=(1, 1),
                                                conv_cfg=dict(type='Conv2d'),
                                                norm_cfg=dict(type='BN2d'),
                                                act_cfg=dict(type='ReLU'),
                                                bias='auto')
            self.student_gcn_adapt = ConvModule(in_channels=student_channel*2,
                                                out_channels=student_channel,
                                                kernel_size=(1, 1),
                                                stride=(1, 1),
                                                conv_cfg=dict(type='Conv2d'),
                                                norm_cfg=dict(type='BN2d'),
                                                act_cfg=dict(type='ReLU'),
                                                bias='auto')

        # KD layers for PointRCNN
        # 8x compression (teacher)
        # self.teacher_gcn_adapt = ConvModule(in_channels=128*2,
        #                                     out_channels=45,
        #                                     kernel_size=(1, 1),
        #                                     stride=(1, 1),
        #                                     conv_cfg=dict(type='Conv2d'),
        #                                     norm_cfg=dict(type='BN2d'),
        #                                     act_cfg=dict(type='ReLU'),
        #                                     bias='auto')

        # 16x compression (teacher)
        # self.teacher_gcn_adapt = ConvModule(in_channels=128*2,
        #                                     out_channels=32,
        #                                     kernel_size=(1, 1),
        #                                     stride=(1, 1),
        #                                     conv_cfg=dict(type='Conv2d'),
        #                                     norm_cfg=dict(type='BN2d'),
        #                                     act_cfg=dict(type='ReLU'),
        #                                     bias='auto')

        # 8x compression (student)
        # self.student_gcn_adapt = ConvModule(in_channels=45*2,
        #                                     out_channels=45,
        #                                     kernel_size=(1, 1),
        #                                     stride=(1, 1),
        #                                     conv_cfg=dict(type='Conv2d'),
        #                                     norm_cfg=dict(type='BN2d'),
        #                                     act_cfg=dict(type='ReLU'),
        #                                     bias='auto')

        # 16x compression (student)
        # self.student_gcn_adapt = ConvModule(in_channels=32*2,
        #                                     out_channels=32,
        #                                     kernel_size=(1, 1),
        #                                     stride=(1, 1),
        #                                     conv_cfg=dict(type='Conv2d'),
        #                                     norm_cfg=dict(type='BN2d'),
        #                                     act_cfg=dict(type='ReLU'),
        #                                     bias='auto')

    def get_teacher_info(self, points, img_metas, gt_bboxes_3d, gt_labels_3d):
        points_cat = torch.stack(points)
        x, info_dict = self.extract_feat(points_cat, return_more_info=True)

        info_dict['teacher_feature'] = x
        return info_dict

    def extract_feat(self, points, return_more_info=False):
        """Directly extract features from the backbone+neck.

        Args:
            points (torch.Tensor): Input points.

        Returns:
            dict: Features from the backbone+neck
        """
        info_dict = {}
        x = self.backbone(points)
        info_dict['backbone_features'] = x['sa_features']

        if self.with_neck:
            x = self.neck(x)
        info_dict['fp_features'] = x['fp_features'].clone()
        info_dict['fp_xyz'] = x['fp_xyz'].clone()
        if return_more_info is True:
            return x, info_dict
        return x

    def forward_train(self, points, img_metas, gt_bboxes_3d, gt_labels_3d, teacher_info=None):
        """Forward of training.

        Args:
            points (list[torch.Tensor]): Points of each batch.
            img_metas (list[dict]): Meta information of each sample.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): gt bboxes of each batch.
            gt_labels_3d (list[torch.Tensor]): gt class labels of each batch.

        Returns:
            dict: Losses.
        """

        def get_knn_point_backbone(teacher_feature, student_feature, point_saliency=None, 
                                   num_points=6000, kneighbours=128, reweight=False,
                                   pool_mode='none', sample_mode='top', fuse_mode='idendity'):
            # B C N -> B N C: batch_size x npoints x channel
            teacher_feature = teacher_feature.transpose(2, 1).contiguous()
            student_feature = student_feature.transpose(2, 1).contiguous()

            if point_saliency is None:
                point_saliency = pool_features1d(teacher_feature).squeeze(-1)
                # point_saliency = teacher_feature.abs().mean(-1)
            # query_point_idx: batch x num_points
            if sample_mode == 'top':
                _, query_point_idx = torch.topk(point_saliency, num_points, dim=-1, largest=True, sorted=False)
            elif sample_mode == 'bottom':
                _, query_point_idx = torch.topk(point_saliency, num_points, dim=-1, largest=False, sorted=False)
            elif sample_mode == 'rand':
                noise = torch.rand_like(point_saliency)
                _, query_point_idx = torch.topk(noise, num_points, dim=-1, largest=False, sorted=False)
            elif sample_mode == 'mixed':
                _, query_point_idx_pos = torch.topk(point_saliency, num_points//2, dim=-1, largest=True, sorted=False)
                _, query_point_idx_neg = torch.topk(point_saliency, num_points//2, dim=-1, largest=False, sorted=False)
                query_point_idx = torch.cat([query_point_idx_pos, query_point_idx_neg])
            else:
                raise NotImplementedError
            # query_feature_xxx: B x num_points x channel
            query_feature_tea, query_feature_stu = index_feature(teacher_feature, query_point_idx), index_feature(student_feature, query_point_idx)
            # cluster_idx: B x num_points x kneighbours
            # cluster_idx = query_ball_feature(radius, kneighbours, teacher_feature, query_feature_tea) # ball query
            cluster_idx = knn_feature(kneighbours, teacher_feature, query_feature_tea) # knn
            # grouped_points_xxx: B x num_points x kneighbours x channel -> B x channel x num_points x kneighbours
            grouped_points_tea = index_feature(teacher_feature, cluster_idx).permute(0, 3, 1, 2).contiguous()
            grouped_points_stu = index_feature(student_feature, cluster_idx).permute(0, 3, 1, 2).contiguous()
            
            # B x num_points x channel -> B x channels x num_points x 1 -> B x num_points x channels x kneighbours
            new_query_feature_tea = query_feature_tea.transpose(2, 1).unsqueeze(-1).repeat(
                                        1, 1, 1, kneighbours).contiguous()
            new_query_feature_stu = query_feature_stu.transpose(2, 1).unsqueeze(-1).repeat(
                                        1, 1, 1, kneighbours).contiguous()

            # KNN graph center feature fusion
            # 1 x channel x num_points x kneighbours -> 1 x (2xchannel) x num_points x kneighbours 
            if fuse_mode == 'idendity':
                # XXX We can concat the center feature
                grouped_points_tea = torch.cat([grouped_points_tea, new_query_feature_tea], dim=1)
                grouped_points_stu = torch.cat([grouped_points_stu, new_query_feature_stu], dim=1)
            elif fuse_mode == 'sub':
                # XXX Or, we can also use residual graph feature
                grouped_points_tea = torch.cat([new_query_feature_tea-grouped_points_tea, new_query_feature_tea], dim=1)
                grouped_points_stu = torch.cat([new_query_feature_stu-grouped_points_stu, new_query_feature_stu], dim=1)

            grouped_points_tea = self.teacher_gcn_adapt(grouped_points_tea)
            grouped_points_stu = self.student_gcn_adapt(grouped_points_stu)
            
            if pool_mode != 'none':
                # global feature extraction via local feature aggreation
                # B C N K -> B C N
                grouped_points_tea = pool_features(grouped_points_tea, pool_mode)
                grouped_points_stu = pool_features(grouped_points_stu, pool_mode)

            # batch x channel' x num_points x (kneighbours) -> batch x num_points x channel' x (kneighbours)
            grouped_points_tea = grouped_points_tea.transpose(2, 1)
            grouped_points_stu = grouped_points_stu.transpose(2, 1)

            # calculate teacher and student knowledge gap
            dist = torch.FloatTensor([0.0]).cuda()
            if reweight is False:
                dist += torch.dist(grouped_points_tea, grouped_points_stu) * 5e-4
            else:
                reweight = index_feature(point_saliency.unsqueeze(-1), query_point_idx)
                reweight = F.softmax(reweight.float() / 1.e-3, dim=-1) * reweight.shape[0]
                # reweight: B x num_point x 1 -> B x num_point x 1 x 1
                if grouped_points_tea.ndim == 4:
                    reweight = reweight.unsqueeze(-1)
                _dist = F.mse_loss(grouped_points_tea, grouped_points_stu, reduce=False)
                dist += (_dist.mean() * 2)
            return dist

        losses = dict()
        points_cat = torch.stack(points)
        if teacher_info is not None:
            x, student_info = self.extract_feat(points_cat, return_more_info=True)
        else:
            x = self.extract_feat(points_cat)

        # features for rcnn
        backbone_feats = x['fp_features'].clone()
        backbone_xyz = x['fp_xyz'].clone()
        rcnn_feats = {'features': backbone_feats, 'points': backbone_xyz}

        bbox_preds, cls_preds = self.rpn_head(x)

        rpn_loss = self.rpn_head.loss(
            bbox_preds=bbox_preds,
            cls_preds=cls_preds,
            points=points,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            img_metas=img_metas)
        losses.update(rpn_loss)

        bbox_list = self.rpn_head.get_bboxes(points_cat, bbox_preds, cls_preds,
                                             img_metas)
        proposal_list = [
            dict(
                boxes_3d=bboxes,
                scores_3d=scores,
                labels_3d=labels,
                cls_preds=preds_cls)
            for bboxes, scores, labels, preds_cls in bbox_list
        ]
        rcnn_feats.update({'points_cls_preds': cls_preds})

        roi_losses = self.roi_head.forward_train(rcnn_feats, img_metas,
                                                 proposal_list, gt_bboxes_3d,
                                                 gt_labels_3d)
        losses.update(roi_losses)

        # PointDistiller Knowledge Distillation
        if teacher_info is not None:
            kd_loss = torch.FloatTensor([0.0]).cuda()

            # PointRCNN KD
            kd_loss += get_knn_point_backbone(
                teacher_feature=teacher_info['fp_features'], student_feature=student_info['fp_features'],
                point_saliency=None, num_points=6000, kneighbours=128,
                reweight=True, sample_mode='top', pool_mode='max', fuse_mode='idendity'
            )

            losses.update({'kd_loss': kd_loss})

        return losses

    def simple_test(self, points, img_metas, imgs=None, rescale=False):
        """Forward of testing.

        Args:
            points (list[torch.Tensor]): Points of each sample.
            img_metas (list[dict]): Image metas.
            imgs (list[torch.Tensor], optional): Images of each sample.
                Defaults to None.
            rescale (bool, optional): Whether to rescale results.
                Defaults to False.

        Returns:
            list: Predicted 3d boxes.
        """
        points_cat = torch.stack(points)

        x = self.extract_feat(points_cat)
        # features for rcnn
        backbone_feats = x['fp_features'].clone()
        backbone_xyz = x['fp_xyz'].clone()
        rcnn_feats = {'features': backbone_feats, 'points': backbone_xyz}
        bbox_preds, cls_preds = self.rpn_head(x)
        rcnn_feats.update({'points_cls_preds': cls_preds})

        bbox_list = self.rpn_head.get_bboxes(
            points_cat, bbox_preds, cls_preds, img_metas, rescale=rescale)

        proposal_list = [
            dict(
                boxes_3d=bboxes,
                scores_3d=scores,
                labels_3d=labels,
                cls_preds=preds_cls)
            for bboxes, scores, labels, preds_cls in bbox_list
        ]
        bbox_results = self.roi_head.simple_test(rcnn_feats, img_metas,
                                                 proposal_list)

        return bbox_results
