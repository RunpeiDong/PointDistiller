# --------------------------------------------------------
# PointDistiller
# Copyright (c) 2022-2023 Runpei Dong & Linfeng Zhang
# Licensed under The MIT License [see LICENSE for details]
# Written by OpenMMLab
# Hacked by Runpei Dong & Linfeng Zhang
# --------------------------------------------------------
import torch
import torch.nn.functional as F

from mmcv.runner import force_fp32
from mmdet3d.ops import Voxelization
from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet.models import DETECTORS
from .. import builder
from .single_stage import SingleStage3DDetector

from mmdet3d.utils.kd_utils import *

@DETECTORS.register_module()
class VoxelNet(SingleStage3DDetector):
    r"""`VoxelNet <https://arxiv.org/abs/1711.06396>`_ for 3D detection."""

    def __init__(self,
                 voxel_layer,
                 voxel_encoder,
                 middle_encoder,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(VoxelNet, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            pretrained=pretrained)
        self.voxel_layer = Voxelization(**voxel_layer)
        self.voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
        self.middle_encoder = builder.build_middle_encoder(middle_encoder)

        # KD layers for PointPillars
        # 4x compression (teacher)
        self.teacher_gcn_adapt = ConvModule(in_channels=64*2,
                                            out_channels=32,
                                            kernel_size=(1, 1),
                                            stride=(1, 1),
                                            conv_cfg=dict(type='Conv2d'),
                                            norm_cfg=dict(type='BN2d'),
                                            act_cfg=dict(type='ReLU'),
                                            bias='auto')
        # 4x compression (student)
        self.student_gcn_adapt = ConvModule(in_channels=32*2,
                                            out_channels=32,
                                            kernel_size=(1, 1),
                                            stride=(1, 1),
                                            conv_cfg=dict(type='Conv2d'),
                                            norm_cfg=dict(type='BN2d'),
                                            act_cfg=dict(type='ReLU'),
                                            bias='auto')

        # 16x compression (teacher)
        # self.teacher_gcn_adapt = ConvModule(in_channels=64*2,
        #                                     out_channels=16,
        #                                     kernel_size=(1, 1),
        #                                     stride=(1, 1),
        #                                     conv_cfg=dict(type='Conv2d'),
        #                                     norm_cfg=dict(type='BN2d'),
        #                                     act_cfg=dict(type='ReLU'),
        #                                     bias='auto')

        # 16x compression (student)
        # self.student_gcn_adapt = ConvModule(in_channels=16*2,
        #                                     out_channels=16,
        #                                     kernel_size=(1, 1),
        #                                     stride=(1, 1),
        #                                     conv_cfg=dict(type='Conv2d'),
        #                                     norm_cfg=dict(type='BN2d'),
        #                                     act_cfg=dict(type='ReLU'),
        #                                     bias='auto')
        
        # KD layer configurations (for SECOND)
        if self.kd_cfg is not None:
            import math
            self.student_type = self.kd_cfg['type']
            self.compression_ratio = self.kd_cfg['compression_ratio']
            self.stu_channel_reduction = int(math.sqrt(self.compression_ratio))
            self.kd_num_voxels = self.kd_cfg["num_voxels"]
            self.kd_kneighbours = self.kd_cfg["kneighbours"]
            self.kd_temperature = self.kd_cfg["temperature"]
            assert self.student_type in ['PointPillars', 'SECOND'], "student type not supported"
            assert self.compression_ratio in [4, 16], "compression ratio not supported"
            
            if self.student_type == 'SECOND':
                # KD layers for SECOND
                student_channel = 32
                
                # FPN features
                # self.teacher_gcn_adapt = ConvModule(in_channels=512*2,
                #                                     out_channels=16,
                #                                     kernel_size=(1, 1),
                #                                     stride=(1, 1),
                #                                     conv_cfg=dict(type='Conv2d'),
                #                                     norm_cfg=dict(type='BN2d'),
                #                                     act_cfg=dict(type='ReLU'),
                #                                     bias='auto')
                # self.student_gcn_adapt = ConvModule(in_channels=128*2,
                #                                     out_channels=16,
                #                                     kernel_size=(1, 1),
                #                                     stride=(1, 1),
                #                                     conv_cfg=dict(type='Conv2d'),
                #                                     norm_cfg=dict(type='BN2d'),
                #                                     act_cfg=dict(type='ReLU'),
                #                                     bias='auto')

                # backbone features
                self.teacher_gcn_adapt = torch.nn.ModuleList(
                    [ConvModule(in_channels=_c*2,
                                out_channels=student_channel,
                                kernel_size=(1, 1),
                                stride=(1, 1),
                                conv_cfg=dict(type='Conv2d'),
                                norm_cfg=dict(type='BN2d'),
                                act_cfg=dict(type='ReLU'),
                                bias='auto')
                                for _c in [128, 256]]
                )
                self.student_gcn_adapt = torch.nn.ModuleList(
                    [ConvModule(in_channels=_c*2,
                                out_channels=student_channel,
                                kernel_size=(1, 1),
                                stride=(1, 1),
                                conv_cfg=dict(type='Conv2d'),
                                norm_cfg=dict(type='BN2d'),
                                act_cfg=dict(type='ReLU'),
                                bias='auto') 
                                for _c in [128 // self.stu_channel_reduction, 
                                           256 // self.stu_channel_reduction]]
                )

        # KD layers for SECOND
        # 4x compression (teacher)
        # self.teacher_gcn_adapt = nn.ModuleList(
        #     [ConvModule(in_channels=_c*2,
        #                 out_channels=32,
        #                 kernel_size=(1, 1),
        #                 stride=(1, 1),
        #                 conv_cfg=dict(type='Conv2d'),
        #                 norm_cfg=dict(type='BN2d'),
        #                 act_cfg=dict(type='ReLU'),
        #                 bias='auto')
        #                 for _c in [128, 256]]
        # )
        # 4x compression (student)
        # self.student_gcn_adapt = nn.ModuleList(
        #     [ConvModule(in_channels=_c*2,
        #                 out_channels=32,
        #                 kernel_size=(1, 1),
        #                 stride=(1, 1),
        #                 conv_cfg=dict(type='Conv2d'),
        #                 norm_cfg=dict(type='BN2d'),
        #                 act_cfg=dict(type='ReLU'),
        #                 bias='auto') 
        #                 for _c in [64, 128]]
        # )

        # 16x compression (teacher)
        # self.teacher_gcn_adapt = nn.ModuleList(
        #     [ConvModule(in_channels=_c*2,
        #                 out_channels=16,
        #                 kernel_size=(1, 1),
        #                 stride=(1, 1),
        #                 conv_cfg=dict(type='Conv2d'),
        #                 norm_cfg=dict(type='BN2d'),
        #                 act_cfg=dict(type='ReLU'),
        #                 bias='auto')
        #                 for _c in [128, 256]]
        # )
        # 16x compression (student)
        # self.student_gcn_adapt = nn.ModuleList(
        #     [ConvModule(in_channels=_c*2,
        #                 out_channels=16,
        #                 kernel_size=(1, 1),
        #                 stride=(1, 1),
        #                 conv_cfg=dict(type='Conv2d'),
        #                 norm_cfg=dict(type='BN2d'),
        #                 act_cfg=dict(type='ReLU'),
        #                 bias='auto') 
        #                 for _c in [32, 64]]
        # )



    def get_teacher_info(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      gt_bboxes_ignore=None):
        
        x, info_dict = self.extract_feat(points, img_metas, return_more_info=True)

        info_dict['teacher_feature'] = x
        outs = self.bbox_head(x)
        info_dict['bbox_head_out'] = outs
        return info_dict

    def extract_feat(self, points, img_metas=None, return_more_info=False):
        """Extract features from point clouds."""
        info_dict = {}
        voxels, num_points, coors = self.voxelize(points)
        info_dict['num_points'] = num_points
        
        voxel_features = self.voxel_encoder(voxels, num_points, coors)
        info_dict['voxel_features'] = voxel_features
        batch_size = coors[-1, 0].item() + 1
        x = self.middle_encoder(voxel_features, coors, batch_size)
        info_dict['middle_encoder_features'] = x
        x = self.backbone(x)
        info_dict['backbone_features'] = x
        
        if self.with_neck:
            x = self.neck(x)
            info_dict['fp_features'] = x
        if return_more_info:
            return x, info_dict
        else:
            return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch
    
    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      gt_bboxes_ignore=None, teacher_info=None):
        """Training forward function.

        Args:
            points (list[torch.Tensor]): Point cloud of each sample.
            img_metas (list[dict]): Meta information of each sample
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        def get_voxel_knn(teacher_feature, student_feature, num_voxel_points, 
                          num_voxels=6000, kneighbours=128, reweight=False,
                          sample_mode='top', fuse_mode='idendity'):
            """Perform voxel Local KNN graph knowledge distillation with reweighting strategy.

            Args:
                teacher_feature (torch.Tensor): (nvoxels, C)
                    Teacher features of locally grouped points/pillars before pooling.
                student_feature (torch.Tensor): (nvoxels, C)
                    Student features of locally grouped points/pillars before pooling.
                num_voxel_points (torch.Tensor): (nvoxels, npoints)
                    Number of points in the voxel.
                num_voxels (int):
                    Number of voxels after sampling.
                kneighbours (int):
                    Value of number of knn neighbours.
                reweight (bool, optional):
                    Whether to use reweight to further filter voxel features for KD.
                    Defaults to False.
                sample_mode (str, optional):
                    Type of sampling method. Defaults to 'top'.
                fuse_mode (str, optional): Defaults to 'idendity'.
                    Type of fusing the knn graph centering features.
            Returns:
                torch.Tensor:
                    Feature distance between the student and teacher.
            """
            # query_voxel_idx: num_voxels
            if sample_mode == 'top':
                _, query_voxel_idx = torch.topk(num_voxel_points, num_voxels, dim=-1, largest=True, sorted=False)
            elif sample_mode == 'bottom':
                _, query_voxel_idx = torch.topk(num_voxel_points, num_voxels, dim=-1, largest=False, sorted=False)
            elif sample_mode == 'rand':
                query_voxel_idx = torch.randperm(teacher_feature.shape[0], device=teacher_feature.device)
                query_voxel_idx = query_voxel_idx[int(min(num_voxels, query_voxel_idx.shape[0])):]
            elif sample_mode == 'mixed':
                _, query_voxel_idx_pos = torch.topk(num_voxel_points, num_voxels//2, dim=-1, largest=True, sorted=False)
                _, query_voxel_idx_neg = torch.topk(num_voxel_points, num_voxels//2, dim=-1, largest=False, sorted=False)
                query_voxel_idx = torch.cat([query_voxel_idx_pos, query_voxel_idx_neg])
            else:
                raise NotImplementedError
            # query_feature_xxx: num_voxels x channel
            query_feature_tea, query_feature_stu = teacher_feature[query_voxel_idx], student_feature[query_voxel_idx]
            # cluster_idx: 1 x num_voxels x kneighbours
            # cluster_idx = query_ball_feature(radius, kneighbours, teacher_feature, query_feature_tea) # ball query
            cluster_idx = knn_feature(kneighbours, teacher_feature.unsqueeze(0), query_feature_tea.unsqueeze(0)) # knn
            # grouped_voxels_xxx: 1 x num_voxels x kneighbours x channel -> 1 x channel x num_voxels x kneighbours
            grouped_voxels_tea = index_feature(teacher_feature.unsqueeze(0), cluster_idx).permute(0, 3, 1, 2).contiguous()
            grouped_voxels_stu = index_feature(student_feature.unsqueeze(0), cluster_idx).permute(0, 3, 1, 2).contiguous()
            
            # num_voxels x channel -> 1 x channels x num_voxels x 1 -> 1 x num_voxels x channels x kneighbours
            new_query_feature_tea = query_feature_tea.transpose(1, 0).unsqueeze(0).unsqueeze(-1).repeat(
                                        1, 1, 1, kneighbours).contiguous()
            new_query_feature_stu = query_feature_stu.transpose(1, 0).unsqueeze(0).unsqueeze(-1).repeat(
                                        1, 1, 1, kneighbours).contiguous()

            # KNN graph center feature fusion
            # 1 x channel x num_voxels x kneighbours -> 1 x (2xchannel) x num_voxels x kneighbours 
            if fuse_mode == 'idendity':
                # XXX We can concat the center feature
                grouped_voxels_tea = torch.cat([grouped_voxels_tea, new_query_feature_tea], dim=1)
                grouped_voxels_stu = torch.cat([grouped_voxels_stu, new_query_feature_stu], dim=1)
            elif fuse_mode == 'sub':
                # XXX Or, we can also use residual graph feature
                grouped_voxels_tea = torch.cat([new_query_feature_tea-grouped_voxels_tea, new_query_feature_tea], dim=1)
                grouped_voxels_stu = torch.cat([new_query_feature_stu-grouped_voxels_stu, new_query_feature_stu], dim=1)

            grouped_voxels_tea = self.teacher_gcn_adapt(grouped_voxels_tea)
            grouped_voxels_stu = self.student_gcn_adapt(grouped_voxels_stu)
            
            # 1 x channel' x num_voxels x kneighbours -> num_voxels x channel' x kneighbours
            grouped_voxels_tea = grouped_voxels_tea.squeeze(0).transpose(1, 0)
            grouped_voxels_stu = grouped_voxels_stu.squeeze(0).transpose(1, 0)

            # calculate teacher and student knowledge gap
            dist = torch.FloatTensor([0.0]).cuda()
            if reweight is False:
                dist += torch.dist(grouped_voxels_tea, grouped_voxels_stu) * 5e-4
            else:
                reweight = num_voxel_points[query_voxel_idx]
                reweight = F.softmax(reweight.float() / 1.e-2, dim=-1) * reweight.shape[0]
                reweight = reweight.view(reweight.shape[0], 1, 1)
                _dist = F.mse_loss(grouped_voxels_tea, grouped_voxels_stu, reduce=False) * reweight
                dist += (_dist.mean() * 2)
            return dist
        
        def get_knn_voxel_fp(teacher_feature, student_feature, num_voxel_points=None, 
                             num_voxels=256, kneighbours=128, reweight=False, relation=False,
                             pool_mode='none', sample_mode='top', fuse_mode='idendity'):
            batch_size = teacher_feature.shape[0]
            # B C H W -> B (H*W) C: batch_size x nvoxels x channel_tea
            teacher_feature = teacher_feature.view(batch_size, teacher_feature.shape[1], -1).transpose(2, 1).contiguous()
            student_feature = student_feature.view(batch_size, student_feature.shape[1], -1).transpose(2, 1).contiguous()

            if num_voxel_points is None:
                num_voxel_points = pool_features1d(teacher_feature).squeeze(-1)
                # num_voxel_points = teacher_feature.abs().mean(-1)
            # query_voxel_idx: batch x num_voxels
            if sample_mode == 'top':
                _, query_voxel_idx = torch.topk(num_voxel_points, num_voxels, dim=-1, largest=True, sorted=False)
            elif sample_mode == 'bottom':
                _, query_voxel_idx = torch.topk(num_voxel_points, num_voxels, dim=-1, largest=False, sorted=False)
            elif sample_mode == 'rand':
                query_voxel_idx = torch.randperm(teacher_feature.shape[0], device=teacher_feature.device)
                query_voxel_idx = query_voxel_idx[int(min(num_voxels, query_voxel_idx.shape[0])):]
            elif sample_mode == 'mixed':
                _, query_voxel_idx_pos = torch.topk(num_voxel_points, num_voxels//2, dim=-1, largest=True, sorted=False)
                _, query_voxel_idx_neg = torch.topk(num_voxel_points, num_voxels//2, dim=-1, largest=False, sorted=False)
                query_voxel_idx = torch.cat([query_voxel_idx_pos, query_voxel_idx_neg])
            else:
                raise NotImplementedError
            # query_feature_xxx: B x num_voxels x channel
            query_feature_tea, query_feature_stu = index_feature(teacher_feature, query_voxel_idx), index_feature(student_feature, query_voxel_idx)
            # cluster_idx: B x num_voxels x kneighbours
            # cluster_idx = query_ball_feature(radius, kneighbours, teacher_feature, query_feature_tea) # ball query
            cluster_idx = knn_feature(kneighbours, teacher_feature, query_feature_tea) # knn
            # grouped_voxels_xxx: B x num_voxels x kneighbours x channel -> B x channel x num_voxels x kneighbours
            grouped_voxels_tea = index_feature(teacher_feature, cluster_idx).permute(0, 3, 1, 2).contiguous()
            grouped_voxels_stu = index_feature(student_feature, cluster_idx).permute(0, 3, 1, 2).contiguous()

            # B x num_voxels x channel -> B x channels x num_voxels x 1 -> B x num_voxels x channels x kneighbours
            new_query_feature_tea = query_feature_tea.transpose(2, 1).unsqueeze(-1).repeat(
                                        1, 1, 1, kneighbours).contiguous()
            new_query_feature_stu = query_feature_stu.transpose(2, 1).unsqueeze(-1).repeat(
                                        1, 1, 1, kneighbours).contiguous()

            # KNN graph center feature fusion
            # 1 x channel x num_voxels x kneighbours -> 1 x (2xchannel) x num_voxels x kneighbours 
            if fuse_mode == 'idendity':
                # XXX We can concat the center feature
                grouped_voxels_tea = torch.cat([grouped_voxels_tea, new_query_feature_tea], dim=1)
                grouped_voxels_stu = torch.cat([grouped_voxels_stu, new_query_feature_stu], dim=1)
            elif fuse_mode == 'sub':
                # XXX Or, we can also use residual graph feature
                grouped_voxels_tea = torch.cat([new_query_feature_tea-grouped_voxels_tea, new_query_feature_tea], dim=1)
                grouped_voxels_stu = torch.cat([new_query_feature_stu-grouped_voxels_stu, new_query_feature_stu], dim=1)

            grouped_voxels_tea = self.teacher_gcn_adapt(grouped_voxels_tea)
            grouped_voxels_stu = self.student_gcn_adapt(grouped_voxels_stu)

            if pool_mode != 'none':
                # global feature extraction via local feature aggreation
                # B C N K -> B C N
                grouped_points_tea = pool_features(grouped_points_tea, pool_mode)
                grouped_points_stu = pool_features(grouped_points_stu, pool_mode)

            # batch x channel' x num_voxels x kneighbours -> batch x num_voxels x channel' x kneighbours
            grouped_voxels_tea = grouped_voxels_tea.transpose(2, 1)
            grouped_voxels_stu = grouped_voxels_stu.transpose(2, 1)

            # calculate teacher and student knowledge gap
            dist = torch.FloatTensor([0.0]).cuda()
            if reweight is False:
                dist += torch.dist(grouped_voxels_tea, grouped_voxels_stu) * 5e-4
            else:
                # reweight: B x num_voxel x 1
                reweight = index_feature(num_voxel_points.unsqueeze(-1), query_voxel_idx)
                reweight = F.softmax(reweight.float() / self.kd_temperature, dim=-1) * reweight.shape[0]
                if grouped_voxels_tea.ndim == 4:
                    reweight = reweight.unsqueeze(-1)
                _dist = F.mse_loss(grouped_voxels_tea, grouped_voxels_stu, reduce=False) * reweight
                dist += (_dist.mean() * 2)
            return dist

        def get_knn_voxel_backbone(teacher_feature, student_feature, layer_idx, num_voxel_points=None, 
                                   num_voxels=6000, kneighbours=128, reweight=False,
                                   sample_mode='top', fuse_mode='idendity'):
            """Perform voxel Local KNN graph knowledge distillation with reweighting strategy.

            Args:
                teacher_feature (torch.Tensor): (nvoxels, C)
                    Teacher features of locally grouped points/pillars before pooling.
                student_feature (torch.Tensor): (nvoxels, C)
                    Student features of locally grouped points/pillars before pooling.
                num_voxel_points (torch.Tensor): (nvoxels, npoints)
                    Number of points in the voxel.
                num_voxels (int):
                    Number of voxels after sampling.
                kneighbours (int):
                    Value of number of knn neighbours.
                reweight (bool, optional):
                    Whether to use reweight to further filter voxel features for KD.
                    Defaults to False.
                sample_mode (str, optional):
                    Type of sampling method. Defaults to 'top'.
                fuse_mode (str, optional): Defaults to 'idendity'.
                    Type of fusing the knn graph centering features.
            Returns:
                torch.Tensor:
                    Feature distance between the student and teacher.
            """
            teacher_feature, student_feature = teacher_feature[layer_idx], student_feature[layer_idx]
            batch_size = teacher_feature.shape[0]
            # B C H W -> B (H*W) C: batch_size x nvoxels x channel_tea
            teacher_feature = teacher_feature.view(batch_size, teacher_feature.shape[1], -1).transpose(2, 1).contiguous()
            student_feature = student_feature.view(batch_size, student_feature.shape[1], -1).transpose(2, 1).contiguous()

            if num_voxel_points is None:
                num_voxel_points = pool_features1d(teacher_feature).squeeze(-1)
            # query_voxel_idx: batch x num_voxels
            if sample_mode == 'top':
                _, query_voxel_idx = torch.topk(num_voxel_points, num_voxels, dim=-1, largest=True, sorted=False)
            elif sample_mode == 'bottom':
                _, query_voxel_idx = torch.topk(num_voxel_points, num_voxels, dim=-1, largest=False, sorted=False)
            elif sample_mode == 'rand':
                query_voxel_idx = torch.randperm(teacher_feature.shape[0], device=teacher_feature.device)
                query_voxel_idx = query_voxel_idx[int(min(num_voxels, query_voxel_idx.shape[0])):]
            elif sample_mode == 'mixed':
                _, query_voxel_idx_pos = torch.topk(num_voxel_points, num_voxels//2, dim=-1, largest=True, sorted=False)
                _, query_voxel_idx_neg = torch.topk(num_voxel_points, num_voxels//2, dim=-1, largest=False, sorted=False)
                query_voxel_idx = torch.cat([query_voxel_idx_pos, query_voxel_idx_neg])
            else:
                raise NotImplementedError

            # query_feature_xxx: B x num_voxels x channel
            query_feature_tea, query_feature_stu = index_feature(teacher_feature, query_voxel_idx), index_feature(student_feature, query_voxel_idx)
            # cluster_idx: B x num_voxels x kneighbours
            # cluster_idx = query_ball_feature(radius, kneighbours, teacher_feature, query_feature_tea) # ball query
            cluster_idx = knn_feature(kneighbours, teacher_feature, query_feature_tea) # knn

            # grouped_voxels_xxx: B x num_voxels x kneighbours x channel -> B x channel x num_voxels x kneighbours
            grouped_voxels_tea = index_feature(teacher_feature, cluster_idx).permute(0, 3, 1, 2).contiguous()
            grouped_voxels_stu = index_feature(student_feature, cluster_idx).permute(0, 3, 1, 2).contiguous()
            
            # B x num_voxels x channel -> B x channels x num_voxels x 1 -> B x num_voxels x channels x kneighbours
            new_query_feature_tea = query_feature_tea.transpose(2, 1).unsqueeze(-1).repeat(
                                        1, 1, 1, kneighbours).contiguous()
            new_query_feature_stu = query_feature_stu.transpose(2, 1).unsqueeze(-1).repeat(
                                        1, 1, 1, kneighbours).contiguous()

            # KNN graph center feature fusion
            # 1 x channel x num_voxels x kneighbours -> 1 x (2xchannel) x num_voxels x kneighbours 
            if fuse_mode == 'idendity':
                # XXX We can concat the center feature
                grouped_voxels_tea = torch.cat([grouped_voxels_tea, new_query_feature_tea], dim=1)
                grouped_voxels_stu = torch.cat([grouped_voxels_stu, new_query_feature_stu], dim=1)
            elif fuse_mode == 'sub':
                # XXX Or, we can also use residual graph feature
                grouped_voxels_tea = torch.cat([new_query_feature_tea-grouped_voxels_tea, new_query_feature_tea], dim=1)
                grouped_voxels_stu = torch.cat([new_query_feature_stu-grouped_voxels_stu, new_query_feature_stu], dim=1)

            grouped_voxels_tea = self.teacher_gcn_adapt[layer_idx](grouped_voxels_tea)
            grouped_voxels_stu = self.student_gcn_adapt[layer_idx](grouped_voxels_stu)
            
            # batch x channel' x num_voxels x kneighbours -> batch x num_voxels x channel' x kneighbours
            grouped_voxels_tea = grouped_voxels_tea.transpose(2, 1)
            grouped_voxels_stu = grouped_voxels_stu.transpose(2, 1)

            # calculate teacher and student knowledge gap
            dist = torch.FloatTensor([0.0]).cuda()
            if reweight is False:
                dist += torch.dist(grouped_voxels_tea, grouped_voxels_stu) * 5e-4
            else:
                # reweight: B x num_voxel x 1
                reweight = index_feature(num_voxel_points.unsqueeze(-1), query_voxel_idx)
                reweight = F.softmax(reweight.float() / 1.e-4, dim=-1) * reweight.shape[0]
                if grouped_voxels_tea.ndim == 4:
                    reweight = reweight.unsqueeze(-1)
                _dist = F.mse_loss(grouped_voxels_tea, grouped_voxels_stu, reduce=False) * reweight
                dist += (_dist.mean() * 2)
            return dist

        x, student_info = self.extract_feat(points, img_metas, return_more_info=True)

        outs = self.bbox_head(x)

        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        # PointDistiller Knowledge Distillation
        if teacher_info is not None:
            # initialize kd loss
            kd_loss = torch.FloatTensor([0.0]).cuda()

            # PointPillars KD
            kd_loss += get_voxel_knn(
                teacher_feature=teacher_info['voxel_features'], student_feature=student_info['voxel_features'],
                num_voxel_points=teacher_info['num_points'], num_voxels=6000, kneighbours=128, 
                reweight=True, sample_mode='top', fuse_mode='idendity'
            )
            
            # SECOND KD
            # for _idx in range(2):
            #     kd_loss += get_knn_voxel_backbone(
            #         teacher_feature=teacher_info['backbone_features'], student_feature=student_info['backbone_features'],
            #         layer_idx=_idx, num_voxel_points=None, num_voxels=256, kneighbours=128, 
            #         reweight=True, sample_mode='top', fuse_mode='idendity'
            #     )
            losses['loss_kd'] = kd_loss

        return losses

    def simple_test(self, points, img_metas, imgs=None, rescale=False, attn_vis=False):
        """Test function without augmentaiton."""
        if attn_vis:
            x, info_dict = self.extract_feat(points, img_metas, True)
        else:
            x = self.extract_feat(points, img_metas)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        if attn_vis:
            return bbox_results, info_dict['attn_data']
        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        feats = self.extract_feats(points, img_metas)

        # only support aug_test for one sample
        aug_bboxes = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.bbox_head(x)
            bbox_list = self.bbox_head.get_bboxes(
                *outs, img_meta, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.bbox_head.test_cfg)

        return [merged_bboxes]
