_base_ = [
    '../_base_/datasets/kitti-3d-car.py', '../_base_/models/point_rcnn.py',
    '../_base_/default_runtime.py', '../_base_/schedules/cyclic_40e.py'
]

# model settings
model = dict(
    type='PointRCNN',
    backbone=dict(
        type='PointNet2SAMSG',
        in_channels=4,
        num_points=(4096, 1024, 256, 64),
        radii=((0.1, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0)),
        num_samples=((16, 32), (16, 32), (16, 32), (16, 32)),
        # sa_channels=(((16, 16, 32), (32, 32, 64)), ((64, 64, 128), (64, 96, 128)),
        #              ((128, 196, 256), (128, 196, 256)), ((256, 256, 512), (256, 384, 512))),
        sa_channels=(((4, 4, 8), (8, 8, 16)), ((16, 16, 32), (16, 24, 32)),
                     ((32, 49, 64), (32, 49, 64)), ((64, 64, 128), (64, 96, 128))),
        fps_mods=(('D-FPS'), ('D-FPS'), ('D-FPS'), ('D-FPS')),
        fps_sample_range_lists=((-1), (-1), (-1), (-1)),
        aggregation_channels=(None, None, None, None),
        dilated_group=(False, False, False, False),
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.1),
        sa_cfg=dict(
            type='PointSAModuleMSG',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=False)),
    neck=dict(
        type='PointNetFPNeck',
        # fp_channels=((1536, 512, 512), (768, 512, 512), (608, 256, 256),
        #              (257, 128, 128))),
        fp_channels=((384, 128, 128), (192, 128, 128), (152, 64, 64),
                     (65, 32, 32))),
    rpn_head=dict(
        type='PointRPNHead',
        num_classes=3,
        enlarge_width=0.1,
        # pred_layer_cfg=dict(
        #     in_channels=128,
        #     cls_linear_channels=(256, 256),
        #     reg_linear_channels=(256, 256)),
        pred_layer_cfg=dict(
            in_channels=32,
            cls_linear_channels=(64, 64),
            reg_linear_channels=(64, 64)),
        cls_loss=dict(
            type='FocalLoss',
            use_sigmoid=True,
            reduction='sum',
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        bbox_loss=dict(
            type='SmoothL1Loss',
            beta=1.0 / 9.0,
            reduction='sum',
            loss_weight=1.0),
        bbox_coder=dict(
            type='PointXYZWHLRBBoxCoder',
            code_size=8,
            # code_size: (center residual (3), size regression (3),
            #             torch.cos(yaw) (1), torch.sin(yaw) (1)
            use_mean_size=True,
            mean_size=[[3.9, 1.6, 1.56], [0.8, 0.6, 1.73], [1.76, 0.6, 1.73]])),
    roi_head=dict(
        type='PointRCNNRoIHead',
        point_roi_extractor=dict(
            type='Single3DRoIPointExtractor',
            roi_layer=dict(type='RoIPointPool3d', num_sampled_points=512)),
        bbox_head=dict(
            type='PointRCNNBboxHead',
            num_classes=1,
            # pred_layer_cfg=dict(
            #     in_channels=512,
            #     cls_conv_channels=(256, 256),
            #     reg_conv_channels=(256, 256),
            #     bias=True),
            pred_layer_cfg=dict(
                in_channels=128,
                cls_conv_channels=(64, 64),
                reg_conv_channels=(64, 64),
                bias=True),
            in_channels=5,
            # 5 = 3 (xyz) + scores + depth
            # mlp_channels=[128, 128],
            mlp_channels=[32, 32],
            num_points=(128, 32, -1),
            radius=(0.2, 0.4, 100),
            num_samples=(16, 16, 16),
            # sa_channels=((128, 128, 128), (128, 128, 256), (256, 256, 512)),
            sa_channels=((32, 32, 32), (32, 32, 64), (64, 64, 128)),
            with_corner_loss=True),
        depth_normalizer=70.0),
    # model training and testing settings
    train_cfg=dict(
        # KD cfg
        kd_cfg=dict(
            type="PointRCNN",
            compression_ratio=16,
        ),
        pos_distance_thr=10.0,
        rpn=dict(
            nms_cfg=dict(
                use_rotate_nms=True, iou_thr=0.8, nms_pre=9000, nms_post=512),
            score_thr=None),
        rcnn=dict(
            assigner=[
                dict(  # for Car
                    type='MaxIoUAssigner',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.55,
                    min_pos_iou=0.55,
                    ignore_iof_thr=-1,
                    match_low_quality=False),
                dict(  # for Pedestrian
                    type='MaxIoUAssigner',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.55,
                    min_pos_iou=0.55,
                    ignore_iof_thr=-1,
                    match_low_quality=False),
                dict(  # for Cyclist
                    type='MaxIoUAssigner',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.55,
                    min_pos_iou=0.55,
                    ignore_iof_thr=-1,
                    match_low_quality=False)
            ],
            sampler=dict(
                type='IoUNegPiecewiseSampler',
                num=128,
                pos_fraction=0.5,
                neg_piece_fractions=[0.8, 0.2],
                neg_iou_piece_thrs=[0.55, 0.1],
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
                return_iou=True),
            cls_pos_thr=0.7,
            cls_neg_thr=0.25)),
    test_cfg=dict(
        rpn=dict(
            nms_cfg=dict(
                use_rotate_nms=True, iou_thr=0.85, nms_pre=9000, nms_post=512),
            score_thr=None),
        rcnn=dict(use_rotate_nms=True, nms_thr=0.1, score_thr=0.1)))

# dataset settings
dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
class_names = ['Car', 'Pedestrian', 'Cyclist']
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
input_modality = dict(use_lidar=True, use_camera=False)

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=5, Cyclist=5)),
    sample_groups=dict(Car=20, Pedestrian=15, Cyclist=15),
    classes=class_names)

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[1.0, 1.0, 0.5],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.78539816, 0.78539816]),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointSample', num_points=16384, sample_range=40.0),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(type='PointSample', num_points=16384, sample_range=40.0),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(pipeline=train_pipeline, classes=class_names)),
    val=dict(pipeline=test_pipeline, classes=class_names),
    test=dict(pipeline=test_pipeline, classes=class_names))

# optimizer
lr = 0.001  # max learning rate
optimizer = dict(lr=lr, betas=(0.95, 0.85))
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=80)
evaluation = dict(interval=2)
# yapf:disable
log_config = dict(
    interval=30,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
