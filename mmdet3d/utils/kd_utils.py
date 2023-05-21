# --------------------------------------------------------
# PointDistiller
# Copyright (c) 2022-2023 Runpei Dong & Linfeng Zhang
# Licensed under The MIT License [see LICENSE for details]
# Written by Runpei Dong & Linfeng Zhang
# --------------------------------------------------------

import torch
import torch.nn.functional as F

from mmcv.cnn import ConvModule

def pool_features(features, pool_mode='max'):
    """Perform feature aggregation using adaptive pooling operation.

    Args:
        features (torch.Tensor): (B, C, N, K)
            Features of locally grouped points/pillars before pooling.
        pool_mode (str, optional): 
            Type of pooling method. Defaults to 'max'.
    Returns:
        torch.Tensor: (B, C, N)
            Pooled features aggregating local information.
    """
    if pool_mode == 'max':
        # (B, C, N, 1)
        new_features = F.max_pool2d(
            features, kernel_size=[1, features.size(3)])
    elif pool_mode == 'avg':
        # (B, C, N, 1)
        new_features = F.avg_pool2d(
            features, kernel_size=[1, features.size(3)])
    else:
        raise NotImplementedError

    return new_features.squeeze(-1).contiguous()

def pool_features1d(features, pool_mode='max'):
    """Perform feature aggregation using adaptive pooling operation.

    Args:
        features (torch.Tensor): (N, C, K)
            Features of locally grouped points/pillars before pooling.
        pool_mode (str, optional): 
            Type of pooling method. Defaults to 'max'.
    Returns:
        torch.Tensor: (N, C)
            Pooled features aggregating local information.
    """
    if pool_mode == 'max':
        # (N, C, 1)
        new_features = F.adaptive_max_pool1d(features, 1)
    elif pool_mode == 'avg':
        # (N, C, 1)
        new_features = F.adaptive_avg_pool1d(features, 1)
    else:
        raise NotImplementedError

    return new_features
    # return new_features.squeeze(-1).contiguous()

def index_feature(feature, idx):
    """
    Input:
        feature: input points/voxels feature, [B, N, C]
        idx: group sample index, [B, S]
    Return:
        new_feature:, indexed points/voxels data, [B, S, C]
    """
    device = feature.device
    B = feature.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_feature = feature[batch_indices, idx, :]
    return new_feature

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points/voxels, [B, N, C]
        dst: target points/voxels, [B, M, C]
    Output:
        dist: per-point/per-voxel square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def query_ball_feature(radius, ngroup, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        ngroup: max group number in local region
        xyz: all points/voxels, [B, N, 3]
        new_xyz: query points/voxels, [B, S, 3]
    Return:
        group_idx: grouped points/voxels index, [B, S, ngroup]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :ngroup]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, ngroup])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def knn_feature(ngroup, xyz, new_xyz):
    """
    Input:
        ngroup: max group number in local region
        xyz: all points/voxels features, [B, N, C]
        new_xyz: query points/voxels features, [B, S, C]
    Return:
        group_idx: grouped points/voxels index, [B, S, ngroup]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, ngroup, dim=-1, largest=False, sorted=False)
    return group_idx