# Copyright (c) OpenMMLab. All rights reserved.
import torch
import pdb
from mmcv.cnn import build_norm_layer
from mmcv.runner import force_fp32
from torch import nn

from mmdet3d.ops import DynamicScatter
from .. import builder
from ..builder import VOXEL_ENCODERS
from .utils import VFELayer, get_paddings_indicator

#chgd
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
from torch.nn.init import normal_
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence, BaseTransformerLayer, TransformerLayerSequence
from mmcv.cnn import constant_init
import math
from mmcv.cnn.bricks.registry import (TRANSFORMER_LAYER,ATTENTION, FEEDFORWARD_NETWORK,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.utils import build_from_cfg
import copy
from mmcv import ConfigDict
import warnings
from mmcv.utils import ext_loader
from torch.autograd.function import Function, once_differentiable
from mmcv.cnn import constant_init, xavier_init
from mmcv import deprecated_api_warning
from mmdet.models.utils.transformer import inverse_sigmoid


@VOXEL_ENCODERS.register_module()
class HardSimpleVFE(nn.Module):
    """Simple voxel feature encoder used in SECOND.

    It simply averages the values of points in a voxel.

    Args:
        num_features (int): Number of features to use. Default: 4.
    """

    def __init__(self, num_features=4):
        super(HardSimpleVFE, self).__init__()
        self.num_features = num_features
        self.fp16_enabled = False

    @force_fp32(out_fp16=True)
    def forward(self, features, num_points, coors):
        """Forward function.

        Args:
            features (torch.Tensor): Point features in shape
                (N, M, 3(4)). N is the number of voxels and M is the maximum
                number of points inside a single voxel.
            num_points (torch.Tensor): Number of points in each voxel,
                 shape (N, ).
            coors (torch.Tensor): Coordinates of voxels.

        Returns:
            torch.Tensor: Mean of points inside each voxel in shape (N, 3(4))
        """
        points_mean = features[:, :, :self.num_features].sum(
            dim=1, keepdim=False) / num_points.type_as(features).view(-1, 1)
        return points_mean.contiguous()


@VOXEL_ENCODERS.register_module()
class DynamicSimpleVFE(nn.Module):
    """Simple dynamic voxel feature encoder used in DV-SECOND.

    It simply averages the values of points in a voxel.
    But the number of points in a voxel is dynamic and varies.

    Args:
        voxel_size (tupe[float]): Size of a single voxel
        point_cloud_range (tuple[float]): Range of the point cloud and voxels
    """

    def __init__(self,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1)):
        super(DynamicSimpleVFE, self).__init__()
        self.scatter = DynamicScatter(voxel_size, point_cloud_range, True)
        self.fp16_enabled = False

    @torch.no_grad()
    @force_fp32(out_fp16=True)
    def forward(self, features, coors):
        """Forward function.

        Args:
            features (torch.Tensor): Point features in shape
                (N, 3(4)). N is the number of points.
            coors (torch.Tensor): Coordinates of voxels.

        Returns:
            torch.Tensor: Mean of points inside each voxel in shape (M, 3(4)).
                M is the number of voxels.
        """
        # This function is used from the start of the voxelnet
        # num_points: [concated_num_points]
        features, features_coors = self.scatter(features, coors)
        return features, features_coors


@VOXEL_ENCODERS.register_module()
class DynamicVFE(nn.Module):
    """Dynamic Voxel feature encoder used in DV-SECOND.

    It encodes features of voxels and their points. It could also fuse
    image feature into voxel features in a point-wise manner.
    The number of points inside the voxel varies.

    Args:
        in_channels (int): Input channels of VFE. Defaults to 4.
        feat_channels (list(int)): Channels of features in VFE.
        with_distance (bool): Whether to use the L2 distance of points to the
            origin point. Default False.
        with_cluster_center (bool): Whether to use the distance to cluster
            center of points inside a voxel. Default to False.
        with_voxel_center (bool): Whether to use the distance to center of
            voxel for each points inside a voxel. Default to False.
        voxel_size (tuple[float]): Size of a single voxel. Default to
            (0.2, 0.2, 4).
        point_cloud_range (tuple[float]): The range of points or voxels.
            Default to (0, -40, -3, 70.4, 40, 1).
        norm_cfg (dict): Config dict of normalization layers.
        mode (str): The mode when pooling features of points inside a voxel.
            Available options include 'max' and 'avg'. Default to 'max'.
        fusion_layer (dict | None): The config dict of fusion layer used in
            multi-modal detectors. Default to None.
        return_point_feats (bool): Whether to return the features of each
            points. Default to False.
    """

    def __init__(self,
                 in_channels=4,
                 feat_channels=[],
                 with_distance=False,
                 with_cluster_center=False,
                 with_voxel_center=False,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode='max',
                 fusion_layer=None,
                 return_point_feats=False):
        super(DynamicVFE, self).__init__()
        assert mode in ['avg', 'max']
        assert len(feat_channels) > 0
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 3
        if with_distance:
            in_channels += 1
        self.in_channels = in_channels
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self.return_point_feats = return_point_feats
        self.fp16_enabled = False

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.point_cloud_range = point_cloud_range
        self.scatter = DynamicScatter(voxel_size, point_cloud_range, True)

        feat_channels = [self.in_channels] + list(feat_channels)
        vfe_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i > 0:
                in_filters *= 2
            norm_name, norm_layer = build_norm_layer(norm_cfg, out_filters)
            vfe_layers.append(
                nn.Sequential(
                    nn.Linear(in_filters, out_filters, bias=False), norm_layer,
                    nn.ReLU(inplace=True)))
        self.vfe_layers = nn.ModuleList(vfe_layers)
        self.num_vfe = len(vfe_layers)
        self.vfe_scatter = DynamicScatter(voxel_size, point_cloud_range,
                                          (mode != 'max'))
        self.cluster_scatter = DynamicScatter(
            voxel_size, point_cloud_range, average_points=True)
        self.fusion_layer = None
        if fusion_layer is not None:
            self.fusion_layer = builder.build_fusion_layer(fusion_layer)

    def map_voxel_center_to_point(self, pts_coors, voxel_mean, voxel_coors):
        """Map voxel features to its corresponding points.

        Args:
            pts_coors (torch.Tensor): Voxel coordinate of each point.
            voxel_mean (torch.Tensor): Voxel features to be mapped.
            voxel_coors (torch.Tensor): Coordinates of valid voxels

        Returns:
            torch.Tensor: Features or centers of each point.
        """
        # Step 1: scatter voxel into canvas
        # Calculate necessary things for canvas creation
        canvas_z = int(
            (self.point_cloud_range[5] - self.point_cloud_range[2]) / self.vz)
        canvas_y = int(
            (self.point_cloud_range[4] - self.point_cloud_range[1]) / self.vy)
        canvas_x = int(
            (self.point_cloud_range[3] - self.point_cloud_range[0]) / self.vx)
        # canvas_channel = voxel_mean.size(1)
        batch_size = pts_coors[-1, 0] + 1
        canvas_len = canvas_z * canvas_y * canvas_x * batch_size
        # Create the canvas for this sample
        canvas = voxel_mean.new_zeros(canvas_len, dtype=torch.long)
        # Only include non-empty pillars
        indices = (
            voxel_coors[:, 0] * canvas_z * canvas_y * canvas_x +
            voxel_coors[:, 1] * canvas_y * canvas_x +
            voxel_coors[:, 2] * canvas_x + voxel_coors[:, 3])
        # Scatter the blob back to the canvas
        canvas[indices.long()] = torch.arange(
            start=0, end=voxel_mean.size(0), device=voxel_mean.device)

        # Step 2: get voxel mean for each point
        voxel_index = (
            pts_coors[:, 0] * canvas_z * canvas_y * canvas_x +
            pts_coors[:, 1] * canvas_y * canvas_x +
            pts_coors[:, 2] * canvas_x + pts_coors[:, 3])
        voxel_inds = canvas[voxel_index.long()]
        center_per_point = voxel_mean[voxel_inds, ...]
        return center_per_point

    @force_fp32(out_fp16=True)
    def forward(self,
                features,
                coors,
                points=None,
                img_feats=None,
                img_metas=None):
        """Forward functions.

        Args:
            features (torch.Tensor): Features of voxels, shape is NxC.
            coors (torch.Tensor): Coordinates of voxels, shape is  Nx(1+NDim).
            points (list[torch.Tensor], optional): Raw points used to guide the
                multi-modality fusion. Defaults to None.
            img_feats (list[torch.Tensor], optional): Image fetures used for
                multi-modality fusion. Defaults to None.
            img_metas (dict, optional): [description]. Defaults to None.

        Returns:
            tuple: If `return_point_feats` is False, returns voxel features and
                its coordinates. If `return_point_feats` is True, returns
                feature of each points inside voxels.
        """
        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            voxel_mean, mean_coors = self.cluster_scatter(features, coors)
            points_mean = self.map_voxel_center_to_point(
                coors, voxel_mean, mean_coors)
            # TODO: maybe also do cluster for reflectivity
            f_cluster = features[:, :3] - points_mean[:, :3]
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        if self._with_voxel_center:
            f_center = features.new_zeros(size=(features.size(0), 3))
            f_center[:, 0] = features[:, 0] - (
                coors[:, 3].type_as(features) * self.vx + self.x_offset)
            f_center[:, 1] = features[:, 1] - (
                coors[:, 2].type_as(features) * self.vy + self.y_offset)
            f_center[:, 2] = features[:, 2] - (
                coors[:, 1].type_as(features) * self.vz + self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :3], 2, 1, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)
        for i, vfe in enumerate(self.vfe_layers):
            point_feats = vfe(features)
            if (i == len(self.vfe_layers) - 1 and self.fusion_layer is not None
                    and img_feats is not None):
                point_feats = self.fusion_layer(img_feats, points, point_feats,
                                                img_metas)
            voxel_feats, voxel_coors = self.vfe_scatter(point_feats, coors)
            if i != len(self.vfe_layers) - 1:
                # need to concat voxel feats if it is not the last vfe
                feat_per_point = self.map_voxel_center_to_point(
                    coors, voxel_feats, voxel_coors)
                features = torch.cat([point_feats, feat_per_point], dim=1)

        if self.return_point_feats:
            return point_feats
        return voxel_feats, voxel_coors


@VOXEL_ENCODERS.register_module()
class HardVFE(nn.Module):
    """Voxel feature encoder used in DV-SECOND.

    It encodes features of voxels and their points. It could also fuse
    image feature into voxel features in a point-wise manner.

    Args:
        in_channels (int): Input channels of VFE. Defaults to 4.
        feat_channels (list(int)): Channels of features in VFE.
        with_distance (bool): Whether to use the L2 distance of points to the
            origin point. Default False.
        with_cluster_center (bool): Whether to use the distance to cluster
            center of points inside a voxel. Default to False.
        with_voxel_center (bool): Whether to use the distance to center of
            voxel for each points inside a voxel. Default to False.
        voxel_size (tuple[float]): Size of a single voxel. Default to
            (0.2, 0.2, 4).
        point_cloud_range (tuple[float]): The range of points or voxels.
            Default to (0, -40, -3, 70.4, 40, 1).
        norm_cfg (dict): Config dict of normalization layers.
        mode (str): The mode when pooling features of points inside a voxel.
            Available options include 'max' and 'avg'. Default to 'max'.
        fusion_layer (dict | None): The config dict of fusion layer used in
            multi-modal detectors. Default to None.
        return_point_feats (bool): Whether to return the features of each
            points. Default to False.
    """

    def __init__(self,
                 in_channels=4,
                 feat_channels=[],
                 with_distance=False,
                 with_cluster_center=False,
                 with_voxel_center=False,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode='max',
                 fusion_layer=None,
                 return_point_feats=False):
        super(HardVFE, self).__init__()
        assert len(feat_channels) > 0
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 3
        if with_distance:
            in_channels += 1
        self.in_channels = in_channels
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self.return_point_feats = return_point_feats
        self.fp16_enabled = False

        # Need pillar (voxel) size and x/y offset to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.point_cloud_range = point_cloud_range
        self.scatter = DynamicScatter(voxel_size, point_cloud_range, True)

        feat_channels = [self.in_channels] + list(feat_channels)
        vfe_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i > 0:
                in_filters *= 2
            # TODO: pass norm_cfg to VFE
            # norm_name, norm_layer = build_norm_layer(norm_cfg, out_filters)
            if i == (len(feat_channels) - 2):
                cat_max = False
                max_out = True
                if fusion_layer:
                    max_out = False
            else:
                max_out = True
                cat_max = True
            vfe_layers.append(
                VFELayer(
                    in_filters,
                    out_filters,
                    norm_cfg=norm_cfg,
                    max_out=max_out,
                    cat_max=cat_max))
            self.vfe_layers = nn.ModuleList(vfe_layers)
        self.num_vfe = len(vfe_layers)

        self.fusion_layer = None
        if fusion_layer is not None:
            self.fusion_layer = builder.build_fusion_layer(fusion_layer)

    @force_fp32(out_fp16=True)
    def forward(self,
                features,
                num_points,
                coors,
                img_feats=None,
                img_metas=None):
        """Forward functions.

        Args:
            features (torch.Tensor): Features of voxels, shape is MxNxC.
            num_points (torch.Tensor): Number of points in each voxel.
            coors (torch.Tensor): Coordinates of voxels, shape is Mx(1+NDim).
            img_feats (list[torch.Tensor], optional): Image fetures used for
                multi-modality fusion. Defaults to None.
            img_metas (dict, optional): [description]. Defaults to None.

        Returns:
            tuple: If `return_point_feats` is False, returns voxel features and
                its coordinates. If `return_point_feats` is True, returns
                feature of each points inside voxels.
        """
        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            points_mean = (
                features[:, :, :3].sum(dim=1, keepdim=True) /
                num_points.type_as(features).view(-1, 1, 1))
            # TODO: maybe also do cluster for reflectivity
            f_cluster = features[:, :, :3] - points_mean
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        if self._with_voxel_center:
            f_center = features.new_zeros(
                size=(features.size(0), features.size(1), 3))
            f_center[:, :, 0] = features[:, :, 0] - (
                coors[:, 3].type_as(features).unsqueeze(1) * self.vx +
                self.x_offset)
            f_center[:, :, 1] = features[:, :, 1] - (
                coors[:, 2].type_as(features).unsqueeze(1) * self.vy +
                self.y_offset)
            f_center[:, :, 2] = features[:, :, 2] - (
                coors[:, 1].type_as(features).unsqueeze(1) * self.vz +
                self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        voxel_feats = torch.cat(features_ls, dim=-1)
        # The feature decorations were calculated without regard to whether
        # pillar was empty.
        # Need to ensure that empty voxels remain set to zeros.
        voxel_count = voxel_feats.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        voxel_feats *= mask.unsqueeze(-1).type_as(voxel_feats)

        for i, vfe in enumerate(self.vfe_layers):
            voxel_feats = vfe(voxel_feats)

        if (self.fusion_layer is not None and img_feats is not None):
            voxel_feats = self.fusion_with_mask(features, mask, voxel_feats,
                                                coors, img_feats, img_metas)

        return voxel_feats

    def fusion_with_mask(self, features, mask, voxel_feats, coors, img_feats,
                         img_metas):
        """Fuse image and point features with mask.

        Args:
            features (torch.Tensor): Features of voxel, usually it is the
                values of points in voxels.
            mask (torch.Tensor): Mask indicates valid features in each voxel.
            voxel_feats (torch.Tensor): Features of voxels.
            coors (torch.Tensor): Coordinates of each single voxel.
            img_feats (list[torch.Tensor]): Multi-scale feature maps of image.
            img_metas (list(dict)): Meta information of image and points.

        Returns:
            torch.Tensor: Fused features of each voxel.
        """
        # the features is consist of a batch of points
        batch_size = coors[-1, 0] + 1
        points = []
        for i in range(batch_size):
            single_mask = (coors[:, 0] == i)
            points.append(features[single_mask][mask[single_mask]])

        point_feats = voxel_feats[mask]
        point_feats = self.fusion_layer(img_feats, points, point_feats,
                                        img_metas)

        voxel_canvas = voxel_feats.new_zeros(
            size=(voxel_feats.size(0), voxel_feats.size(1),
                  point_feats.size(-1)))
        voxel_canvas[mask] = point_feats
        out = torch.max(voxel_canvas, dim=1)[0]

        return out


#chgd
@VOXEL_ENCODERS.register_module()
class DeformablePrevFreezeTemporalv2(nn.Module): #decoder
    def __init__(self, embed_dims,
                 num_feature_levels,
                    decoder=None,
                    positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=32,
                     normalize=True
                     ),
                ):
        super(DeformablePrevFreezeTemporalv2, self).__init__()
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.embed_dims = embed_dims
        self.decoder = build_transformer_layer_sequence(decoder)
        self.num_feature_levels = num_feature_levels
        self.reference_layer_ = nn.Linear(self.embed_dims,2)
        self.reference_layer__ = nn.Linear(self.embed_dims,2)
        self.init_layers()
        #pdb.set_trace()

    def init_layers(self):
        """Initialize layers of the DeformableDetrTransformer."""
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        
        ##todo0409
        # constant_init(self.sampling_offsets, 0.)
        # thetas = torch.arange(
        #     self.num_heads,
        #     dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        # grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        # grid_init = (grid_init /
        #              grid_init.abs().max(-1, keepdim=True)[0]).view(
        #                  self.num_heads, 1, 1,
        #                  2).repeat(1, self.num_levels, self.num_points, 1)
        # for i in range(self.num_points):
        #     grid_init[:, :, i, :] *= i + 1

        # self.sampling_offsets.bias.data = grid_init.view(-1)
        ##
        

        
    # @staticmethod
    # def get_reference_points(spatial_shapes, valid_ratios, device):
    #     """Get the reference points used in decoder.

    #     Args:
    #         spatial_shapes (Tensor): The shape of all
    #             feature maps, has shape (num_level, 2).
    #         valid_ratios (Tensor): The radios of valid
    #             points on the feature map, has shape
    #             (bs, num_levels, 2)
    #         device (obj:`device`): The device where
    #             reference_points should be.

    #     Returns:
    #         Tensor: reference points used in decoder, has \
    #             shape (bs, num_keys, num_levels, 2).
    #     """
    #     reference_points_list = []
    #     for lvl, (H, W) in enumerate(spatial_shapes):
    #         #  TODO  check this 0.5
    #         ref_y, ref_x = torch.meshgrid(
    #             torch.linspace(
    #                 0.5, H - 0.5, H, dtype=torch.float32, device=device),
    #             torch.linspace(
    #                 0.5, W - 0.5, W, dtype=torch.float32, device=device))
    #         ref_y = ref_y.reshape(-1)[None] / (
    #             valid_ratios[:, None, lvl, 1] * H)
    #         ref_x = ref_x.reshape(-1)[None] / (
    #             valid_ratios[:, None, lvl, 0] * W)
    #         ref = torch.stack((ref_x, ref_y), -1)
    #         reference_points_list.append(ref)
    #     reference_points = torch.cat(reference_points_list, 1)
    #     reference_points = reference_points[:, :, None] * valid_ratios[:, None]
    #     return reference_points
    
    def get_reference_points(self, query, spatial_shapes, valid_ratios, device):
        """Get the reference points used in decoder.

        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            device (obj:`device`): The device where
                reference_points should be.

        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        #pdb.set_trace()
        reference_points_list = []
        flatten_feature_shape = (spatial_shapes[0][0]*spatial_shapes[0][1]).item()
        #query = query[:,:flatten_feature_shape]  # 수정필요
        for lvl, (H, W) in enumerate(spatial_shapes):
            if lvl == 0:
                #  TODO  check this 0.5
                ref_y, ref_x = torch.meshgrid(
                    torch.linspace(
                        0.5, H - 0.5, H, dtype=torch.float32, device=device),
                    torch.linspace(
                        0.5, W - 0.5, W, dtype=torch.float32, device=device))
                ref_y = ref_y.reshape(-1)[None] / (
                    valid_ratios[:, None, lvl, 1] * H)
                ref_x = ref_x.reshape(-1)[None] / (
                    valid_ratios[:, None, lvl, 0] * W)
                ref = torch.stack((ref_x, ref_y), -1)
                reference_points_list.append(ref)
            elif lvl == 1:
                ref = self.reference_layer_(query[:,flatten_feature_shape:flatten_feature_shape*2])
                reference_points_list.append(ref)
            else:
                ref = self.reference_layer__(query[:,flatten_feature_shape*2:])
                reference_points_list.append(ref)
        #reference_points = torch.cat(reference_points_list, 1)
        reference_points = torch.stack(reference_points_list, 2)
        #reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def init_weights(self):  ### 어디서 사용되는지
        normal_(self.level_embeds)

    def get_valid_ratio(self, mask):
        """Get the valid radios of feature maps of all  level."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
        
    def forward(self, features): # B T C H W
        # original feat : list of 5 element, in list B T C H W
        '''
        cur_image_feature = features[:,0:1]  #list of 5 element, in list B T C H W
        prev_image_feature = features[:,1:] # [8, 2, 64, 128, 128]
        feats = []
        for i in range(1):
            feats.append(torch.cat([cur_image_feature[i], prev_image_feature[i]], axis=1))
        '''
        feats = [features]
        temp_dim = self.num_feature_levels # 3 -> 1
        mlvl_dim = 1 # 5 -> 1
        #mlvl_feats = feats
        
        # To extract each three frames
        mlvl_feats = []
        for level in range(mlvl_dim):
            temp_feats = []
            for temp in range(temp_dim):
                temp_feats.append(feats[level][:,temp,...])
            mlvl_feats.append(temp_feats)
        
        # To create positional encodings of each channels of each scales
        encoded_mlvl_feats = []
        for level_ in range(mlvl_dim): # 1
            temp_feats_ = mlvl_feats[level_]
            batch_size = mlvl_feats[0][0].size(0)
            temp_masks = []
            mlvl_positional_encodings = []
            input_img_h = 900
            input_img_w = 1600
            img_masks = temp_feats_[0].new_ones(
                (batch_size, input_img_h, input_img_w)) # (1, 900, 1600)
            for feat_idx in range(len(temp_feats_)): # 3
            #for feat in temp_feats_:
                if feat_idx != 0: # outputs same size, only t-1, t-2 ?
                    temp_temp_feats_ = temp_feats_[feat_idx] # (8, 64, 128, 128)
                    temp_feats_[feat_idx] = F.interpolate(temp_feats_[feat_idx],(int(temp_feats_[feat_idx].shape[-2]),int(temp_feats_[feat_idx].shape[-1]))).contiguous()
                    #print(torch.eq(temp_temp_feats_, temp_feats_[feat_idx]))
                    
                # temp_masks[0] = (1, 116, 200)
                temp_masks.append(
                    F.interpolate(img_masks[None],
                                size=temp_feats_[feat_idx].shape[-2:]).to(torch.bool).squeeze(0)) # change 1 to True
                                # size = (116, 200)
                mlvl_positional_encodings.append(
                    self.positional_encoding(temp_masks[-1]))


            feat_flatten = []
            mask_flatten = []
            lvl_pos_embed_flatten = []
            spatial_shapes = []
            mlvl_pos_embeds = mlvl_positional_encodings # [8, 256, 128, 128] x3 of list
 
            ## flatten 과정
            for lvl, (feat, mask, pos_embed) in enumerate(
                    zip(temp_feats_, temp_masks, mlvl_pos_embeds)):
                bs, c, h, w = feat.shape # [8, 64, 128, 128]
                spatial_shape = (h, w)
                spatial_shapes.append(spatial_shape)
                feat = feat.flatten(2).transpose(1, 2) # [8, 16384, 64]
                mask = mask.flatten(1) # [8, 16384]
                pos_embed = pos_embed.flatten(2).transpose(1, 2) # [8, 16384, 256]
                lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1) 
                lvl_pos_embed_flatten.append(lvl_pos_embed)
                if lvl == 0:
                    cur_flat_feat_size = feat.shape[1]
                feat_flatten.append(feat) # [1, 3, 23200, 256]
                mask_flatten.append(mask) # (1, 3, 23200)

            feat_flatten = torch.cat(feat_flatten, 1) # [1, 69600, 256]
            mask_flatten = torch.cat(mask_flatten, 1) # [1, 69600]
            lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1) # [1, 69600, 256]
            spatial_shapes = torch.as_tensor(
                spatial_shapes, dtype=torch.long, device=feat_flatten.device) # (3, 2) -> 3x[116, 200]
            level_start_index = torch.cat((spatial_shapes.new_zeros(
                (1, )), spatial_shapes.prod(1).cumsum(0)[:-1])) # (3) -> (0, 23200, 46400)
            valid_ratios = torch.stack(
                [self.get_valid_ratio(m) for m in temp_masks], 1)
            # (1,3,2)
            #[[0., 0.],
            # [0., 0.],
            # [0., 0.]],

            reference_points = \
                self.get_reference_points(feat_flatten, spatial_shapes,  #feat_flatten 추가
                                        valid_ratios,
                                        device=feat.device) # (1, 23200, 3, 2)
                #[[    inf,     inf],
                #[-0.4532, -0.2340],
                #[ 0.6294,  0.4104]]

            feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W*3, bs, embed_dims) = (69600, 1, 256)
            lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(
                1, 0, 2)  # (H*W*3, bs, embed_dims)

            # output, reference points
            inter_states, inter_references = self.decoder(
                query=feat_flatten[:cur_flat_feat_size], #cur_flat_feat_size = 16384, shape:[16384, 8, 64]
                key=None, # ?
                value=feat_flatten, # [49152, 8, 64]
                key_padding_mask=mask_flatten,
                reference_points=reference_points, # [8, 16384, 3, 2]
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index, #[    0, 16384, 32768]
                valid_ratios=valid_ratios)

            
            inter_states = inter_states.permute(1, 0, 2) # (1, 23200, 256)
            bs,c,h,w = temp_feats_[0].shape # (1, 256, 116, 200)
            bs, _, c = inter_states.shape # (1, __, 256)
            target_memory = inter_states[:,:h*w,:] # (1, 23200, 256)
            target_memory = target_memory.view(bs,h,w,c).permute(0,3,1,2).contiguous() # (1, 256, 116, 200)
            encoded_mlvl_feats.append(target_memory)
        #pdb.set_trace()
        return encoded_mlvl_feats # torch.Size([bs, 256, 96, 312]) for five multi-level of h, w


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DeformableDetrTransformerDecoderv2(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.

    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):

        super(DeformableDetrTransformerDecoderv2, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

    def forward(self,
                query,
                *args,
                reference_points=None,
                valid_ratios=None,
                reg_branches=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.

        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.

        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        #pdb.set_trace()
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] * \
                    torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                # reference_points_input = reference_points[:, :, None] * \
                #     valid_ratios[:, None]
                reference_points_input = reference_points
                
            # 7/12 start from here
            pdb.set_trace()
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(
                        reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[
                        ..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points


@TRANSFORMER_LAYER.register_module()
class DetrTransformerDecoderLayerv2(BaseTransformerLayer):
    """Implements decoder layer in DETR transformer.

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(DetrTransformerDecoderLayerv2, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        assert len(operation_order) == 4
        assert set(operation_order) == set(
            ['norm', 'cross_attn', 'ffn'])

def build_transformer_layer(cfg, default_args=None):
    """Builder for transformer layer."""
    return build_from_cfg(cfg, TRANSFORMER_LAYER, default_args)

def build_attention(cfg, default_args=None):
    """Builder for attention."""
    return build_from_cfg(cfg, ATTENTION, default_args)


def build_feedforward_network(cfg, default_args=None):
    """Builder for feed-forward network (FFN)."""
    return build_from_cfg(cfg, FEEDFORWARD_NETWORK, default_args)