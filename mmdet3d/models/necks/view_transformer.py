# Copyright (c) Phigent Robotics. All rights reserved.

import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F
from torch.autograd import Variable
from mmcv.runner import BaseModule
from ..builder import NECKS
from .convGRU import ConvGRU
from .convGRU_V2 import ConvGRUV2


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.Tensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx


def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])
    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))
    return x, geom_feats


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


@NECKS.register_module()
class ViewTransformerLiftSplatShoot(BaseModule):
    def __init__(self, grid_config=None, data_config=None,
                 numC_input=512, numC_Trans=64, downsample=16,
                 image_view_supervision=False, voxel=False, **kwargs):
        super(ViewTransformerLiftSplatShoot, self).__init__()
        if grid_config is None:
            grid_config = {
                'xbound': [-51.2, 51.2, 0.8],
                'ybound': [-51.2, 51.2, 0.8],
                'zbound': [-10.0, 10.0, 20.0],
                'dbound': [1.0, 60.0, 1.0],}
        self.grid_config = grid_config
        dx, bx, nx = gen_dx_bx(self.grid_config['xbound'],
                               self.grid_config['ybound'],
                               self.grid_config['zbound'],
                               )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        if data_config is None:
            data_config = {'input_size': (256, 704)}
        self.data_config = data_config
        self.downsample = downsample

        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.numC_input = numC_input
        self.numC_Trans = numC_Trans
        self.depthnet = nn.Conv2d(self.numC_input, self.D + self.numC_Trans, kernel_size=1, padding=0)
        self.geom_feats = None
        self.image_view_supervision = image_view_supervision
        self.voxel=voxel

    def get_depth_dist(self, x):
        return x.softmax(dim=1)

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_config['input_size']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.grid_config['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans, offset=None):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        if offset is not None:
            _,D,H,W = offset.shape
            points[:,:,:,:,:,2] = points[:,:,:,:,:,2]+offset.view(B,N,D,H,W)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        if intrins.shape[3]==4: # for KITTI
            shift = intrins[:,:,:3,3]
            points  = points - shift.view(B,N,1,1,1,3,1)
            intrins = intrins[:,:,:3,:3]
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        # points_numpy = points.detach().cpu().numpy()
        return points

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        #Bg, Ng, Dg, Hg, Wg, Cg = geom_feats.shape
        #pdb.set_trace()
        Nprime = B * N * D * H * W
        #Nprime_g = Bg * Ng * Dg * Hg * Wg
        nx = self.nx.to(torch.long)
        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        #geom_feats = geom_feats.view(-1, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                         device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
                + geom_feats[:, 1] * (self.nx[2] * B) \
                + geom_feats[:, 2] * B \
                + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, nx[2], nx[1], nx[0]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1], geom_feats[:, 0]] = x
        if self.voxel:
            return final.sum(2), x, geom_feats
        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def forward(self, input):
        x, rots, trans, intrins, post_rots, post_trans = input
        #pdb.set_trace()
        B, N, C, H, W = x.shape # N = 6
        #pdb.set_trace()
        x = x.view(B * N, C, H, W)
        x = self.depthnet(x)
        depth = self.get_depth_dist(x[:, :self.D])
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        img_feat = x[:, self.D:(self.D + self.numC_Trans)]

        # Lift
        volume = depth.unsqueeze(1) * img_feat.unsqueeze(2)
        volume = volume.view(B, N, self.numC_Trans, self.D, H, W)
        volume = volume.permute(0, 1, 3, 4, 5, 2)

        # Splat
        bev_feat = self.voxel_pooling(geom, volume) # (4, 64, 128, 128)
        #pdb.set_trace()
        if self.image_view_supervision:
            return bev_feat, [x[:, :self.D].view(B,N,self.D,H,W), x[:, self.D:].view(B,N,self.numC_Trans,H,W)]
        return bev_feat


@NECKS.register_module()
class ViewTransformerLiftSplatShootTemporal(BaseModule):
    def __init__(self, grid_config=None, data_config=None,
                 numC_input=512, numC_Trans=64, downsample=16,
                 image_view_supervision=False, voxel=False, **kwargs):
        super(ViewTransformerLiftSplatShootTemporal, self).__init__()
        if grid_config is None:
            grid_config = {
                'xbound': [-51.2, 51.2, 0.8],
                'ybound': [-51.2, 51.2, 0.8],
                'zbound': [-10.0, 10.0, 20.0],
                'dbound': [1.0, 60.0, 1.0],}
        self.grid_config = grid_config
        dx, bx, nx = gen_dx_bx(self.grid_config['xbound'],
                               self.grid_config['ybound'],
                               self.grid_config['zbound'],
                               )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        if data_config is None:
            data_config = {'input_size': (256, 704)}
        self.data_config = data_config
        self.downsample = downsample

        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.numC_input = numC_input
        self.numC_Trans = numC_Trans
        self.depthnet = nn.Conv2d(self.numC_input, self.D + self.numC_Trans, kernel_size=1, padding=0)
        self.geom_feats = None
        self.image_view_supervision = image_view_supervision
        self.voxel=voxel

        # GRU
        '''
        self.dropout = nn.Dropout(0.2)
        self.n_layers = 2
        self.hidden_dim = 64
        self.embed_dim = 32768
        self.out = nn.Linear(self.hidden_dim, self.embed_dim)
        self.gru = nn.GRU(self.embed_dim, self.hidden_dim, num_layers=self.n_layers, batch_first=True)
        '''

        # ConvGRU
        height = width = 128
        channels = 64
        hidden_dim = [64, 64]
        dtype = 'torch.cuda.FloatTensor'
        kernel_size = (3,3) # kernel size for two stacked hidden layer
        num_layers = 2 # number of stacked hidden layer
        '''
        self.convgru = ConvGRU(input_size=(height, width),
                        input_dim=channels,
                        hidden_dim=hidden_dim,
                        kernel_size=kernel_size,
                        num_layers=num_layers,
                        dtype=dtype,
                        batch_first=True,
                        bias = True,
                        return_all_layers = False)
        '''
        self.hidden_dim_ = 64
        self.embed_dim = 32768
        self.out = nn.Linear(self.hidden_dim_, self.embed_dim)
        self.dropout = nn.Dropout(0.2)

        
        # ConvGRU Version2
        input_tensor = torch.zeros(8, 2, 64, 128, 128)
        self.convgruv2 = ConvGRUV2(input_size=(height, width),
                    input_shape = input_tensor.shape,
                    input_dim=channels,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    num_layers=num_layers,
                    dtype=dtype,
                    batch_first=True,
                    bias = True,
                    return_all_layers = False)


    def get_depth_dist(self, x):
        return x.softmax(dim=1)

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_config['input_size']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.grid_config['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans, offset=None):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        if offset is not None:
            _,D,H,W = offset.shape
            points[:,:,:,:,:,2] = points[:,:,:,:,:,2]+offset.view(B,N,D,H,W)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        if intrins.shape[3]==4: # for KITTI
            shift = intrins[:,:,:3,3]
            points  = points - shift.view(B,N,1,1,1,3,1)
            intrins = intrins[:,:,:3,:3]
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        # points_numpy = points.detach().cpu().numpy()
        return points

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        #Bg, Ng, Dg, Hg, Wg, Cg = geom_feats.shape
        #pdb.set_trace()
        Nprime = B * N * D * H * W
        #Nprime_g = Bg * Ng * Dg * Hg * Wg
        nx = self.nx.to(torch.long)
        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        #geom_feats = geom_feats.view(-1, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                         device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
                + geom_feats[:, 1] * (self.nx[2] * B) \
                + geom_feats[:, 2] * B \
                + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, nx[2], nx[1], nx[0]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1], geom_feats[:, 0]] = x
        if self.voxel:
            return final.sum(2), x, geom_feats
        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def forward(self, input):
        x, rots, trans, intrins, post_rots, post_trans, prev_x, prev_rots, prev_trans, prev_intrins, prev_post_rots, prev_post_trans = input
        B, N, C, H, W = x.shape # N = 6
        #pdb.set_trace()
        x = x.view(B * N, C, H, W)
        x = self.depthnet(x)
        depth = self.get_depth_dist(x[:, :self.D])
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        img_feat = x[:, self.D:(self.D + self.numC_Trans)]

        # Lift
        volume = depth.unsqueeze(1) * img_feat.unsqueeze(2)
        volume = volume.view(B, N, self.numC_Trans, self.D, H, W)
        volume = volume.permute(0, 1, 3, 4, 5, 2)

        # Splat
        cur_bev = self.voxel_pooling(geom, volume) # (batch, 64, 128, 128)
        
        # chgd (handle previous frames)
        iters = int(prev_x.shape[1]/6)
        prev_img1 = prev_x[:,:6] # t-1
        prev_img2 = prev_x[:,6:] # t-2
        prev_bevs = torch.zeros([2, B, 64, 128,128])

        #pdb.set_trace()
        for i in range(iters):
            if i == 0:
                x = prev_img1
                rots, trans, intrins, post_rots, post_trans = \
                            prev_rots[:,:6], prev_trans[:,:6], prev_intrins[:,:6], prev_post_rots[:,:6], prev_post_trans[:,:6]
            if i == 1:
                x = prev_img2
                rots, trans, intrins, post_rots, post_trans = \
                            prev_rots[:,6:], prev_trans[:,6:], prev_intrins[:,6:], prev_post_rots[:,6:], prev_post_trans[:,6:]

            #pdb.set_trace()
            B, N, C, H, W = x.shape
            x = x.reshape(B * N, C, H, W)
            x = self.depthnet(x)
            depth = self.get_depth_dist(x[:, :self.D])
            geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
            img_feat = x[:, self.D:(self.D + self.numC_Trans)]

            # Lift
            volume = depth.unsqueeze(1) * img_feat.unsqueeze(2)
            volume = volume.view(B, N, self.numC_Trans, self.D, H, W)
            volume = volume.permute(0, 1, 3, 4, 5, 2)

            # Splat
            bev_feat = self.voxel_pooling(geom, volume) # (batch, 64, 128, 128)

            prev_bevs[:][i] = bev_feat

        #pdb.set_trace()
        # Feed current and prev bev features into GRU
        #prev_bevs = prev_bevs.permute(1, 2, 0, 3, 4)
        #prev_bevs = prev_bevs.reshape(B, 64, -1)
        prev_bevs = prev_bevs.permute(1, 0, 2, 3, 4)
        res, _ = self.convgruv2(prev_bevs.cuda())
        res = self.dropout(res[0])
        #res = self.out(res)

        # Reshape to original shape
        #res = res.reshape(2, 64, B, 128, 128)
        #res = res.permute(0, 2, 1, 3, 4) # B, N, C, H, W

        #pdb.set_trace()
        # Concat with frame at t and max along
        if self.training == False:
            cur_bev = torch.unsqueeze(cur_bev, 1).permute(1,0,2,3,4)
            #res = res.permute(1,0,2,3,4)
        else:
            cur_bev = torch.unsqueeze(cur_bev, 1)
            #res = res.permute(1,0,2,3,4)

        '''
        # sigmoid attention map element-wise product version
        sigmoid = nn.Sigmoid()
        prev_sigmoid = sigmoid(res)
        bev_tmp = torch.mul(cur_bev, prev_sigmoid)
        bev_feat, max_indices = torch.max(bev_tmp, dim=1)
        bev_feat = bev_feat.cuda()
        '''

        # Max pooling version
        #pdb.set_trace()
        tmp_concat = torch.cat([cur_bev.cuda(), res], axis=1)
        bev_feat = torch.max(tmp_concat, dim=1)
        bev_feat = bev_feat[0].cuda()
        

        if self.image_view_supervision:
            return bev_feat, [x[:, :self.D].view(B,N,self.D,H,W), x[:, self.D:].view(B,N,self.numC_Trans,H,W)]
        return bev_feat


@NECKS.register_module()
class ViewTransformerLiftSplatShootTemporalDETR(BaseModule):
    def __init__(self, grid_config=None, data_config=None,
                 numC_input=512, numC_Trans=64, downsample=16,
                 image_view_supervision=False, voxel=False, **kwargs):
        super(ViewTransformerLiftSplatShootTemporalDETR, self).__init__()
        if grid_config is None:
            grid_config = {
                'xbound': [-51.2, 51.2, 0.8],
                'ybound': [-51.2, 51.2, 0.8],
                'zbound': [-10.0, 10.0, 20.0],
                'dbound': [1.0, 60.0, 1.0],}
        self.grid_config = grid_config
        dx, bx, nx = gen_dx_bx(self.grid_config['xbound'],
                               self.grid_config['ybound'],
                               self.grid_config['zbound'],
                               )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        if data_config is None:
            data_config = {'input_size': (256, 704)}
        self.data_config = data_config
        self.downsample = downsample

        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.numC_input = numC_input
        self.numC_Trans = numC_Trans
        self.depthnet = nn.Conv2d(self.numC_input, self.D + self.numC_Trans, kernel_size=1, padding=0)
        self.geom_feats = None
        self.image_view_supervision = image_view_supervision
        self.voxel=voxel

        # GRU
        self.dropout = nn.Dropout(0.2)
        self.n_layers = 2
        self.hidden_dim = 64
        self.embed_dim = 32768
        self.out = nn.Linear(self.hidden_dim, self.embed_dim)
        self.gru = nn.GRU(self.embed_dim, self.hidden_dim, num_layers=self.n_layers, batch_first=True)

    def get_depth_dist(self, x):
        return x.softmax(dim=1)

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_config['input_size']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.grid_config['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans, offset=None):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        if offset is not None:
            _,D,H,W = offset.shape
            points[:,:,:,:,:,2] = points[:,:,:,:,:,2]+offset.view(B,N,D,H,W)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        if intrins.shape[3]==4: # for KITTI
            shift = intrins[:,:,:3,3]
            points  = points - shift.view(B,N,1,1,1,3,1)
            intrins = intrins[:,:,:3,:3]
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        # points_numpy = points.detach().cpu().numpy()
        return points

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        #Bg, Ng, Dg, Hg, Wg, Cg = geom_feats.shape
        #pdb.set_trace()
        Nprime = B * N * D * H * W
        #Nprime_g = Bg * Ng * Dg * Hg * Wg
        nx = self.nx.to(torch.long)
        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        #geom_feats = geom_feats.view(-1, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                         device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
                + geom_feats[:, 1] * (self.nx[2] * B) \
                + geom_feats[:, 2] * B \
                + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, nx[2], nx[1], nx[0]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1], geom_feats[:, 0]] = x
        if self.voxel:
            return final.sum(2), x, geom_feats
        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def forward(self, input):
        x, rots, trans, intrins, post_rots, post_trans, prev_x, prev_rots, prev_trans, prev_intrins, prev_post_rots, prev_post_trans = input
        B, N, C, H, W = x.shape # N = 6
        #pdb.set_trace()
        x = x.view(B * N, C, H, W)
        x = self.depthnet(x)
        depth = self.get_depth_dist(x[:, :self.D])
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        img_feat = x[:, self.D:(self.D + self.numC_Trans)]

        # Lift
        volume = depth.unsqueeze(1) * img_feat.unsqueeze(2)
        volume = volume.view(B, N, self.numC_Trans, self.D, H, W)
        volume = volume.permute(0, 1, 3, 4, 5, 2)

        # Splat
        cur_bev = self.voxel_pooling(geom, volume) # (batch, 64, 128, 128)
        
        # chgd (handle previous frames)
        iters = int(prev_x.shape[1]/6)
        prev_img1 = prev_x[:,:6] # t-1
        prev_img2 = prev_x[:,6:] # t-2
        prev_bevs = torch.zeros([2, B, 64, 128,128])

        #pdb.set_trace()
        for i in range(iters):
            if i == 0:
                x = prev_img1
                rots, trans, intrins, post_rots, post_trans = \
                            prev_rots[:,:6], prev_trans[:,:6], prev_intrins[:,:6], prev_post_rots[:,:6], prev_post_trans[:,:6]
            if i == 1:
                x = prev_img2
                rots, trans, intrins, post_rots, post_trans = \
                            prev_rots[:,6:], prev_trans[:,6:], prev_intrins[:,6:], prev_post_rots[:,6:], prev_post_trans[:,6:]

            #pdb.set_trace()
            B, N, C, H, W = x.shape
            x = x.reshape(B * N, C, H, W)
            x = self.depthnet(x)
            depth = self.get_depth_dist(x[:, :self.D])
            geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
            img_feat = x[:, self.D:(self.D + self.numC_Trans)]

            # Lift
            volume = depth.unsqueeze(1) * img_feat.unsqueeze(2)
            volume = volume.view(B, N, self.numC_Trans, self.D, H, W)
            volume = volume.permute(0, 1, 3, 4, 5, 2)

            # Splat
            bev_feat = self.voxel_pooling(geom, volume) # (batch, 64, 128, 128)

            prev_bevs[:][i] = bev_feat

        # Feed current and prev bev features into GRU
        prev_bevs = prev_bevs.permute(1, 2, 0, 3, 4)
        prev_bevs = prev_bevs.reshape(B, 64, -1)
        res, _ = self.gru(prev_bevs.cuda())
        res = self.dropout(res)
        res = self.out(res)

        # Reshape to original shape
        res = res.reshape(2, 64, B, 128, 128)
        res = res.permute(0, 2, 1, 3, 4) # B, N, C, H, W

        #pdb.set_trace()
        # Concat with frame at t and max along
        if self.training == False:
            cur_bev = torch.unsqueeze(cur_bev, 1).permute(1,0,2,3,4)
            res = res.permute(1,0,2,3,4)
        else:
            cur_bev = torch.unsqueeze(cur_bev, 1)
            res = res.permute(1,0,2,3,4)


        # Omit max operation for attention calculation
        '''
        tmp_concat = torch.cat([cur_bev.cuda(), res], axis=1)
        bev_feat = torch.max(tmp_concat, dim=1)
        bev_feat = bev_feat[0].cuda()
        '''
        bev_feat = torch.cat([cur_bev.cuda(), res], axis=1)

        return bev_feat



@NECKS.register_module()
class ViewTransformerLiftSplatShootTemporalAlign(BaseModule):
    def __init__(self, grid_config=None, data_config=None,
                 numC_input=512, numC_Trans=64, downsample=16,
                 image_view_supervision=False, voxel=False, **kwargs):
        super(ViewTransformerLiftSplatShootTemporalAlign, self).__init__()
        if grid_config is None:
            grid_config = {
                'xbound': [-51.2, 51.2, 0.8],
                'ybound': [-51.2, 51.2, 0.8],
                'zbound': [-10.0, 10.0, 20.0],
                'dbound': [1.0, 60.0, 1.0],}
        self.grid_config = grid_config
        dx, bx, nx = gen_dx_bx(self.grid_config['xbound'],
                               self.grid_config['ybound'],
                               self.grid_config['zbound'],
                               )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        if data_config is None:
            data_config = {'input_size': (256, 704)}
        self.data_config = data_config
        self.downsample = downsample

        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.numC_input = numC_input
        self.numC_Trans = numC_Trans
        self.depthnet = nn.Conv2d(self.numC_input, self.D + self.numC_Trans, kernel_size=1, padding=0)
        self.geom_feats = None
        self.image_view_supervision = image_view_supervision
        self.voxel=voxel

        # GRU
        self.dropout = nn.Dropout(0.2)
        self.n_layers = 2
        self.hidden_dim = 64
        self.embed_dim = 32768
        self.out = nn.Linear(self.hidden_dim, self.embed_dim)
        self.gru = nn.GRU(self.embed_dim, self.hidden_dim, num_layers=self.n_layers, batch_first=True)

        # Feature Alignment
        self.before=True
        self.interpolation_mode='bilinear'

    def get_depth_dist(self, x):
        return x.softmax(dim=1)

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_config['input_size']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.grid_config['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans, offset=None):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        if offset is not None:
            _,D,H,W = offset.shape
            points[:,:,:,:,:,2] = points[:,:,:,:,:,2]+offset.view(B,N,D,H,W)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        if intrins.shape[3]==4: # for KITTI
            shift = intrins[:,:,:3,3]
            points  = points - shift.view(B,N,1,1,1,3,1)
            intrins = intrins[:,:,:3,:3]
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        # points_numpy = points.detach().cpu().numpy()
        return points

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        #Bg, Ng, Dg, Hg, Wg, Cg = geom_feats.shape
        #pdb.set_trace()
        Nprime = B * N * D * H * W
        #Nprime_g = Bg * Ng * Dg * Hg * Wg
        nx = self.nx.to(torch.long)
        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        #geom_feats = geom_feats.view(-1, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                         device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
                + geom_feats[:, 1] * (self.nx[2] * B) \
                + geom_feats[:, 2] * B \
                + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, nx[2], nx[1], nx[0]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1], geom_feats[:, 0]] = x
        if self.voxel:
            return final.sum(2), x, geom_feats
        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    # Feature Alignment
    def shift_feature(self, input, trans, rots):
        n, c, h, w = input.shape #[8, 64, 128, 128]
        _,v,_ =trans[0].shape # [8,6,3]
        # generate grid
        xs = torch.linspace(0, w - 1, w, dtype=input.dtype, device=input.device).view(1, w).expand(h, w)
        ys = torch.linspace(0, h - 1, h, dtype=input.dtype, device=input.device).view(h, 1).expand(h, w)
        grid = torch.stack((xs, ys, torch.ones_like(xs)), -1).view(1, h, w, 3).expand(n, h, w, 3).view(n,h,w,3,1)
        grid = grid

        # get transformation from current frame to adjacent frame
        l02c = torch.zeros((n,v,4,4),dtype=grid.dtype).to(grid)
        l02c[:,:,:3,:3] = rots[0]
        l02c[:,:,:3,3] = trans[0]
        l02c[:,:,3,3] =1

        l12c = torch.zeros((n,v,4,4),dtype=grid.dtype).to(grid)
        l12c[:,:,:3,:3] = rots[1]
        l12c[:,:,:3,3] = trans[1]
        l12c[:,:,3,3] =1
        # l0tol1 = l12c.matmul(torch.inverse(l02c))[:,0,:,:].view(n,1,1,4,4)
        l0tol1 = l02c.matmul(torch.inverse(l12c))[:,0,:,:].view(n,1,1,4,4)
        #pdb.set_trace()

        l0tol1 = l0tol1[:,:,:,[True,True,False,True],:][:,:,:,:,[True,True,False,True]]

        feat2bev = torch.zeros((3,3),dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.dx[0]
        feat2bev[1, 1] = self.dx[1]
        feat2bev[0, 2] = self.bx[0] - self.dx[0] / 2.
        feat2bev[1, 2] = self.bx[1] - self.dx[1] / 2.
        feat2bev[2, 2] = 1
        feat2bev = feat2bev.view(1,3,3)
        tf = torch.inverse(feat2bev).matmul(l0tol1).matmul(feat2bev)

        # transform and normalize
        grid = tf.matmul(grid)
        normalize_factor = torch.tensor([w - 1.0, h - 1.0], dtype=input.dtype, device=input.device)
        grid = grid[:,:,:,:2,0] / normalize_factor.view(1, 1, 1, 2) * 2.0 - 1.0
        output = F.grid_sample(input, grid.to(input.dtype), align_corners=True, mode=self.interpolation_mode)
        return output

    def forward(self, input):
        torch.autograd.set_detect_anomaly(True)
        x, rots, trans, intrins, post_rots, post_trans, prev_x, prev_rots, prev_trans, prev_intrins, prev_post_rots, prev_post_trans = input
        B, N, C, H, W = x.shape # N = 6
        #pdb.set_trace()
        x = x.view(B * N, C, H, W)
        x = self.depthnet(x)
        depth = self.get_depth_dist(x[:, :self.D])
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        img_feat = x[:, self.D:(self.D + self.numC_Trans)]

        # Lift
        volume = depth.unsqueeze(1) * img_feat.unsqueeze(2)
        volume = volume.view(B, N, self.numC_Trans, self.D, H, W)
        volume = volume.permute(0, 1, 3, 4, 5, 2)

        # Splat
        cur_bev = self.voxel_pooling(geom, volume) # (batch, 64, 128, 128)
        
        # chgd (handle previous frames)
        iters = int(prev_x.shape[1]/6)
        prev_img1 = prev_x[:,:6] # t-1
        prev_img2 = prev_x[:,6:] # t-2
        prev_bevs = torch.zeros([2, B, 64, 128,128])

        #pdb.set_trace()
        for i in range(iters):
            if i == 0:
                x = prev_img1
                rots, trans, intrins, post_rots, post_trans = \
                            prev_rots[:,:6], prev_trans[:,:6], prev_intrins[:,:6], prev_post_rots[:,:6], prev_post_trans[:,:6]
            if i == 1:
                x = prev_img2
                rots, trans, intrins, post_rots, post_trans = \
                            prev_rots[:,6:], prev_trans[:,6:], prev_intrins[:,6:], prev_post_rots[:,6:], prev_post_trans[:,6:]

            #pdb.set_trace()
            B, N, C, H, W = x.shape
            x = x.reshape(B * N, C, H, W)
            x = self.depthnet(x)
            depth = self.get_depth_dist(x[:, :self.D])
            geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
            img_feat = x[:, self.D:(self.D + self.numC_Trans)]

            # Lift
            volume = depth.unsqueeze(1) * img_feat.unsqueeze(2)
            volume = volume.view(B, N, self.numC_Trans, self.D, H, W)
            volume = volume.permute(0, 1, 3, 4, 5, 2)

            # Splat
            bev_feat = self.voxel_pooling(geom, volume) # (batch, 64, 128, 128)

            prev_bevs[:][i] = bev_feat

        # Feature alignment
        prev_1=prev_bevs[0].clone()
        prev_2=prev_bevs[1].clone()

        prev_1_rots = [rots]*2
        prev_1_trans = [trans]*2
        prev_2_rots = [rots]*2
        prev_2_trans = [trans]*2

        for i in range(2):
            if i == 0:
                prev_feat = prev_1
                prev_1_rots[1] = prev_rots[:,:6]
                prev_1_trans[1] = prev_trans[:,:6]
                prev_feat = self.shift_feature(prev_feat, prev_1_trans, prev_1_rots)
                #pdb.set_trace()
            else:
                prev_feat = prev_2
                prev_2_rots[1] = prev_rots[:,6:]
                prev_2_trans[1] = prev_trans[:,6:]
                prev_feat = self.shift_feature(prev_feat, prev_2_trans, prev_2_rots)

            prev_bevs[i] = prev_feat

        #pdb.set_trace()

        # Feed current and prev bev features into GRU
        # prev_bevs.shape: [2,8,64,128,128]
        prev_bevs = prev_bevs.permute(1, 2, 0, 3, 4)
        prev_bevs = prev_bevs.reshape(B, 64, -1)
        res, _ = self.gru(prev_bevs.cuda())
        res = self.dropout(res)
        res = self.out(res)

        # Reshape to original shape
        res = res.reshape(2, 64, B, 128, 128)
        res = res.permute(0, 2, 1, 3, 4) # B, N, C, H, W

        #pdb.set_trace()
        # Concat with frame at t and max along
        if self.training == False:
            cur_bev = torch.unsqueeze(cur_bev, 1).permute(1,0,2,3,4)
            res = res.permute(1,0,2,3,4)
        else:
            cur_bev = torch.unsqueeze(cur_bev, 1)
            res = res.permute(1,0,2,3,4)

        #pdb.set_trace()
        tmp_concat = torch.cat([cur_bev.cuda(), res], axis=1)
        bev_feat = torch.max(tmp_concat, dim=1)
        bev_feat = bev_feat[0].cuda()

        if self.image_view_supervision:
            return bev_feat, [x[:, :self.D].view(B,N,self.D,H,W), x[:, self.D:].view(B,N,self.numC_Trans,H,W)]
        return bev_feat


@NECKS.register_module()
class ViewTransformerLiftSplatShootTemporal_DFE(BaseModule):
    def __init__(self, grid_config=None, data_config=None,
                 numC_input=512, numC_Trans=64, downsample=16,
                 image_view_supervision=False, voxel=False, **kwargs):
        super(ViewTransformerLiftSplatShootTemporal_DFE, self).__init__()
        if grid_config is None:
            grid_config = {
                'xbound': [-51.2, 51.2, 0.8],
                'ybound': [-51.2, 51.2, 0.8],
                'zbound': [-10.0, 10.0, 20.0],
                'dbound': [1.0, 60.0, 1.0],}
        self.grid_config = grid_config
        dx, bx, nx = gen_dx_bx(self.grid_config['xbound'],
                               self.grid_config['ybound'],
                               self.grid_config['zbound'],
                               )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        if data_config is None:
            data_config = {'input_size': (256, 704)}
        self.data_config = data_config
        self.downsample = downsample

        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.numC_input = numC_input
        self.numC_Trans = numC_Trans
        self.depthnet = nn.Conv2d(self.numC_input, self.D + self.numC_Trans, kernel_size=1, padding=0)
        self.geom_feats = None
        self.image_view_supervision = image_view_supervision
        self.voxel=voxel

        # GRU
        self.dropout = nn.Dropout(0.2)
        self.n_layers = 2
        self.hidden_dim = 64
        self.embed_dim = 32768
        self.out = nn.Linear(self.hidden_dim, self.embed_dim)
        self.gru = nn.GRU(self.embed_dim, self.hidden_dim, num_layers=self.n_layers, batch_first=True)

        # dfe
        self.output_channel_num = 512
        self.depth_output = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.output_channel_num, int(self.output_channel_num/2), 3, padding=1),
            nn.BatchNorm2d(int(self.output_channel_num/2)),
            nn.ReLU(),
            nn.Conv2d(int(self.output_channel_num/2), 96, 1),
        )
        self.depth_down = nn.Conv2d(96, 12, 3, stride=1, padding=1, groups=12)
        self.acf = dfe_module(512, 512)


    def get_depth_dist(self, x):
        return x.softmax(dim=1)

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_config['input_size']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.grid_config['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans, offset=None):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        if offset is not None:
            _,D,H,W = offset.shape
            points[:,:,:,:,:,2] = points[:,:,:,:,:,2]+offset.view(B,N,D,H,W)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        if intrins.shape[3]==4: # for KITTI
            shift = intrins[:,:,:3,3]
            points  = points - shift.view(B,N,1,1,1,3,1)
            intrins = intrins[:,:,:3,:3]
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        # points_numpy = points.detach().cpu().numpy()
        return points

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        #Bg, Ng, Dg, Hg, Wg, Cg = geom_feats.shape
        #pdb.set_trace()
        Nprime = B * N * D * H * W
        #Nprime_g = Bg * Ng * Dg * Hg * Wg
        nx = self.nx.to(torch.long)
        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        #geom_feats = geom_feats.view(-1, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                         device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
                + geom_feats[:, 1] * (self.nx[2] * B) \
                + geom_feats[:, 2] * B \
                + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, nx[2], nx[1], nx[0]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1], geom_feats[:, 0]] = x
        if self.voxel:
            return final.sum(2), x, geom_feats
        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def forward(self, input):
        x, rots, trans, intrins, post_rots, post_trans, prev_x, prev_rots, prev_trans, prev_intrins, prev_post_rots, prev_post_trans = input
        B, N, C, H, W = x.shape # N = 6

        # depth enhencement
        tmp_x = torch.zeros([N, C, H, W])
        tmp_final_x = torch.zeros([B, N, C, H, W])
        for idx1, batch in enumerate(x):
            for idx2, img in enumerate(batch):
                img = torch.unsqueeze(img, 0)
                depth = self.depth_output(img)
                #N_, C_, H_, W_ = img.shape
                depth_guide = F.interpolate(depth, size=img.size()[2:], mode='bilinear', align_corners=False)
                depth_guide = self.depth_down(depth_guide)
                img = img + self.acf(img, depth_guide) # [1, 512, 16, 44]
                tmp_x[idx2] = img[0]

            tmp_final_x[idx1] = tmp_x

        x = tmp_final_x

        # original code
        x = x.view(B * N, C, H, W).cuda()
        x = self.depthnet(x)
        depth = self.get_depth_dist(x[:, :self.D])
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        img_feat = x[:, self.D:(self.D + self.numC_Trans)]

        # Lift
        volume = depth.unsqueeze(1) * img_feat.unsqueeze(2)
        volume = volume.view(B, N, self.numC_Trans, self.D, H, W)
        volume = volume.permute(0, 1, 3, 4, 5, 2)

        # Splat
        cur_bev = self.voxel_pooling(geom, volume) # (batch, 64, 128, 128)
       
        # chgd (handle previous frames)
        iters = int(prev_x.shape[1]/6)
        prev_img1 = prev_x[:,:6] # t-1
        prev_img2 = prev_x[:,6:] # t-2
        prev_bevs = torch.zeros([2, B, 64, 128,128])

        #pdb.set_trace()
        for i in range(iters):
            if i == 0:
                x = prev_img1
                rots, trans, intrins, post_rots, post_trans = \
                            prev_rots[:,:6], prev_trans[:,:6], prev_intrins[:,:6], prev_post_rots[:,:6], prev_post_trans[:,:6]
            if i == 1:
                x = prev_img2
                rots, trans, intrins, post_rots, post_trans = \
                            prev_rots[:,6:], prev_trans[:,6:], prev_intrins[:,6:], prev_post_rots[:,6:], prev_post_trans[:,6:]

            #pdb.set_trace()
            B, N, C, H, W = x.shape
            x = x.reshape(B * N, C, H, W)
            x = self.depthnet(x)
            depth = self.get_depth_dist(x[:, :self.D])
            geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
            img_feat = x[:, self.D:(self.D + self.numC_Trans)]

            # Lift
            volume = depth.unsqueeze(1) * img_feat.unsqueeze(2)
            volume = volume.view(B, N, self.numC_Trans, self.D, H, W)
            volume = volume.permute(0, 1, 3, 4, 5, 2)

            # Splat
            bev_feat = self.voxel_pooling(geom, volume) # (batch, 64, 128, 128)

            prev_bevs[:][i] = bev_feat

        # Feed current and prev bev features into GRU
        prev_bevs = prev_bevs.permute(1, 2, 0, 3, 4)
        prev_bevs = prev_bevs.reshape(B, 64, -1)
        res, _ = self.gru(prev_bevs.cuda())
        res = self.dropout(res)
        res = self.out(res)

        # Reshape to original shape
        res = res.reshape(2, 64, B, 128, 128)
        res = res.permute(0, 2, 1, 3, 4) # B, N, C, H, W

        #pdb.set_trace()
        # Concat with frame at t and max along
        if self.training == False:
            cur_bev = torch.unsqueeze(cur_bev, 1).permute(1,0,2,3,4)
            res = res.permute(1,0,2,3,4)
        else:
            cur_bev = torch.unsqueeze(cur_bev, 1)
            res = res.permute(1,0,2,3,4)


        # max pooling version
        tmp_concat = torch.cat([cur_bev.cuda(), res], axis=1) # 
        bev_feat = torch.max(tmp_concat, dim=1)
        bev_feat = bev_feat[0].cuda()
        

        return bev_feat


class dfe_module(nn.Module):

    def __init__(self, in_channels, out_channels): # 256, 256
        super(dfe_module, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True),
                                nn.Dropout2d(0.2, False))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        #pdb.set_trace()

    def forward(self, feat_ffm, coarse_x):
        #pdb.set_trace()
        N, D, H, W = coarse_x.size() # (1, 12, 36, 160)

        #depth prototype
        feat_ffm = self.conv1(feat_ffm) # X' (1, 256, 36, 160)
        _, C, _, _ = feat_ffm.size() # C = 256

        proj_query = coarse_x.view(N, D, -1) #
        proj_key = feat_ffm.view(N, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy # ?
        attention = self.softmax(energy_new)

        #depth enhancement
        attention = attention.permute(0, 2, 1)
        proj_value = coarse_x.view(N, D, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(N, C, H, W)
        out = self.conv2(out)

        return out