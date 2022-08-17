import os
import torch
import pdb
import torchvision
import torch.nn.functional as F
import cv2
import numpy as np
from matplotlib import pyplot as plt
from torch import nn
from torch.autograd import Variable


class ConvGRUCell(nn.Module):
    def __init__(self, input_size, input_shape, input_dim, hidden_dim, kernel_size, bias, dtype):
        """
        Initialize the ConvLSTM cell
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int
            Number of channels of input tensor.
        :param hidden_dim: int
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param bias: bool
            Whether or not to add the bias.
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        """
        super(ConvGRUCell, self).__init__()
        self.height, self.width = input_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.dtype = dtype
        self.input_shape = input_shape

        self.conv_gates = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                    out_channels=2*self.hidden_dim,  # for update_gate,reset_gate respectively
                                    kernel_size=kernel_size,
                                    padding=self.padding,
                                    bias=self.bias).cuda()

        self.conv_can = nn.Conv2d(in_channels=input_dim+hidden_dim,
                              out_channels=self.hidden_dim, # for candidate neural memory
                              kernel_size=kernel_size,
                              padding=self.padding,
                              bias=self.bias).cuda()
        # Deformable-conv
        kh, kw = 3, 3
        self.offset = torch.rand(self.input_shape[0], 2 * kh * kw, self.height, self.width).cuda() # [1, 18, 128, 128]
        #self.weight_1 = nn.init.xavier_normal_(torch.empty(64, 128, kh, kw)).cuda()
        #self.weight_2 = nn.init.xavier_normal_(torch.empty(64, 128, kh, kw)).cuda()
        self.mask = torch.rand(self.input_shape[0], kh * kw, self.height, self.width).cuda()

        self.deform_conv_gates = torchvision.ops.DeformConv2d(in_channels= 64, out_channels= 64, kernel_size= 3, padding=1).cuda()
        self.deform_conv_can = torchvision.ops.DeformConv2d(in_channels= 64, out_channels= 64, kernel_size= 3, padding=1).cuda()


    # Ratio Temporary Idea
    def solveProportion(a, b1, b2, c):
        A = a * b2
        B = b1 * b2
        C = b1 * c
     
        # To print the given proportion
        # in simplest form.
        gcd1 = math.gcd(math.gcd(A, B), C)
        return gcd1

    def init_hidden(self, batch_size):
        # TODO: Initialize with normal distribution
        #return (Variable(torch.normal(mean=2, std=3, size=(batch_size, self.hidden_dim, self.height, self.width))).type(self.dtype))
       
        # Original Ver:
        #return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).type(self.dtype))

        # He Normal
        #return nn.init.kaiming_normal_((Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).type(self.dtype)))

        # Xavier Normal
        return nn.init.xavier_normal_((Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).type(self.dtype)))

    def normalize(self, image_features):
        image_features = image_features.squeeze(0).cpu().detach().numpy()
        min = image_features.min()
        max = image_features.max()
        image_features = (image_features-min)/(max-min)
        image_features = (image_features *255)
        return image_features

    def forward(self, input_tensor, h_cur):
        """
        :param self:
        :param input_tensor: (b, c, h, w) # test: (1, 64, 128, 128)
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w)
            current hidden and cell states respectively
        :return: h_next,
            next hidden state
        """
        '''
        # BEV visualization
        image_feature = self.normalize(input_tensor)
        sum_image_feature = (np.sum(np.transpose(image_feature, (1,2,0)),axis=2)/64).astype("uint8")
        max_image_feature = np.max(np.transpose(image_feature.astype("uint8"), (1,2,0)), axis=2)
        sum_image_feature = cv2.applyColorMap(sum_image_feature, cv2.COLORMAP_JET)
        max_image_feature = cv2.applyColorMap(max_image_feature, cv2.COLORMAP_JET)
        cv2.imwrite("sum_image_feature_1.jpg", sum_image_feature)
        cv2.imwrite("max_image_feature_1.jpg", max_image_feature)
        pdb.set_trace()
        '''
        '''
        # BEV visualization ver2
        for i in range(input_tensor[0].shape[0]):
            bev = input_tensor[0][i].cpu().detach().numpy()
            cv2.imwrite('./vis_images2/test_'+str(i)+'.jpg', bev*255/bev.max())
        pdb.set_trace()
        '''
        
        '''
        ############ For offset visualization
        #self.offset = torch.rand(self.input_shape[0], 2 * kh * kw, self.height, self.width).cuda() # [1, 18, 128, 128]
        #img_path = "./test_10_1.jpg" # t-1
        img_path = "./test_10_2.jpg" # t-2
        #img_path = "./data/visualize/n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281633662460.jpg"
        im = cv2.imread(img_path) # [900, 1600, 3]
        im = cv2.resize(im, (512, 512))

        '''
        #while 1:
        #    implot = cv2.imshow("output", im)
        #    key = cv2.waitKey(30)
        #    if key == 27:
        #        break
        #pdb.set_trace()
        '''

        # ROT ver 1
        #roi_x, roi_y = 253, 392
        #roi_x, roi_y = 250, 420
        #roi_x, roi_y = 240, 373 # static 2

        # ROI ver 2
        #roi_xs, roi_ys = [255, 253, 251, 251], [378, 387, 394, 404] # moving 1
        #roi_xs, roi_ys = [251, 251, 251, 250], [411, 420, 426, 435] # moving 2
        roi_xs, roi_ys = [241, 239, 240, 239], [356, 369, 376, 382] # static

        # original ver
        t_offset = self.offset # [1, 18, 128, 128]

        # second ver
        #t_offset = self.offset.cpu().detach().numpy()
        #t_offset = t_offset.reshape(1, 128,128,-1)
        #pdb.set_trace()
        #t_offset = cv2.resize(t_offset, dsize=None, fx=12, fy=7)
        #t_offset = t_offset.reshape(1, 18, 225, 400)
        for roi_x, roi_y in zip(roi_xs, roi_ys):
            for i in range(5):
                if i==0:
                    pass
                elif i==1:
                    roi_x += 1
                elif i==2:
                    roi_x -= 2
                elif i==3:
                    roi_y += 1
                else:
                    roi_y -= 2

                cv2.circle(im, center=(roi_x, roi_y), color=(0, 255, 0), radius=1, thickness=-1)
                #t_offset = F.interpolate(t_offset, size=(im.shape[0], im.shape[1]), mode='bilinear')
                #t_offset = t_offset.permute(0,2,3,1) # [1, 900, 1600, 18]

                #pdb.set_trace()
                offsets_y = t_offset[:, ::2]*10 # [1, 9, 128, 128]
                offsets_x = t_offset[:, 1::2]*10 # chgd
                #offsets_y = t_offset[:, :9] # [1, 9, 128, 128]
                #offsets_x = t_offset[:, 9:]

                grid_y = np.arange(0, 128) # array([0, 1, 2, 3, 4, 5, 6])
                grid_x = np.arange(0, 128) # array([0, 1, 2, 3, 4, 5, 6])

                grid_x, grid_y = np.meshgrid(grid_x, grid_y) # (128 x 128) grid each

                sampling_y = grid_y + offsets_y.detach().cpu().numpy() # (1, 9, 7, 7)
                sampling_x = grid_x + offsets_x.detach().cpu().numpy()

                resize_factor=4
                sampling_y *= resize_factor
                sampling_x *= resize_factor

                # remove batch axis
                sampling_y = sampling_y[0]
                sampling_x = sampling_x[0]

                sampling_y = sampling_y.transpose(1, 2, 0) # c, h, w -> h, w, c # (7, 7, 9)
                sampling_x = sampling_x.transpose(1, 2, 0) # c, h, w -> h, w, c

                sampling_y = np.clip(sampling_y, 0, 512) # (7, 7, 9)
                sampling_x = np.clip(sampling_x, 0, 512)

                sampling_y = cv2.resize(sampling_y, dsize=None, fx=resize_factor, fy=resize_factor) # (512, 512, 9)
                sampling_x = cv2.resize(sampling_x, dsize=None, fx=resize_factor, fy=resize_factor)
                #pdb.set_trace()

                sampling_y = sampling_y[roi_y, roi_x] # (9,)
                sampling_x = sampling_x[roi_y, roi_x]

                for y, x in zip(sampling_y, sampling_x):
                    y = round(y) # (9,)
                    x = round(x)
                    cv2.circle(im, center=(x, y), color=(0, 0, 255), radius=1, thickness=1)
        
        while 1:
            implot = cv2.imshow("output", im)
            key = cv2.waitKey(30)
            if key == 27:
                break
        pdb.set_trace()

        ######################################
        '''

        combined_ = torch.cat([input_tensor, h_cur], dim=1) # [6, 96(64+32), 128, 128]
        combined_conv = self.conv_gates(combined_) # [6, 64(2*hidden_dim), 128, 128]
        #alpha = self.deform_conv_gates(combined)

        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1) # [6, 32, 128, 128] x 3
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([input_tensor, reset_gate*h_cur], dim=1) # [6, 96(64+32), 128, 128]
        cc_cnm = self.conv_can(combined) # [6, 32, 128, 128]
        cnm = torch.tanh(cc_cnm) # [6, 32, 128, 128]

        #pdb.set_trace()
        # Motion Gate
        '''
        combined_conv2 = torchvision.ops.deform_conv2d(input = combined_, #[6, 96, 128, 128]
                                                offset = self.offset,
                                                padding = self.padding,
                                                weight = self.weight_1,
                                                mask = self.mask) # -> [6, 64, 128, 128]
        '''
        combined_conv2 = self.deform_conv_gates(combined_, offset = self.offset, mask = self.mask)

        motion_gate = torch.sigmoid(combined_conv2) # [6, 64, 128, 128]
        combined_2 = torch.cat([input_tensor, motion_gate*h_cur], dim=1) # [6, 128, 128, 128]
        '''
        cc_tnm = torchvision.ops.deform_conv2d(input = combined_2,
                                                offset = self.offset,
                                                padding = self.padding,
                                                weight = self.weight_2,
                                                mask = self.mask)
        '''
        cc_tnm = self.deform_conv_can(combined_2, offset = self.offset, mask = self.mask)
        tnm = torch.tanh(cc_tnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm + update_gate * tnm
        return h_next


class ConvGRUV2(nn.Module):
    def __init__(self, input_size, input_shape, input_dim, hidden_dim, kernel_size, num_layers,
                 dtype, batch_first=False, bias=True, return_all_layers=False):
        """

        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int e.g. 256
            Number of channels of input tensor.
        :param hidden_dim: int e.g. 1024
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param num_layers: int
            Number of ConvLSTM layers
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        :param alexnet_path: str
            pretrained alexnet parameters
        :param batch_first: bool
            if the first position of array is batch or not
        :param bias: bool
            Whether or not to add the bias.
        :param return_all_layers: bool
            if return hidden and cell states for all layers
        """
        super(ConvGRUV2, self).__init__()

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_shape = input_shape
        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.dtype = dtype
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers): # 2
            cur_input_dim = input_dim if i == 0 else hidden_dim[i - 1]
            cell_list.append(ConvGRUCell(input_size=(self.height, self.width),
                                         input_shape = self.input_shape,
                                         input_dim=cur_input_dim,
                                         hidden_dim=self.hidden_dim[i],
                                         kernel_size=self.kernel_size[i],
                                         bias=self.bias,
                                         dtype=self.dtype))

        # convert python list to pytorch module
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        :param input_tensor: (b, t, c, h, w) or (t,b,c,h,w) depends on if batch first or not
            extracted features from alexnet
        :param hidden_state:
        :return: layer_output_list, last_state_list
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0)) # list of len 2, [6, 32, 128, 128]

        layer_output_list = []
        last_state_list   = []

        #pdb.set_trace()
        seq_len = input_tensor.size(1) # 2
        cur_layer_input = input_tensor # [6, 2, 64, 128, 128]

        for layer_idx in range(self.num_layers): # 2
            h = hidden_state[layer_idx] # [6, 32, 128, 128]
            output_inner = []
            for t in range(seq_len):
                # input current hidden and cell state then compute the next hidden and cell state through ConvGRUCell forward function
                h = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :].cuda(), h_cur=h.cuda()) # (b,t,c,h,w)
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output # X_t-1 for next cell input

            layer_output_list.append(layer_output)
            last_state_list.append([h])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list   = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers): # 2
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


if __name__ == '__main__':
    # set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # detect if CUDA is available or not
    use_gpu = torch.cuda.is_available()
    use_gpu = False
    if use_gpu:
        dtype = torch.cuda.FloatTensor # computation in GPU
    else:
        dtype = torch.FloatTensor

    print("CUDA Availability: ", use_gpu)

    batch_size = 6
    time_steps = 2
    height = width = 128
    channels = 64
    input_tensor = torch.rand(batch_size, time_steps, channels, height, width)  # (b,t,c,h,w) = (6, 2, 64, 128, 128)
   
    hidden_dim = [64, 64]
    kernel_size = (3,3) # kernel size for two stacked hidden layer
    num_layers = 2 # number of stacked hidden layer
    model = ConvGRUV2(input_size=(height, width),
                    input_shape = input_tensor.shape,
                    input_dim=channels,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    num_layers=num_layers,
                    dtype=dtype,
                    batch_first=True,
                    bias = True,
                    return_all_layers = False)

   
    #pdb.set_trace()
    #input_tensor = torch.rand(batch_size, time_steps, channels, height, width)  # (b,t,c,h,w) = (6, 2, 64, 128, 128)
    layer_output_list, last_state_list = model(input_tensor)
    print("Result: ", layer_output_list[0].shape)