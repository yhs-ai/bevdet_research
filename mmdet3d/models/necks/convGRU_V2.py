import os
import torch
import pdb
import torchvision
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
                                    bias=self.bias)

        self.conv_can = nn.Conv2d(in_channels=input_dim+hidden_dim,
                              out_channels=self.hidden_dim, # for candidate neural memory
                              kernel_size=kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        # Deformable-conv
        kh, kw = 3, 3
        self.offset = torch.rand(self.input_shape[0], 2 * kh * kw, self.height, self.width).cuda()
        self.weight_1 = nn.init.xavier_normal_(torch.empty(64, 128, kh, kw)).cuda()
        self.weight_2 = nn.init.xavier_normal_(torch.empty(64, 128, kh, kw)).cuda()
        self.mask = torch.rand(self.input_shape[0], kh * kw, self.height, self.width).cuda()

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

    def forward(self, input_tensor, h_cur):
        """
        :param self:
        :param input_tensor: (b, c, h, w)
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w)
            current hidden and cell states respectively
        :return: h_next,
            next hidden state
        """
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
        combined_conv2 = torchvision.ops.deform_conv2d(input = combined_, #[6, 96, 128, 128]
                                                offset = self.offset,
                                                padding = self.padding,
                                                weight = self.weight_1,
                                                mask = self.mask) # -> [6, 64, 128, 128]

        motion_gate = torch.sigmoid(combined_conv2) # [6, 64, 128, 128]
        combined_2 = torch.cat([input_tensor, motion_gate*h_cur], dim=1) # [6, 128, 128, 128]

        cc_tnm = torchvision.ops.deform_conv2d(input = combined_2,
                                                offset = self.offset,
                                                padding = self.padding,
                                                weight = self.weight_2,
                                                mask = self.mask)
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
                # input current hidden and cell state then compute the next hidden and cell state through ConvLSTMCell forward function
                h = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], h_cur=h) # (b,t,c,h,w)
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
    model = ConvGRU(input_size=(height, width),
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