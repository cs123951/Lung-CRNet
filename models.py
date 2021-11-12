#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 15:27:14 2018

@author: user
"""


import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class conv_down(nn.Module):
    """
    it includes: convolution, batchnorm, relu, and pool(if is downsampled)
    The kernel size of convolution layer is constantly 3.
    """
    def __init__(self, inChan, outChan, down=True, pool_kernel=2):
        super(conv_down, self).__init__()
        self.down = down
        self.conv = nn.Sequential(
            nn.Conv3d(inChan, outChan, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(outChan),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.AvgPool3d(pool_kernel)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, inp_tensor):
        inp_tensor = self.conv(inp_tensor)
        if self.down:
            inp_tensor = self.pool(inp_tensor)
        return inp_tensor


class conv_disp(nn.Module):
    def __init__(self, inChan, kernel_size=1):
        super(conv_disp, self).__init__()
        self.conv = nn.Conv3d(
            inChan, 3, kernel_size=kernel_size, stride=1, padding=int(kernel_size//2), bias=True)
        self.conv.weight.data.normal_(mean=0, std=1e-5)
        if self.conv.bias is not None:
            self.conv.bias.data.zero_()

    def forward(self, inp_tensor):
        # print("x.shape: ", x.shape, type(x), type(self.conv.weight))
        inp_tensor = self.conv(inp_tensor)
        return inp_tensor


class FCN(nn.Module):
    def __init__(self, device_ids, input_device, output_device, img_size, down_out_channel_list, same_out_channel_list):
        super(FCN, self).__init__()
        self.img_size = img_size
        self.device_ids = device_ids
        self.input_device = input_device
        self.output_device = output_device
        self.down_device = self.device_ids[0]
        self.same_device = self.device_ids[0]
        self.disp_device = self.device_ids[0]
        assert input_device == self.down_device and output_device == self.disp_device

        self.ndown = len(down_out_channel_list)
        self.scale = 2**(self.ndown)
        input_channel = 2
        self.down1 = conv_down(input_channel, down_out_channel_list[0]).cuda(self.down_device)
        self.down2 = conv_down(down_out_channel_list[0], down_out_channel_list[1]).cuda(self.down_device)
        self.down3 = conv_down(down_out_channel_list[1], down_out_channel_list[2]).cuda(self.down_device)
        self.same1 = conv_down(down_out_channel_list[2], same_out_channel_list[0], down=False).cuda(self.same_device)
        self.same2 = conv_down(same_out_channel_list[0], same_out_channel_list[1], down=False).cuda(self.same_device)
        self.same3 = conv_down(same_out_channel_list[1], same_out_channel_list[2], down=False).cuda(self.same_device)
        self.outconv = nn.Conv3d(
            same_out_channel_list[2], 3, kernel_size=1, stride=1, padding=0, bias=True).cuda(self.disp_device)

    def forward(self, x):
        # x [batch, seq, channel, x, y, z] ==> [batch, seq, x, y, z]
        x = x.squeeze(2)
        # print("1. shape of x 3: ", x.shape, ", max of x: ", x.max(), ", min of x: ", x.min())
        x = self.down1(x.cuda(self.down_device))
        # print("2. shape of x 3: ", x.shape, ", max of x: ", x.max(), ", min of x: ", x.min())
        x = self.down2(x)
        # print("3. shape of x 3: ", x.shape, ", max of x: ", x.max(), ", min of x: ", x.min())
        x = self.down3(x)
        # print("4. shape of x 3: ", x.shape, ", max of x: ", x.max(), ", min of x: ", x.min())

        x = self.same1(x.cuda(self.same_device))
        # print("5. shape of x 3: ", x.shape, ", max of x: ", x.max(), ", min of x: ", x.min())
        # print("shape of x: ", x.shape)
        x = self.same2(x)
        # print("6. shape of x 3: ", x.shape, ", max of x: ", x.max(), ", min of x: ", x.min())
        # print("shape of x: ", x.shape)
        x = self.same3(x)
        # print("7. shape of x 3: ", x.shape, ", max of x: ", x.max(), ", min of x: ", x.min())
        x = self.outconv(x.cuda(self.disp_device))
        # print("shape of x 3: ", x.shape, ", max of x: ", x.max(), ", min of x: ", x.min())
        # 上采样

        x = F.interpolate(x, scale_factor=self.scale, mode='trilinear', align_corners=True)  # False
        # print("shape of flow: ", x.shape, ", max of x: ", x.max(), ", min of x: ", x.min())
        return x


# https://github.com/happyjin/ConvGRU-pytorch/blob/master/convGRU.py
# https://github.com/bionick87/ConvGRUCell-pytorch/blob/master/Conv-GRU.py ——比较容易懂
class ConvGRUCell(nn.Module):
    def __init__(self, input_size, input_dim=2, hidden_dim=16, kernel_size=3, bias=True):
        """

        :param input_size: [x,y,z]
            Shape of the input tensor except for batch and channel
        :param input_dim: int
            Number of channels of input tensor.
        :param hidden_dim: int
            Number of channels of hidden state.
        :param kernel_size: [3,3,3]
             Size of the convolutional kernel.
        :param bias:
             Whether or not to add the bias.
        """
        super(ConvGRUCell, self).__init__()
        self.x_size, self.y_size, self.z_size = input_size
        self.padding = kernel_size//2
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.conv_gates = nn.Conv3d(in_channels=input_dim + hidden_dim,
                                    out_channels=2*self.hidden_dim,  # for update_gate,reset_gate respectively
                                    kernel_size=kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

        self.conv_can = nn.Conv3d(in_channels=input_dim + hidden_dim,
                                  out_channels=self.hidden_dim,  # for candidate neural memory
                                  kernel_size=kernel_size,
                                  padding=self.padding,
                                  bias=self.bias)

    def forward(self, input_tensor, h_cur):
        # input_tensor: [batch, channel, x,y,z]
        if h_cur is None:
            h_cur = Variable(torch.zeros(input_tensor.shape[0], self.hidden_dim, self.x_size, self.y_size, self.z_size)).cuda(input_tensor.device)
        combined = torch.cat([input_tensor, h_cur], dim=1)  # [batch, hidden_dim+input_dim, x, y, z]
        combined_conv = self.conv_gates(combined)  # [batch, 2*hidden_dim, x, y, z]
        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)  # [batch, hidden_dim, x, y, z], [batch, hidden_dim, x, y, z]
        reset_gate = torch.sigmoid(gamma)  # R_t  [batch, hidden_dim, x, y, z]
        update_gate = torch.sigmoid(beta)  # Z_t  [batch, hidden_dim, x, y, z]

        combined = torch.cat([input_tensor, reset_gate * h_cur], dim=1)  # X_t, R_t \circ H_(t-1)   [batch, input_dim+hidden_dim, x, y, z]
        cc_cnm = self.conv_can(combined)  # [batch, hidden_dim, x, y, z]
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm  # [batch, hidden_dim, x, y, z]

        return h_next


class ConvGRU(nn.Module):
    def __init__(self, img_size, input_dim, hidden_dim_list, kernel_size_list, num_layers,
                 batch_first=True, bias=True):
        """
        :param device_ids: [int,int,...]
            The GPU ids that are assigned to the module
        :param input_size: (int, int, int)
            The shape of the input tensor
        :param input_dim: int e.g. 256
            Number of channels of input tensor.
        :param hidden_dim_list: list of size num_layers
            Number of channels of hidden state.
        :param kernel_size_list: list of size num_layers
            Size of the convolutional kernel.
        :param num_layers: int
            Number of ConvLSTM layers
        :param batch_first: bool
            if the first position of array is batch or not
        :param bias: bool
            Whether or not to add the bias.
        """
        super(ConvGRU, self).__init__()

        self.img_size= img_size
        self.input_dim = input_dim
        self.hidden_dim_list = hidden_dim_list
        self.kernel_size_list = kernel_size_list
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias

        cell_list = []
        for i in range(0, self.num_layers):
            # for [1,] layer, the input is the last hidden output
            cur_input_dim = input_dim if i == 0 else hidden_dim_list[i - 1]
            cell_list.append(ConvGRUCell(input_size=img_size,
                                         input_dim=cur_input_dim,
                                         hidden_dim=self.hidden_dim_list[i],
                                         kernel_size=self.kernel_size_list[i],
                                         bias=self.bias))

        # convert python list to pytorch module
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor):
        """
        :param input_tensor: (b, t, c, x, y, z)
        :return:
        """
        seq_len = input_tensor.shape[1]
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h = None
            output_inner = []
            for t in range(seq_len):
                # input current hidden and cell state, then compute the next hidden and cell state.
                convgru_cell = self.cell_list[layer_idx]
                h = convgru_cell(input_tensor=cur_layer_input[:, t], h_cur=h)
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

        return layer_output


class CRNet(nn.Module):
    def __init__(self, device_ids, input_device, output_device, img_size, down_out_channel_list,
                 num_layers, hidden_dim_list, kernel_size_list,
                 batch_first=True, bias=True):
        super(CRNet, self).__init__()

        # setting devices for different modules
        self.device_ids = device_ids
        self.input_device = input_device
        self.output_device = output_device
        self.down1_device = self.device_ids[0]
        self.down2_device = self.device_ids[0]
        self.convgru_device = self.device_ids[0]
        self.disp_device = output_device # self.device_ids[0]
        assert input_device == self.down1_device and output_device == self.disp_device

        self.down1 = conv_down(1, down_out_channel_list[0]).cuda(self.down1_device)
        self.down2 = conv_down(down_out_channel_list[0], down_out_channel_list[1]).cuda(self.down2_device)

        convgru_size = [img_size[i]//4 for i in range(3)]
        self.multiConvGRU = ConvGRU(img_size=convgru_size, input_dim=down_out_channel_list[-1], hidden_dim_list=hidden_dim_list,
                                   kernel_size_list=kernel_size_list, num_layers=num_layers,
                                    batch_first=batch_first, bias=bias).cuda(self.convgru_device)

        self.outconv3 = conv_disp(hidden_dim_list[-1], kernel_size=3).cuda(self.disp_device)

    def forward(self, img_seq):
        cur_layer_input = []

        seq_len = img_seq.shape[1]
        for t in range(seq_len):
            d1 = self.down1(img_seq[:, t].cuda(self.down1_device))
            d2 = self.down2(d1.cuda(self.down2_device))  # [batch, channel, x//4, y//4, z//4]
            cur_layer_input.append(d2)
        cur_layer_input = torch.stack(cur_layer_input, dim=1).cuda(self.convgru_device)  # [batch, seq, channel, x//4, y//4, z//4]

        output_convgru = self.multiConvGRU(cur_layer_input).cuda(self.disp_device)

        disp_list = []
        for t in range(1, seq_len):
            tmp_disp = self.outconv3(output_convgru[:, t])
            full_disp = F.interpolate(tmp_disp, scale_factor=4, mode="trilinear", align_corners=True)
            disp_list.append(full_disp)

        disp_output = torch.stack(disp_list, dim=1)
        return disp_output


class BiCRNet(nn.Module):
    """Bi-convolutional-recurrent neural network, three different devices"""
    def __init__(self, device_ids, input_device, output_device, img_size, down_out_channel_list,
                 num_layers, hidden_dim_list, kernel_size_list,
                 batch_first=True, bias=True):
        """

        :param device_ids:
        :param img_size:
        :param input_dim: int,
            Channel number of the input tensor
        :param hidden_dim: list,
            size of num_layers,
        :param kernel_size:
        :param num_layers:
        :param batch_first:
        :param bias:
        """
        super(BiCRNet, self).__init__()

        # setting devices for different modules
        self.device_ids = device_ids
        self.down1_device = input_device # self.device_ids[0]
        self.down2_device = self.device_ids[1]
        self.fwd_net_device = self.device_ids[1]
        self.rev_net_device = self.device_ids[1]
        self.disp_device = output_device # self.device_ids[1]

        # initialize modules
        self.down1 = conv_down(1, down_out_channel_list[0]).cuda(self.down1_device)
        self.down2 = conv_down(down_out_channel_list[0], down_out_channel_list[1]).cuda(self.down2_device)

        convgru_img_size = [img_size[i]//4 for i in range(3)]
        self.forward_net = ConvGRU(img_size=convgru_img_size, input_dim=down_out_channel_list[-1], hidden_dim_list=hidden_dim_list,
                                   kernel_size_list=kernel_size_list, num_layers=num_layers, batch_first=batch_first, bias=bias).cuda(self.fwd_net_device)
        self.reverse_net = ConvGRU(img_size=convgru_img_size, input_dim=down_out_channel_list[-1], hidden_dim_list=hidden_dim_list,
                                   kernel_size_list=kernel_size_list, num_layers=num_layers, batch_first=batch_first, bias=bias).cuda(self.rev_net_device)

        self.outconv3 = conv_disp(hidden_dim_list[-1]*2, kernel_size=3).cuda(self.disp_device)

    def forward(self, img_seq):
        """
        :param img_seq: [batch, seq, channel, x,y,z]
        """
        # append the first image into the tail of the sequence, so that it forms a ring
        img_seq_ring = torch.cat([img_seq, img_seq[:,0].unsqueeze(1)], dim=1)
        ring_seq_len = img_seq_ring.shape[1]

        # be input into the convolution down sampling layers
        cur_layer_input = []
        for t in range(ring_seq_len):
            d1 = self.down1(img_seq_ring[:, t].cuda(self.down1_device))
            d2 = self.down2(d1.cuda(self.down2_device))
            cur_layer_input.append(d2)

        # stacke the down sampled input into the format of [batch, seq, channel, x, y, z)
        cur_layer_input = torch.stack(cur_layer_input, dim=1)
        # the seq order of cur_layer_input_rev the reverse of cur_layer_input
        cur_layer_input_rev = torch.flip(cur_layer_input, dims=[1])

        y_out_fwd = self.forward_net(cur_layer_input.cuda(self.fwd_net_device))
        y_out_rev = self.reverse_net(cur_layer_input_rev.cuda(self.rev_net_device))  # [batch, seq, channel, x,y,z]

        y_out_rev_rev = torch.flip(y_out_rev, dims=[1])
        ycat = torch.cat((y_out_fwd, y_out_rev_rev), dim=2)  # cat in the channel axis

        disp_list = []
        for t in range(1, ring_seq_len-1):
            disp_list.append(self.outconv3(ycat[:, t]))

        for t in range(len(disp_list)):
            disp_list[t] = F.interpolate(disp_list[t], scale_factor=4, mode="trilinear", align_corners=True)

        seq_disp = torch.stack(disp_list, dim=1)
        return seq_disp

