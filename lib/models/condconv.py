import torch
from torch import nn

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _single, _pair


class CondConv1d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, att_inchannel, stride=1, num_workers=8, padding=0,
                 dilation=1, dropout_rate=0.5, groups=1, bias=True, padding_mode='zeros'):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        self.isbias = bias
        super(CondConv1d, self).__init__(
            in_channels, out_channels * num_workers, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias, padding_mode)

        if num_workers < 1:
            raise ValueError('num_workers must be positive integer')

        self.num_workers = num_workers

        # self.weights = nn.Parameter(torch.Tensor(num_workers, in_channels, out_channels // groups, kernel_size[0]))
        self.weight = nn.Parameter(
            self.weight.view(in_channels, out_channels, kernel_size[0], num_workers))

        if bias:
            self.bias = nn.Parameter(self.bias.view(out_channels, num_workers))

        self.att_linear1 = nn.Linear(att_inchannel, in_channels)
        self.att_linear2 = nn.Linear(in_channels, in_channels)
        self.att_linear = nn.Linear(in_channels, num_workers)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, att_x):  # batchsize * 512 * 16   batchsize * 512
        # x_shape = x.shape
        # 结合x
        # mix_feature = torch.tanh(
        #     self.att_linear1(att_x) + self.att_linear2(nn.functional.avg_pool1d(x, x_shape[2]).squeeze()))

        # 单纯使用word
        mix_feature = torch.tanh(self.att_linear1(att_x))

        attention = nn.functional.softmax(self.att_linear(mix_feature), dim=1)
        attention = self.dropout(attention)  # batchsize * num_workers

        kernel = torch.sum(attention[:, None, None, None, :] * self.weight[None, :, :, :, :], 4)

        output = []

        inputs = torch.split(x, 1, 0)

        if self.isbias:
            biases = torch.sum(attention[:, None, :] * self.bias[None, :, :], 2)
            for input_tensor, one_kernel, one_bias in zip(inputs, kernel, biases):
                t_out = nn.functional.conv1d(input_tensor, one_kernel, one_bias, self.stride, self.padding,
                                             self.dilation,
                                             self.groups)
                output.append(t_out)

        else:
            for input_tensor, one_kernel in zip(inputs, kernel):
                t_out = nn.functional.conv1d(input_tensor, one_kernel, None, self.stride, self.padding, self.dilation,
                                             self.groups)
                output.append(t_out)

        return torch.cat(output, dim=0)


class CondConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, att_inchannel, stride=1, num_workers=8, padding=0,
                 dilation=1, dropout_rate=0.1, groups=1, bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.isbias = bias
        super(CondConv2d, self).__init__(
            in_channels, out_channels * num_workers, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias, padding_mode)

        if num_workers < 1:
            raise ValueError('num_workers must be positive integer')

        self.num_workers = num_workers

        # self.weights = nn.Parameter(torch.Tensor(num_workers, in_channels, out_channels // groups, kernel_size[0]))
        self.weight = nn.Parameter(
            self.weight.view(in_channels, out_channels, kernel_size[0], kernel_size[1], num_workers))

        if bias:
            self.bias = nn.Parameter(self.bias.view(out_channels, num_workers))

        self.att_linear1 = nn.Linear(att_inchannel, in_channels)
        self.att_linear2 = nn.Linear(in_channels, in_channels)
        self.att_linear = nn.Linear(in_channels, num_workers)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, att_x, att_with_x: bool = True):  # batchsize * 512 * 16   batchsize * 512
        if att_with_x:
            # 结合x
            mix_feature = torch.tanh(
                self.att_linear1(att_x) + self.att_linear2(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze()))
        else:
            # 单纯使用word
            mix_feature = torch.tanh(self.att_linear1(att_x))

        attention = nn.functional.softmax(self.att_linear(mix_feature), dim=1)
        attention = self.dropout(attention)  # batchsize * num_workers

        # kernel = torch.sum(attention[:, None, None, None, None, :] * self.weight[None, :, :, :, :, :], 5)
        batchsize = x.size(0)
        kernel = torch.zeros(batchsize, self.weight.shape[0], self.weight.shape[1], self.weight.shape[2],
                             self.weight.shape[3]).cuda()
        for i in range(self.num_workers):
            kernel += attention[:, None, None, None, None, i] * self.weight[None, :, :, :, :, i]

        output = []

        inputs = torch.split(x, 1, 0)

        if self.isbias:
            # biases = torch.sum(attention[:, None, :] * self.bias[None, :, :], 2)
            biases = torch.zeros(batchsize, self.bias.shape[0]).cuda()
            for i in range(self.num_workers):
                biases += attention[:, None, i] * self.bias[None, :, i]

            for input_tensor, one_kernel, one_bias in zip(inputs, kernel, biases):
                t_out = nn.functional.conv2d(input_tensor, one_kernel, one_bias, self.stride, self.padding,
                                             self.dilation,
                                             self.groups)
                output.append(t_out)

        else:
            for input_tensor, one_kernel in zip(inputs, kernel):
                t_out = nn.functional.conv2d(input_tensor, one_kernel, None, self.stride, self.padding, self.dilation,
                                             self.groups)
                output.append(t_out)

        return torch.cat(output, dim=0)


class EfficientCondConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, att_inchannel, stride=1, num_workers=8, padding=0,
                 dilation=1, dropout_rate=0.1, groups=1, bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.isbias = bias

        super(EfficientCondConv2d, self).__init__(
            in_channels, out_channels * num_workers, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias, padding_mode)

        if num_workers < 1:
            raise ValueError('num_workers must be positive integer')

        self.num_workers = num_workers

        self.weight = nn.Parameter(self.weight.view(num_workers, -1))

        if bias:
            self.bias = nn.Parameter(self.bias.view(num_workers, -1))

        self.att_linear1 = nn.Linear(att_inchannel, in_channels)
        self.att_linear2 = nn.Linear(in_channels, in_channels)
        self.att_linear = nn.Linear(in_channels, num_workers)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, att_x, att_with_x: bool = True):  # batchsize * 512 * 16   batchsize * 512
        if att_with_x:
            # 结合x
            mix_feature = torch.tanh(
                self.att_linear1(att_x) + self.att_linear2(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze()))
        else:
            # 单纯使用word
            mix_feature = torch.tanh(self.att_linear1(att_x))

        attention = nn.functional.softmax(self.att_linear(mix_feature), dim=1)
        attention = self.dropout(attention)  # batchsize * num_workers

        kernel = torch.matmul(attention, self.weight).view(-1, self.out_channels // self.num_workers, self.in_channels,
                                                           self.kernel_size[0],
                                                           self.kernel_size[1])  # batchsize * xxxx
        output = []

        inputs = torch.split(x, 1, 0)

        if self.isbias:
            bias = torch.matmul(attention, self.bias).view(-1, self.out_channels // self.num_workers)

            for i, input_tensor in enumerate(inputs):
                t_out = nn.functional.conv2d(input_tensor, kernel[i, :, :, :, :], bias[i, :], self.stride,
                                             self.padding, self.dilation, self.groups)
                output.append(t_out)

        else:
            for i, input_tensor in enumerate(inputs):
                t_out = nn.functional.conv2d(input_tensor, kernel[i, :, :, :, :], None, self.stride, self.padding,
                                             self.dilation, self.groups)
                output.append(t_out)

        return torch.cat(output, dim=0)


class MoreEfficientCondConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, att_inchannel, stride=1, num_workers=8, padding=0,
                 dilation=1, dropout_rate=0.1, groups=1, bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.isbias = bias

        super(MoreEfficientCondConv2d, self).__init__(
            in_channels, out_channels * num_workers, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias, padding_mode)

        if num_workers < 1:
            raise ValueError('num_workers must be positive integer')

        self.true_out_channels = out_channels

        self.weight = nn.Parameter(self.weight.view(num_workers, -1))

        if bias:
            self.bias = nn.Parameter(self.bias.view(num_workers, -1))

        self.att_linear1 = nn.Linear(att_inchannel, in_channels)
        self.att_linear2 = nn.Linear(in_channels, in_channels)
        self.att_linear = nn.Linear(in_channels, num_workers)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, att_x, att_with_x: bool = True,
                att_with_att: bool = True):  # batchsize * 512 * 16 * 16  batchsize * 512
        if att_with_x and att_with_att:
            # 结合x
            mix_feature = torch.tanh(
                self.att_linear1(att_x) + self.att_linear2(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(2).squeeze(2)))
        elif not att_with_x and att_with_att:
            # 单纯使用word
            mix_feature = torch.tanh(self.att_linear1(att_x))
        elif att_with_x and not att_with_att:
            mix_feature = torch.tanh(self.att_linear2(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(2).squeeze(2)))
        else:
            raise Exception('must have attention in condconv')

        attention = nn.functional.softmax(self.att_linear(mix_feature), dim=1)
        attention = self.dropout(attention)  # batchsize * num_workers

        kernel = torch.matmul(attention, self.weight).view(-1, self.in_channels, self.kernel_size[0],
                                                           self.kernel_size[1])
        b, c, w, h = x.size()
        inputs = x.view(1, -1, w, h)

        if self.isbias:
            bias = torch.matmul(attention, self.bias).view(-1)
        else:
            bias = None

        t_out = nn.functional.conv2d(inputs, kernel, bias, self.stride, self.padding, self.dilation, b)
        output = t_out.view(b, self.true_out_channels, t_out.size(2), t_out.size(3))

        return output


class LittleMoreEfficientCondConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, att_inchannel, stride=1, num_workers=8, padding=0,
                 dilation=1, dropout_rate=0.1, groups=1, bias=True, padding_mode='zeros'):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.isbias = bias

        super(LittleMoreEfficientCondConv2d, self).__init__()

        if num_workers < 1:
            raise ValueError('num_workers must be positive integer')

        self.true_out_channels = out_channels

        self.weight_linear = nn.Linear(num_workers,
                                       out_channels * in_channels * self.kernel_size[0] *
                                       self.kernel_size[1], bias=False)
        if bias:
            self.bias_linear = nn.Linear(num_workers, out_channels, bias=False)

        self.att_linear1 = nn.Linear(att_inchannel, in_channels)
        self.att_linear2 = nn.Linear(in_channels, in_channels)
        self.att_linear = nn.Linear(in_channels, num_workers)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, att_x, att_with_x: bool = True):  # batchsize * 512 * 16 * 16  batchsize * 512
        if att_with_x:
            # 结合x
            mix_feature = torch.tanh(
                self.att_linear1(att_x) + self.att_linear2(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze()))
        else:
            # 单纯使用word
            mix_feature = torch.tanh(self.att_linear1(att_x))

        attention = nn.functional.softmax(self.att_linear(mix_feature), dim=1)
        attention = self.dropout(attention)  # batchsize * num_workers

        kernel = self.weight_linear(attention).view(-1, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        if self.isbias:
            bias = self.bias_linear(attention).view(-1)
        else:
            bias = None

        b, c, w, h = x.size()
        inputs = x.view(1, -1, w, h)

        t_out = nn.functional.conv2d(inputs, kernel, bias, self.stride, self.padding, self.dilation, b)
        output = t_out.view(b, self.true_out_channels, t_out.size(2), t_out.size(3))

        return output