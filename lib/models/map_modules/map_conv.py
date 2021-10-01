from torch import nn
import torch.nn.functional as F
from models.map_modules import get_padded_mask_and_weight

from models.condconv import CondConv2d, MoreEfficientCondConv2d
from models.dilatedconv import ResDilatedConv2d, ResDilatedConv2d_NWS


class MapConv(nn.Module):

    def __init__(self, cfg):
        super(MapConv, self).__init__()
        input_size = cfg.INPUT_SIZE  # 512
        hidden_sizes = cfg.HIDDEN_SIZES  # [512, 512, 512, 512, 512, 512, 512, 512]
        kernel_sizes = cfg.KERNEL_SIZES  # [5, 5, 5, 5, 5, 5, 5, 5]
        strides = cfg.STRIDES  # [1, 1, 1, 1, 1, 1, 1, 1]
        paddings = cfg.PADDINGS  # [16, 0, 0, 0, 0, 0, 0, 0]
        dilations = cfg.DILATIONS  # [1, 1, 1, 1, 1, 1, 1, 1]
        self.convs = nn.ModuleList()
        assert len(hidden_sizes) == len(kernel_sizes) \
               and len(hidden_sizes) == len(strides) \
               and len(hidden_sizes) == len(paddings) \
               and len(hidden_sizes) == len(dilations)
        channel_sizes = [input_size] + hidden_sizes
        for i, (k, s, p, d) in enumerate(zip(kernel_sizes, strides, paddings, dilations)):
            self.convs.append(nn.Conv2d(channel_sizes[i], channel_sizes[i + 1], k, s, p, d))

    def forward(self, x, mask):
        padded_mask = mask
        for i, pred in enumerate(self.convs):
            x = F.relu(pred(x))
            padded_mask, masked_weight = get_padded_mask_and_weight(padded_mask, pred)
            x = x * masked_weight
        return x  # batchsize * 512 * 16 * 16


class DynamicMapResConv(nn.Module):
    def __init__(self, cfg):
        super(DynamicMapResConv, self).__init__()
        input_size = cfg.INPUT_SIZE  # 512
        hidden_sizes = cfg.HIDDEN_SIZES  # [512, 512, 512, 512, 512, 512, 512, 512]
        kernel_sizes = cfg.KERNEL_SIZES  # [5, 5, 5, 5, 5, 5, 5, 5]
        strides = cfg.STRIDES  # [1, 1, 1, 1, 1, 1, 1, 1]
        paddings = cfg.PADDINGS  # [2, 2, 2, 2, 2, 2, 2, 2]
        dilations = cfg.DILATIONS  # [1, 1, 1, 1, 1, 1, 1, 1]
        self.dymactic_conv = cfg.DYMATIC # [True, True, True, True, False, False, False, False]
        self.convs = nn.ModuleList()
        assert len(hidden_sizes) == len(kernel_sizes) \
               and len(hidden_sizes) == len(strides) \
               and len(hidden_sizes) == len(paddings) \
               and len(hidden_sizes) == len(dilations)
        channel_sizes = [input_size] + hidden_sizes
        self.cond_num = cfg.COND_NUM
        num_worker = cfg.NUM_WORKER

        self.bn_layers = nn.ModuleList(
            [nn.BatchNorm2d(hidden_sizes[i]).cuda() for i in range(0, len(hidden_sizes))])
        self.bn_layers.append(nn.BatchNorm2d(hidden_sizes[-1]).cuda())

        for i, (k, s, p, d, u) in enumerate(zip(kernel_sizes, strides, paddings, dilations, self.dymactic_conv)):
            if u:
                self.convs.append(
                    MoreEfficientCondConv2d(channel_sizes[i], channel_sizes[i + 1], k, input_size, s, padding=p,
                                            dilation=d,
                                            num_workers=num_worker))
            else:
                self.convs.append(
                    nn.Conv2d(channel_sizes[i], channel_sizes[i + 1], k, s, padding=p, dilation=d))

    def forward(self, x, mask, text_encode):
        padded_mask = mask
        for i, (pred, u) in enumerate(zip(self.convs, self.dymactic_conv)):
            x = self.bn_layers[i](x)
            if u:
                x = F.relu(pred(x, text_encode, att_with_att=True) + x)
            else:
                x = F.relu(pred(x) + x)
            padded_mask, masked_weight = get_padded_mask_and_weight(padded_mask, pred)
            x = x * masked_weight
        x = self.bn_layers[-1](x)
        return x  # batchsize * 512 * 16 * 16
