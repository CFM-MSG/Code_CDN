import torch
from torch import nn
import torch.nn.functional as F

class EasyFusion(nn.Module):

    def __init__(self, cfg):
        super(EasyFusion, self).__init__()
        self.cfg = cfg
        hidden_size = cfg.HIDDEN_SIZE  # 512
        txt_hidden_size = cfg.TXT_HIDDEN_SIZE  # 512
        self.tex_linear = nn.Linear(txt_hidden_size, hidden_size)
        self.vis_conv = nn.Conv2d(hidden_size, hidden_size, 1, 1)

    def forward(self, textual_input, textual_mask, map_h, map_mask):
        txt_h = self.tex_linear(textual_input)[:, :, None, None]  # batchsize * 512 * 1 * 1
        map_h = self.vis_conv(map_h)  # batchsize * 512 *16 * 16
        fused_h = F.normalize(txt_h * map_h) * map_mask  # batchsize * 512 *16 * 16
        return fused_h