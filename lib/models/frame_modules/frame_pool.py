import torch
from torch import nn


class FrameAvgPool(nn.Module):

    def __init__(self, cfg):
        super(FrameAvgPool, self).__init__()
        input_size = cfg.INPUT_SIZE  # 4096
        hidden_size = cfg.HIDDEN_SIZE  # 512
        kernel_size = cfg.KERNEL_SIZE  # 16
        stride = cfg.STRIDE
        self.vis_conv = nn.Conv1d(input_size, hidden_size, 1, 1)
        self.avg_pool = nn.AvgPool1d(kernel_size, stride)

    def forward(self, visual_input):  # batchsize * 4096 * 256
        vis_h = torch.relu(self.vis_conv(visual_input))
        vis_h = self.avg_pool(vis_h)  # batchsize * 512 * 16
        return vis_h  # batchsize * 512 * 16

class SequentialFrameAttentionPool(nn.Module):

    def __init__(self, cfg):
        super(SequentialFrameAttentionPool, self).__init__()
        input_size = cfg.INPUT_SIZE  # 4096
        self.hidden_size = cfg.HIDDEN_SIZE  # 512
        kernel_size = cfg.KERNEL_SIZE  # 16
        self.stride = cfg.STRIDE  # 16
        self.sqn = cfg.SQN_NUM
        att_hidden_size = 256

        self.vis_conv = nn.Conv1d(input_size, self.hidden_size, 1, 1)
        self.avg_pool = nn.AvgPool1d(kernel_size, self.stride)

        self.global_emb_fn = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.sqn)])
        self.guide_emb_fn = nn.Sequential(*[
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.ReLU()
        ])

        self.att_fn1 = nn.Linear(self.hidden_size, att_hidden_size)
        self.att_fn2 = nn.Linear(self.hidden_size, att_hidden_size)
        self.att_fn3 = nn.Linear(att_hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)
        # self.drop = nn.Dropout()

        self.vis_out_conv = nn.Conv1d(self.hidden_size * self.sqn, self.hidden_size, 1, 1)

    def forward(self, visual_input):
        B, _, v_len = visual_input.shape
        vis_h = torch.relu(self.vis_conv(visual_input))

        avg_vis = self.avg_pool(vis_h)  # batchsize * 512 * 16

        seg_list = []
        att_seg_list = []
        for i in range(v_len // self.stride):
            vis_seg = vis_h[:, :, self.stride * i: self.stride * (i + 1)].transpose(1, 2)  # batchsize * 16 * 512
            avg_seg = avg_vis[:, :, i]
            prev_se = avg_seg.new_zeros(B, self.hidden_size)

            sqn_list = []
            att_list = []
            for m in range(self.sqn):
                v_n = self.global_emb_fn[m](avg_seg)
                g_n = torch.relu(self.guide_emb_fn(torch.cat([v_n, prev_se], dim=1)))  # batchsize * 512

                att = torch.tanh(self.att_fn1(g_n).unsqueeze(1).expand(-1, self.stride, -1) + self.att_fn2(vis_seg))
                att = self.att_fn3(att)

                att = self.softmax(att)  # batchsize * 16 * 1
                # att = torch.sigmoid(att) * 2 - 1

                prev_se = torch.sum(vis_seg * att, dim=1)  # batchsize * 512
                sqn_list.append(prev_se)
                att_list.append(att)

            vis_new = torch.cat(sqn_list, dim=1)
            seg_list.append(vis_new)
            att_seg_list.append(torch.cat(att_list, dim=2))  # batchsize  * 16 * sqn

        vis_out = torch.relu(self.vis_out_conv(torch.stack(seg_list, dim=2)))
        att_out = torch.stack(att_seg_list, dim=1)  # batchsize * 16 * 16 * sqn

        return vis_out, att_out