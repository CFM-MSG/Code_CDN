import torch
from torch import nn


class TextualEncoding(nn.Module):
    def __init__(self, cfg):
        super(TextualEncoding, self).__init__()
        self.cfg = cfg
        txt_input_size = cfg.TXT_INPUT_SIZE  # 300
        self.txt_hidden_size = cfg.TXT_HIDDEN_SIZE  # 512
        self.bidirectional = cfg.RNN.BIDIRECTIONAL

        self.textual_encoder = nn.LSTM(txt_input_size,
                                       self.txt_hidden_size,
                                       num_layers=cfg.RNN.NUM_LAYERS, bidirectional=self.bidirectional,  # 3, False
                                       batch_first=True)
        if self.bidirectional:
            self.tex_linear = nn.Linear(self.txt_hidden_size * 2, self.txt_hidden_size)

    def forward(self, x, textual_mask):
        self.textual_encoder.flatten_parameters()
        txt_h = self.textual_encoder(x)[0] * textual_mask  # batch * seq_len * txt_hidden_size

        if self.bidirectional:
            shape = txt_h.shape
            txt_h = txt_h.view(shape[0], shape[1], 2, self.txt_hidden_size)
            # txt_h = torch.stack(
            #     [torch.cat([txt_h[i][torch.sum(mask).long() - 1][0], txt_h[i][0][1]], dim=0) for i, mask in
            #      enumerate(textual_mask)])
            txt_h = torch.stack(
                [torch.cat([txt_h[i][torch.sum(mask).long() - 1][0], txt_h[i][0][1]], dim=0) for i, mask in
                 enumerate(textual_mask)])
            txt_h = self.tex_linear(txt_h)  # batchsize * 512
        else:
            txt_h = torch.stack(
                [txt_h[i][torch.sum(mask).long() - 1] for i, mask in enumerate(textual_mask)])
        return txt_h  # batch * txt_hidden_size


class WLTextualEncoding(nn.Module):
    def __init__(self, cfg):
        super(WLTextualEncoding, self).__init__()
        self.cfg = cfg
        txt_input_size = cfg.TXT_INPUT_SIZE  # 300
        self.txt_hidden_size = cfg.TXT_HIDDEN_SIZE  # 512
        self.bidirectional = cfg.RNN.BIDIRECTIONAL

        self.textual_encoder = nn.GRU(txt_input_size,
                                       self.txt_hidden_size,
                                       num_layers=cfg.RNN.NUM_LAYERS, bidirectional=self.bidirectional,  # 3, False
                                       batch_first=True)

    def forward(self, x, textual_mask):
        '''
        Bi LSTM
        :param x: text batch * seq_len * input_size
        :param textual_mask: batch * seq_len
        :return:
        '''
        self.textual_encoder.flatten_parameters()

        text_length = torch.sum(textual_mask, dim=1).int().squeeze(1)  # batch
        sorted_lengths, indices = torch.sort(text_length, descending=True)
        x = x[indices]
        inv_ix = indices.clone()
        inv_ix[indices] = torch.arange(0, len(indices)).type_as(inv_ix)

        packed = nn.utils.rnn.pack_padded_sequence(x, sorted_lengths.data.tolist(), batch_first=True)

        out = self.textual_encoder(packed)[0]  # batch * seq_len * txt_hidden_size

        padded = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        cap_emb, cap_len = padded
        cap_emb = cap_emb[inv_ix]
        # cap_len = cap_len[inv_ix]

        if self.bidirectional:
            cap_emb = cap_emb.view(cap_emb.size(0), cap_emb.size(1), 2, -1)
            # txt_h = torch.stack([cap_emb[i, l-1, 0, :] for i, l in enumerate(cap_len)]) + cap_emb[:, 0, 1, :]

            cap_emb = (cap_emb[:, :, 0, :] + cap_emb[:, :, 1, :]) / 2  # batch * seq_len * txt_hidden_size

            txt_h = torch.sum(cap_emb, dim=1) / text_length.unsqueeze(1)

            # txt_h = torch.norm(txt_h, dim=1)

        return txt_h, cap_emb  # batch * txt_hidden_size


class GRUTextualEncoding(nn.Module):
    def __init__(self, cfg):
        super(GRUTextualEncoding, self).__init__()
        self.cfg = cfg
        txt_input_size = cfg.TXT_INPUT_SIZE  # 300
        self.txt_hidden_size = cfg.TXT_HIDDEN_SIZE  # 512
        self.bidirectional = cfg.RNN.BIDIRECTIONAL

        self.textual_encoder = nn.GRU(txt_input_size,
                                      self.txt_hidden_size // 2 if self.bidirectional else self.txt_hidden_size,
                                      num_layers=cfg.RNN.NUM_LAYERS, bidirectional=self.bidirectional,  # 3, False
                                      batch_first=True)

    def forward(self, x, textual_mask):
        self.textual_encoder.flatten_parameters()
        txt_h = self.textual_encoder(x)[0] * textual_mask  # batch * seq_len * txt_hidden_size

        if self.bidirectional:
            shape = txt_h.shape
            txt_h = txt_h.view(shape[0], shape[1], 2, self.txt_hidden_size // 2)
            txt_h = torch.stack(
                [torch.cat([txt_h[i][torch.sum(mask).long() - 1][0], txt_h[i][0][1]], dim=0) for i, mask in
                 enumerate(textual_mask)])
        else:
            txt_h = torch.stack(
                [txt_h[i][torch.sum(mask).long() - 1] for i, mask in enumerate(textual_mask)])
        return txt_h  # batch * txt_hidden_size
