import torch
from torch import nn
from core.config import config
import models.frame_modules as frame_modules
import models.prop_modules as prop_modules
import models.map_modules as map_modules
import models.fusion_modules as fusion_modules
import models.textual_modules as textual_modules

class CDN(nn.Module):
    def __init__(self):
        super(CDN, self).__init__()

        self.frame_layer = getattr(frame_modules, config.TAN.FRAME_MODULE.NAME)(config.TAN.FRAME_MODULE.PARAMS)
        self.prop_layer = getattr(prop_modules, config.TAN.PROP_MODULE.NAME)(config.TAN.PROP_MODULE.PARAMS)
        self.fusion_layer = getattr(fusion_modules, config.TAN.FUSION_MODULE.NAME)(config.TAN.FUSION_MODULE.PARAMS)
        self.map_layer = getattr(map_modules, config.TAN.MAP_MODULE.NAME)(config.TAN.MAP_MODULE.PARAMS)
        # self.pred_layer = nn.Conv2d(config.TAN.PRED_INPUT_SIZE, 1, 16, 1, padding=15)
        self.pred_layer = nn.Conv2d(config.TAN.PRED_INPUT_SIZE, 1, 1, 1)

        if not config.TAN.TEXTUAL_MODULE.NAME:
            self.textual_encoding = None
        else:
            self.textual_encoding = getattr(textual_modules, config.TAN.TEXTUAL_MODULE.NAME)(
                config.TAN.TEXTUAL_MODULE.PARAMS)

        if not config.TAN.FRAME_MODULE.PARAMS.ATTENTION:
            self.channel_attention = None
        else:
            self.channel_attention = getattr(frame_modules, config.TAN.FRAME_MODULE.PARAMS.ATTENTION)(config)

        if not config.TAN.FRAME_MODULE.PARAMS.SEMANTIC_ENHANCE:
            self.semantic_frame_enhance = None
        else:
            self.semantic_frame_enhance = nn.ModuleList(
                [getattr(fusion_modules, config.TAN.FRAME_MODULE.PARAMS.SEMANTIC_ENHANCE)(config)
                 for _ in range(config.TAN.FRAME_MODULE.PARAMS.SEMANTIC_ENHANCE_NUM)])

        if not config.TAN.PROP_MODULE.PARAMS.SEMANTIC_ENHANCE:
            self.semantic_map_enhance = None
        else:
            self.semantic_map_enhance = nn.ModuleList(
                [getattr(fusion_modules, config.TAN.FRAME_MODULE.PARAMS.SEMANTIC_ENHANCE)(config)
                 for _ in range(config.TAN.PROP_MODULE.PARAMS.SEMANTIC_ENHANCE_NUM)])

    def forward(self, textual_input, textual_mask, visual_input):
        if config.TAN.TEXTUAL_MODULE.NAME == 'BiTextualEncoding':
            tex_encode, word_encode = self.textual_encoding(textual_input, textual_mask)
        else:
            tex_encode = self.textual_encoding(textual_input, textual_mask)

        if config.TAN.FRAME_MODULE.NAME == 'WordAttentionPool':
            vis_h = self.frame_layer(visual_input.transpose(1, 2), tex_encode)
        else:
            vis_h, att = self.frame_layer(visual_input.transpose(1, 2))  # batchsize * 512 * 16
        if self.channel_attention is not None:
            vis_h = self.channel_attention(vis_h, tex_encode)

        if self.semantic_frame_enhance is not None:
            vis_h = vis_h.transpose(1, 2)  # batchsize * 16 * 512
            for enhance_module in self.semantic_frame_enhance:
                vis_h = enhance_module(vis_h, word_encode, textual_mask)
            vis_h = vis_h.transpose(1, 2)  # batchsize * 512 * 16

        map_h, map_mask = self.prop_layer(vis_h)  # batchsize * 512 * 16 * 16

        if self.semantic_map_enhance is not None:
            batch, hidden_size, map_l, _ = map_h.size()
            map_h = map_h.view(batch, hidden_size, -1).transpose(1, 2)  # batchsize * 256 * 512
            for enhance_module in self.semantic_map_enhance:
                map_h = enhance_module(map_h, word_encode, textual_mask)
            map_h = map_h.transpose(1, 2).view(batch, hidden_size, map_l, map_l)  # batchsize * 512 * 16 * 16

        fused_h = self.fusion_layer(tex_encode, textual_mask, map_h, map_mask)
        fused_h = self.map_layer(fused_h, map_mask, tex_encode)
        # prediction = self.pred_layer(fused_h)[:, :, 15:31, 0:16] * map_mask
        prediction = self.pred_layer(fused_h) * map_mask  # batchsize * 1 * 16 * 16

        return prediction, map_mask, att

    def extract_features(self, textual_input, textual_mask, visual_input):
        tex_encode = self.textual_encoding(textual_input)

        vis_h = self.frame_layer(visual_input.transpose(1, 2))  # batchsize * 512 * 16
        if self.channel_attention is not None:
            vis_h, attention = self.channel_attention.get_attention(vis_h, tex_encode)

        map_h, map_mask = self.prop_layer(vis_h)  # batchsize * 512 * 16 * 16
        fused_h = self.fusion_layer(tex_encode, textual_mask, map_h, map_mask)
        fused_h = self.map_layer(fused_h, map_mask)
        # prediction = self.pred_layer(fused_h)[:, :, 15:31, 0:16] * map_mask
        prediction = self.pred_layer(fused_h) * map_mask

        return fused_h, prediction, attention