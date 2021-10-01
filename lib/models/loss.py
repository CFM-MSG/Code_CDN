import torch
import torch.nn.functional as F
import torch.nn as nn
from core.config import config


def bce_rescale_loss(scores, masks, targets, cfg, need_target=False):
    min_iou, max_iou, bias = cfg.MIN_IOU, cfg.MAX_IOU, cfg.BIAS
    target_prob = (targets - min_iou) * (1 - bias) / (max_iou - min_iou)
    target_prob[target_prob > 0] += bias
    target_prob[target_prob > 1] = 1
    target_prob[target_prob < 0] = 0
    # scores = torch.where(torch.isnan(scores), torch.zeros_like(scores), scores)
    if config.TRAIN.FP16:
        joint_prob = scores.masked_fill(masks.eq(0.0), float('-inf'))
        loss = F.binary_cross_entropy_with_logits(joint_prob, target_prob, reduction='none') * masks
    else:
        joint_prob = torch.sigmoid(scores) * masks
        assert torch.sum(torch.isnan(joint_prob).int()) == 0
        loss = F.binary_cross_entropy(joint_prob, target_prob, reduction='none') * masks
    loss_value = torch.sum(loss) / torch.sum(masks)
    if need_target:
        joint_prob = [joint_prob, target_prob * masks]
    return loss_value, joint_prob


def att_dis_loss(att, cfg):
    '''
    attention distance loss
    :param att: batchsize * seg_num(16) * seg_len(16) * sqn(3)
    :return: loss
    '''

    lamb = cfg.LAMBDA
    beta = cfg.BETA

    B, seg_num, seg_len, sqn = att.shape
    att = att.view(-1, seg_len, sqn)

    att_f = torch.matmul(att, att.transpose(1, 2)) - lamb * torch.eye(seg_len).cuda()

    loss = torch.sum(torch.norm(att_f, dim=(1, 2)) ** 2) / seg_num

    return loss / B * beta
