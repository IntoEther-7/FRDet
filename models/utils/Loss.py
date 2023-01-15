# -*- coding:utf-8 -*-
"""
@Editor: Ether
@Time: 2023/1/15 11:41
@File: Loss
@Contact: 211307040003@hhu.edu.cn
@Version: 1.0
@Description: None
"""
import torch
from torch import nn

def n_pair_loss(distance, dis_gt):
    loss = torch.log((dis_gt - distance).exp().sum())
    loss.requires_grad = True
    return loss

class NPairLoss(nn.Module):
    """
    N-Pair Loss
    """

    def __init__(self):
        super(NPairLoss, self).__init__()

    def forward(self, distance, dis_gt):
        """

        :param distance: FR的距离矩阵
        :param labels: 筛选的gt labels
        :return:
        """
        loss = torch.log((dis_gt - distance).exp().sum())
        loss.requires_grad = True
        return loss