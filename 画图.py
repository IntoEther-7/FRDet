# PROJECT: FRDet
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-11-22 18:03
import json
import os.path

import numpy as np
import torch
from matplotlib import pyplot as plt

# 每个mission的loss
from tqdm import tqdm

root = 'results_fsod_r50/results'
loss_path = os.path.join(root, 'train_loss.json')

if __name__ == '__main__':
    statistics_loss = {}
    loss_iteration_list = []
    with open(loss_path, 'r') as f:
        loss_dict = json.load(f)
    for losses in loss_dict.values():
        loss_this_iteration = 0
        for k, v in losses.items():
            if k not in statistics_loss.keys():
                statistics_loss[k] = []
            statistics_loss[k].append(v)
            loss_this_iteration += v
        loss_iteration_list.append(loss_this_iteration)
    statistics_loss.update({'loss_sum': loss_iteration_list})
    for k, v in statistics_loss.items():
        fig = plt.figure(figsize=(12, 6.75), dpi=320.)
        plt.plot(v, linewidth=0.5)
        plt.legend('train_loss')
        plt.draw()
        plt.savefig(os.path.join(root, '{}.png'.format(k)))
