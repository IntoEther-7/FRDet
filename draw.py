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

root = 'result/result_fsod_5way_5shot/results'
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
        loss_iteration_list.append(loss_this_iteration / len(statistics_loss))
    statistics_loss.update({'loss_sum': loss_iteration_list})
    for k, v in statistics_loss.items():
        epoch = 2000  # 处理
        v = [np.array(v[i * epoch:(i + 1) * epoch]).mean() for i in range(len(v) // epoch)]  # 处理
        fig = plt.figure(figsize=(12, 6.75), dpi=320.)
        plt.plot(v, linewidth=1)
        plt.legend('train_loss')
        plt.draw()
        plt.savefig(os.path.join(root, '{}.png'.format(k)))
