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

root = 'result/flatten_model_r50/result_voc_r50_2way_1shot_lr0.0002/results'
loss_path = os.path.join(root, 'train_loss.json')
val_loss_path = os.path.join(root, 'val_loss.json')

if __name__ == '__main__':
    statistics_loss = {}
    val_statistics_loss = {}
    loss_iteration_list = []
    val_loss_iteration_list = []
    with open(loss_path, 'r') as f:
        loss_dict = json.load(f)
    with open(val_loss_path, 'r') as f:
        val_loss_dict = json.load(f)
    for losses in loss_dict.values():
        loss_this_iteration = 0
        for k, v in losses.items():
            if k not in statistics_loss.keys():
                statistics_loss[k] = []
            statistics_loss[k].append(v)
            loss_this_iteration += v
        loss_iteration_list.append(loss_this_iteration / len(statistics_loss))
    for losses in val_loss_dict.values():
        loss_this_iteration = 0
        for k, v in losses.items():
            if k not in val_statistics_loss.keys():
                val_statistics_loss[k] = []
            val_statistics_loss[k].append(v)
            loss_this_iteration += v
        val_loss_iteration_list.append(loss_this_iteration / len(val_statistics_loss))
    val_statistics_loss.update({'loss_sum': val_loss_iteration_list})
    for k, v in statistics_loss.items():
        epoch = 400  # 处理
        v = [np.array(v[i * epoch:(i + 1) * epoch]).mean() for i in range(len(v) // epoch)]  # 处理
        val_v = val_statistics_loss[k]
        val_v = [np.array(val_v[i * epoch:(i + 1) * epoch]).mean() for i in range(len(val_v) // epoch)]  # 处理
        fig = plt.figure(figsize=(12, 6.75), dpi=320.)
        plt.plot(v, linewidth=1)
        plt.plot(val_v, linewidth=2)
        plt.legend('train_loss, val_loss')
        plt.draw()
        plt.savefig(os.path.join(root, '{}.png'.format(k)))
