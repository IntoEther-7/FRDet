# PROJECT: FRDet
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-11-17 18:54
import random

import torch
from torchvision.models.detection.anchor_utils import AnchorGenerator

from models.FRDet import FRDet
from utils.trainer import trainer

torch.set_printoptions(sci_mode=False)

if __name__ == '__main__':
    trainer(# 基础参数
            way=5, shot=5, query_batch=4, is_cuda=True,
            # 设备参数
            random_seed=114514, gpu_index=0,
            # 数据集参数
            root='../FRNOD/datasets/fsod',
            json_path='annotations/fsod_train.json',
            # 模型
            model=None,
            # 训练轮数
            max_iteration=60000,
            # 继续训练参数
            continue_iteration=None, continue_weight=None,
            # 保存相关的参数
            save_root='tetssssss')
