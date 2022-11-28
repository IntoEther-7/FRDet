# PROJECT: FRDet
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-11-17 18:54

import torch

from utils.tester import tester
from utils.trainer import trainer

torch.set_printoptions(sci_mode=False)
root = '../FRNOD/datasets/fsod'
json_path = 'annotations/fsod_train.json'
img_path = 'images'


def way_shot_test(way, shot):
    save_root = '/data/chenzh/FRDet/result_fsod_r50_{}way_{}shot'.format(way, shot)
    continue_weight = 'FRDet_60000.pth'
    tester(
        # 基础参数
        way=way, shot=shot, query_batch=16, is_cuda=True,
        # 设备参数
        random_seed=None, gpu_index=0,
        # 数据集参数
        root=root,
        json_path=json_path,
        img_path=img_path,
        # 模型
        model=None,
        # 权重文件
        continue_weight=continue_weight,
        # 保存相关的参数
        save_root=save_root)


if __name__ == '__main__':
    way_shot_test(1, 1)
