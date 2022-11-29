# PROJECT: FRDet
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-11-17 18:54

import torch
from utils.trainer import trainer

torch.set_printoptions(sci_mode=False)
root = '../FRNOD/datasets/voc/VOCdevkit/VOC2012'
json_path = 'cocoformatJson/voc_2012_train.json'
img_path = 'JPEGImages'


def way_shot_train(way, shot):
    lr = 2e-04
    save_root = '/data/chenzh/FRDet/flatten_model_r50/result_voc_r50_{}way_{}shot_lr{}'.format(way, shot, lr)
    trainer(
        # 基础参数
        way=way, shot=shot, query_batch=16, is_cuda=True, lr=lr,
        # 设备参数
        random_seed=None, gpu_index=1,
        # 数据集参数
        root=root,
        json_path=json_path,
        img_path=img_path,
        # 模型
        model=None,
        # 训练轮数
        max_iteration=10000,
        # 继续训练参数
        continue_iteration=None, continue_weight=None,
        # 保存相关的参数
        save_root=save_root)


if __name__ == '__main__':
    way_shot_train(2, 5)
    way_shot_train(2, 1)
    way_shot_train(5, 5)
    way_shot_train(1, 1)
