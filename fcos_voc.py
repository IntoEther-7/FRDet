'''
@Author: xxxmy
@Github: github.com/VectXmy
@Date: 2019-09-26
@Email: xxxmy@foxmail.com
'''
import torch

from fcos_frdet.fcos import FCOSDetector
from models.FRDet import FRDet
from utils.trainer import trainer

torch.set_printoptions(sci_mode=False)
root = '../FRNOD/datasets/voc/VOCdevkit/VOC2012'
json_path = 'cocoformatJson/voc_2012_train.json'
img_path = 'JPEGImages'


def way_shot_train(way, shot, lr, loss_weights, gpu_index, loss_weights_index, split_cats):
    save_root = '/data/chenzh/FRDet/not_flatten_model_{}/result_voc_r50_{}way_{}shot_lr{}' \
        .format(loss_weights_index, way, shot, lr)
    model = FCOSDetector(mode="training").cuda()

    trainer(
        # 基础参数
        way=way, shot=shot, query_batch=16, is_cuda=True, lr=lr,
        # 设备参数
        random_seed=None, gpu_index=gpu_index,
        # 数据集参数
        root=root,
        json_path=json_path,
        img_path=img_path,
        split_cats=split_cats,
        # 模型
        model=model,
        # 训练轮数
        max_epoch=40,
        # 继续训练参数
        continue_epoch=None, continue_iteration=None, continue_weight=None,
        # 保存相关的参数
        save_root=save_root,
        # loss权重
        loss_weights=loss_weights
    )
