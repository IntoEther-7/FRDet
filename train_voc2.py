# PROJECT: FRDet
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-11-17 18:54

import torch
from utils.trainer import trainer
import threading

torch.set_printoptions(sci_mode=False)
root = '../FRNOD/datasets/voc/VOCdevkit/VOC2012'
json_path = 'cocoformatJson/voc_2012_train.json'
img_path = 'JPEGImages'

loss_weights1 = {'loss_classifier': 0.95, 'loss_box_reg': 0.05,
                 'loss_objectness': 0.95, 'loss_rpn_box_reg': 0.05,
                 'loss_attention': 0.95, 'loss_aux': 0.05}
loss_weights2 = {'loss_classifier': 0.995, 'loss_box_reg': 0.005,
                 'loss_objectness': 0.995, 'loss_rpn_box_reg': 0.005,
                 'loss_attention': 0.95, 'loss_aux': 0.05}
loss_weights3 = {'loss_classifier': 0.9995, 'loss_box_reg': 0.0005,
                 'loss_objectness': 0.9995, 'loss_rpn_box_reg': 0.0005,
                 'loss_attention': 0.95, 'loss_aux': 0.05}


def way_shot_train(way, shot, lr, loss_weights, gpu_index):
    save_root = '/data/chenzh/FRDet/not_flatten_model_loss_weight/result_voc_r50_{}way_{}shot_lr{}_loss_weight_{}' \
        .format(way, shot, lr, gpu_index)
    trainer(
        # 基础参数
        way=way, shot=shot, query_batch=16, is_cuda=True, lr=lr,
        # 设备参数
        random_seed=None, gpu_index=gpu_index,
        # 数据集参数
        root=root,
        json_path=json_path,
        img_path=img_path,
        # 模型
        model=None,
        # 训练轮数
        max_epoch=50,
        # 继续训练参数
        continue_epoch=25, continue_iteration=300, continue_weight=None,
        # 保存相关的参数
        save_root=save_root,
        # loss权重
        loss_weights=loss_weights
    )


def train1():
    way_shot_train(2, 5, 2e-06, loss_weights1, 0)
    way_shot_train(2, 5, 2e-05, loss_weights1, 0)
    way_shot_train(2, 5, 2e-04, loss_weights1, 0)
    way_shot_train(2, 5, 2e-03, loss_weights1, 0)
    way_shot_train(2, 5, 2e-02, loss_weights1, 0)


def train2():
    way_shot_train(2, 5, 2e-06, loss_weights2, 1)
    way_shot_train(2, 5, 2e-05, loss_weights2, 1)
    way_shot_train(2, 5, 2e-04, loss_weights2, 1)
    way_shot_train(2, 5, 2e-03, loss_weights2, 1)
    way_shot_train(2, 5, 2e-02, loss_weights2, 1)


if __name__ == '__main__':
    train2()
