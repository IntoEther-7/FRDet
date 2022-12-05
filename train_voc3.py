# PROJECT: FRDet
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-11-17 18:54

import torch

from models.FRDet import FRDet
from models.FRHead import FRPredictHeadWithFlatten
from utils.trainer import trainer
import threading

torch.set_printoptions(sci_mode=False)
root = '../FRNOD/datasets/voc/VOCdevkit/VOC2012'
json_path = 'cocoformatJson/voc_2012_train.json'
img_path = 'JPEGImages'
loss_weights0 = {'loss_classifier': 0.5, 'loss_box_reg': 0.5,
                 'loss_objectness': 0.5, 'loss_rpn_box_reg': 0.5,
                 'loss_attention': 0.5, 'loss_aux': 0.5}
loss_weights1 = {'loss_classifier': 0.95, 'loss_box_reg': 0.05,
                 'loss_objectness': 0.95, 'loss_rpn_box_reg': 0.05,
                 'loss_attention': 0.95, 'loss_aux': 0.05}
loss_weights2 = {'loss_classifier': 0.995, 'loss_box_reg': 0.005,
                 'loss_objectness': 0.995, 'loss_rpn_box_reg': 0.005,
                 'loss_attention': 0.95, 'loss_aux': 0.05}
loss_weights3 = {'loss_classifier': 0.9995, 'loss_box_reg': 0.0005,
                 'loss_objectness': 0.9995, 'loss_rpn_box_reg': 0.0005,
                 'loss_attention': 0.5, 'loss_aux': 0.05}
loss_weights4 = {'loss_classifier': 0.99995, 'loss_box_reg': 0.00005,
                 'loss_objectness': 0.99995, 'loss_rpn_box_reg': 0.00005,
                 'loss_attention': 0.95, 'loss_aux': 0.05}
loss_weights5 = {'loss_classifier': 0.5, 'loss_box_reg': 0.5,
                 'loss_objectness': 0.5, 'loss_rpn_box_reg': 0.5,
                 'loss_attention': 0.05, 'loss_aux': 0.05}
loss_weights6 = {'loss_classifier': 0.9995, 'loss_box_reg': 0.0005,
                 'loss_objectness': 0.9995, 'loss_rpn_box_reg': 0.0005,
                 'loss_attention': 0.05, 'loss_aux': 0.05}


def way_shot_train(way, shot, lr, loss_weights, gpu_index, loss_weights_index):
    save_root = '/data/chenzh/FRDet/not_flatten_model_loss_weight_{}/result_voc_r50_{}way_{}shot_lr{}' \
        .format(loss_weights_index, way, shot, lr)
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
        max_epoch=200,
        # 继续训练参数
        continue_epoch=None, continue_iteration=None, continue_weight=None,
        # 保存相关的参数
        save_root=save_root,
        # loss权重
        loss_weights=loss_weights
    )


def train0():
    way_shot_train(2, 5, 2e-02, loss_weights0, 0, 0)
    way_shot_train(2, 5, 2e-03, loss_weights0, 0, 0)
    way_shot_train(2, 5, 2e-01, loss_weights0, 0, 0)
    way_shot_train(2, 5, 2e-04, loss_weights0, 0, 0)
    way_shot_train(2, 5, 2e-05, loss_weights0, 0, 0)
    way_shot_train(2, 5, 2e-06, loss_weights0, 0, 0)


def train1():
    way_shot_train(2, 5, 2e-06, loss_weights1, 0, 1)
    way_shot_train(2, 5, 2e-05, loss_weights1, 0, 1)
    way_shot_train(2, 5, 2e-04, loss_weights1, 0, 1)
    way_shot_train(2, 5, 2e-03, loss_weights1, 0, 1)
    way_shot_train(2, 5, 2e-02, loss_weights1, 0, 1)


def train2():
    way_shot_train(2, 5, 2e-06, loss_weights2, 1, 2)
    way_shot_train(2, 5, 2e-05, loss_weights2, 1, 2)
    way_shot_train(2, 5, 2e-04, loss_weights2, 1, 2)
    way_shot_train(2, 5, 2e-03, loss_weights2, 1, 2)
    way_shot_train(2, 5, 2e-02, loss_weights2, 1, 2)


def train3():
    way_shot_train(2, 5, 2e-02, loss_weights3, 1, 3)
    way_shot_train(2, 5, 2e-03, loss_weights3, 1, 3)
    way_shot_train(2, 5, 2e-01, loss_weights3, 1, 3)
    way_shot_train(2, 5, 2e-04, loss_weights3, 1, 3)
    way_shot_train(2, 5, 2e-05, loss_weights3, 1, 3)
    way_shot_train(2, 5, 2e-06, loss_weights3, 1, 3)


def train4():
    way_shot_train(2, 5, 2e-01, loss_weights4, 1, 4)
    way_shot_train(2, 5, 2e-02, loss_weights4, 1, 4)
    way_shot_train(2, 5, 2e-03, loss_weights4, 1, 4)
    way_shot_train(2, 5, 2e-04, loss_weights4, 1, 4)
    way_shot_train(2, 5, 2e-05, loss_weights4, 1, 4)
    way_shot_train(2, 5, 2e-06, loss_weights4, 1, 4)


if __name__ == '__main__':
    train0()

