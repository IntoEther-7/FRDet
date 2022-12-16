# PROJECT: FRDet
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-11-17 18:54
import random

import torch
from torchvision.models.detection.anchor_utils import AnchorGenerator

from utils.dataset import *
from models.FRDet import FRDet
from utils.trainer_without_loss_weight import trainer

torch.set_printoptions(sci_mode=False)
root = '../FRNOD/datasets/coco'
json_path = 'annotations/instances_train2017.json'
img_path = 'train2017'
loss_weights0 = {'loss_classifier': 1, 'loss_box_reg': 1,
                 'loss_objectness': 1, 'loss_rpn_box_reg': 1,
                 'loss_attention': 1, 'loss_aux': 1}
loss_weights1 = {'loss_classifier': 0.1, 'loss_box_reg': 1,
                 'loss_objectness': 1, 'loss_rpn_box_reg': 1,
                 'loss_attention': 1, 'loss_aux': 0.1}
loss_weights无监督attention = {'loss_classifier': 1, 'loss_box_reg': 1,
                            'loss_objectness': 1, 'loss_rpn_box_reg': 1,
                            'loss_attention': 0, 'loss_aux': 1}


def way_shot_train(way, shot, lr, loss_weights, gpu_index, loss_weights_index):
    save_root = '/data/chenzh/FRDet/not_flatten_model_{}/result_coco_r50_{}way_{}shot_lr{}' \
        .format(loss_weights_index, way, shot, lr)
    model = FRDet(
        # box_predictor params
        way, shot, roi_size=7, num_classes=way + 1,
        # backbone
        backbone_name='resnet50', pretrained=True,
        returned_layers=None, trainable_layers=3,
        # transform parameters
        min_size=600, max_size=1000,
        image_mean=None, image_std=None,
        # RPN parameters
        rpn_anchor_generator=AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)),
        rpn_head=None,
        rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=2000,
        rpn_post_nms_top_n_train=1000, rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
        rpn_score_thresh=0.0,
        # Box parameters
        box_roi_pool=None, box_head=None, box_predictor=None,
        box_score_thresh=0.05, box_nms_thresh=0.3, box_detections_per_img=100,
        box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512, box_positive_fraction=0.25,
        bbox_reg_weights=(10., 10., 5., 5.)
    )

    trainer(
        # 基础参数
        way=way, shot=shot, query_batch=16, is_cuda=True, lr=lr,
        # 设备参数
        random_seed=None, gpu_index=gpu_index,
        # 数据集参数
        root=root,
        json_path=json_path,
        img_path=img_path,
        split_cats=base_ids_coco,
        # 模型
        model=model,
        # 训练轮数
        max_epoch=8,
        # 继续训练参数
        continue_epoch=None, continue_iteration=None, continue_weight=None,
        # 保存相关的参数
        save_root=save_root,
        # loss权重
        loss_weights=loss_weights
    )


def train0():
    # way_shot_train(2, 5, 2e-01, loss_weights0, 1, 0)
    way_shot_train(2, 5, 2e-02, loss_weights0, 1, 0)
    way_shot_train(2, 5, 2e-03, loss_weights0, 1, 0)
    way_shot_train(2, 5, 2e-04, loss_weights0, 1, 0)
    way_shot_train(2, 5, 2e-05, loss_weights0, 1, 0)
    way_shot_train(2, 5, 2e-06, loss_weights0, 1, 0)


if __name__ == '__main__':
    # train0()
    # way_shot_train(2, 5, 2e-01, loss_weights0, 0, 0)
    random.seed(1024)
    # 20221208 上午
    # way_shot_train(2, 5, 2e-03, loss_weights0, 0, '20221208')
    # 20221208 下午四点半
    # way_shot_train(2, 5, 2e-03, loss_weights0, 0, '20221208_减少roi数量')
    # 20221209 下午两点
    # way_shot_train(5, 5, 2e-03, loss_weights0, 0, '20221208_减少roi数量')
    # 20221211 上午十点
    # way_shot_train(2, 5, 2e-03, loss_weights0, 1, '20221210_增加rpn_batch_size_per_image')
    # 20221214 晚上
    way_shot_train(2, 5, 2e-03, loss_weights无监督attention, 0, '20221213_无监督attention')
