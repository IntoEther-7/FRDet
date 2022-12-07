# PROJECT: FRDet
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-11-17 18:54

import torch

from models.FRDet import FRDet
from utils.tester import tester
from utils.trainer import trainer

torch.set_printoptions(sci_mode=False)
root = '../FRNOD/datasets/fsod'
json_path = 'annotations/fsod_train.json'
img_path = 'images'
continue_weight = 'FRDet_28_100.pth'

def way_shot_test(way, shot, lr, index):
    model = FRDet(
        # box_predictor params
        way, shot, roi_size=7, num_classes=way + 1,
        # backbone
        backbone_name='resnet18', pretrained=True,
        returned_layers=None, trainable_layers=3,
        # transform parameters
        min_size=600, max_size=1000,
        image_mean=None, image_std=None,
        # RPN parameters
        rpn_anchor_generator=None, rpn_head=None,
        rpn_pre_nms_top_n_train=12000, rpn_pre_nms_top_n_test=6000,
        rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=500,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
        rpn_score_thresh=0.0,
        # Box parameters
        box_roi_pool=None, box_head=None, box_predictor=None,
        box_score_thresh=0.05, box_nms_thresh=0.3, box_detections_per_img=100,
        box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
        box_batch_size_per_image=100, box_positive_fraction=0.25,
        bbox_reg_weights=(10., 10., 5., 5.)
    )
    save_root = '/data/chenzh/FRDet/not_flatten_model_loss_weight_{}/result_fsod_r50_{}way_{}shot_lr{}' \
        .format(index, way, shot, lr)
    tester(
        # 基础参数
        way=way, shot=shot, query_batch=1, is_cuda=True,
        # 设备参数
        random_seed=None, gpu_index=1,
        # 数据集参数
        root=root,
        json_path=json_path,
        img_path=img_path,
        # 模型
        model=model,
        # 权重文件
        continue_weight=continue_weight,
        # 保存相关的参数
        save_root=save_root)


if __name__ == '__main__':
    way_shot_test(2, 5, 2e-03, 0)
