# PROJECT: FRDet
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-11-23 16:00
import json
import os
import sys
from copy import deepcopy
import random

import torch
from torchvision.transforms import transforms
from tqdm import tqdm

from models.FRDet import FRDet

from utils.dataset import CocoDataset


def trainer(
        # 基础参数
        way=5, shot=2, query_batch=16, is_cuda=True,
        # 设备参数
        random_seed=None, gpu_index=0,
        # 数据集参数
        root=None, json_path=None,
        # 模型
        model: FRDet = None,
        # 训练轮数
        max_iteration=60000,
        # 继续训练, 如果没有seed可能很难完美续上之前的训练, 不过整个流程随机, 可能也可以
        continue_iteration=None, continue_weight=None,
        # 保存相关的参数
        save_root=None
):
    r"""

    :param way:
    :param shot:
    :param query_batch:
    :param is_cuda:
    :param random_seed:
    :param gpu_index:
    :param root:
    :param json_path:
    :param model:
    :param max_iteration:
    :param continue_iteration:
    :param continue_weight:
    :param save_root:
    :return:
    """
    # 检查参数
    assert root is not None, "root is None"
    assert json_path is not None, "json_path is none"
    assert (continue_iteration is None and continue_weight is None) \
           or (continue_iteration is not None and continue_weight is not None), \
        "continue_iteration and continue_weight should be all None, or all not None"
    # 设置
    torch.set_printoptions(sci_mode=False)
    if random_seed is not None:
        random.seed(random_seed)

    # 设置参数
    torch.cuda.set_device(gpu_index)

    # 生成数据集
    fsod = CocoDataset(root=root, ann_path=json_path, img_path='images',
                       way=way, shot=shot, query_batch=query_batch, is_cuda=is_cuda)

    # 模型
    if model is None:
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

    if is_cuda:
        model.cuda()

    # 保存相关的参数
    save_weights = os.path.join(save_root, 'weights')
    save_results = os.path.join(save_root, 'results')
    save_train_loss = os.path.join(save_results, 'train_loss.json')
    save_val_loss = os.path.join(save_results, 'val_loss.json')
    # 创建文件夹保存此次训练
    if not os.path.exists(save_weights):
        os.makedirs(save_weights)
    if not os.path.exists(save_results):
        os.makedirs(save_results)
    # 保存loss
    if not os.path.exists(save_train_loss):
        with open(save_train_loss, 'w') as f:
            json.dump({}, f)
    if not os.path.exists(save_val_loss):
        with open(save_val_loss, 'w') as f:
            json.dump({}, f)

    # 训练轮数
    if continue_iteration is not None and continue_weight is not None:
        iteration = continue_iteration - 1
        weight = torch.load(continue_weight)
        model.load_state_dict(weight['models'])
    else:
        iteration = 1

    while iteration < max_iteration:
        if iteration < 56000:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.9, weight_decay=0.0005)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0002, momentum=0.9, weight_decay=0.0005)

        # 训练一个轮回
        fsod.initial()
        model.train()
        loss_dict_train = {}
        fsod.set_mode(is_training=True)
        pbar = tqdm(fsod)
        for index, item in enumerate(pbar):
            loss_this_iteration = {}
            if iteration > max_iteration:
                break
            support, bg, query, query_anns, cat_ids = item

            # 训练
            fsod.set_mode(is_training=True)
            result = model.forward(support, query, bg, targets=query_anns)
            losses = 0

            for k, v in result.items():
                losses += v
                loss_this_iteration.update({k: float(v)})
            loss_this_iteration = {iteration: loss_this_iteration}
            loss_dict_train.update(loss_this_iteration)

            postfix = {'iteration': '{}/{}'.format(iteration, max_iteration),
                       'mission': '{:3}/{:3}'.format(index + 1, len(pbar)),
                       'catIds': cat_ids,
                       '模式': 'train',
                       '损失': "%.6f" % float(losses)}
            pbar.set_postfix(postfix)
            if torch.isnan(losses).any():
                print('梯度炸了!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                sys.exit(0)
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            # 保存loss与权重
            if (iteration) % 100 == 0:  # 记得改
                with open(save_train_loss, 'r') as f:
                    tmp_loss_dict = json.load(f)
                with open(save_train_loss, 'w') as f:
                    tmp_loss_dict.update(loss_dict_train)
                    loss_dict_train = {}
                    json.dump(tmp_loss_dict, f)
                torch.save({'models': model.state_dict()},
                           os.path.join(save_weights, 'FRDet_{}.pth'.format(iteration)))
            iteration += 1

        # 验证一个轮回
        fsod.set_mode(is_training=False)
        pbar = tqdm(fsod)
        loss_dict_val = {}
        for index, item in enumerate(pbar):
            loss_this_epoch = {}
            support, bg, query, query_anns, cat_ids = item
            # 训练
            result = model.forward(support, query, bg, targets=query_anns)
            losses = 0
            for k, v in result.items():
                losses += v
                loss_this_epoch.update({k: float(v)})
            loss_this_epoch = {index + 1: loss_this_epoch}
            loss_dict_val.update(loss_this_epoch)

            postfix = {'mission': '{:3}/{:3}'.format(index + 1, len(pbar)),
                       'catIds': cat_ids,
                       '模式': 'val',
                       '损失': "%.6f" % float(losses)}
            pbar.set_postfix(postfix)

            # 保存loss
            if (index + 1) % 100 == 0:  # 记得改
                with open(save_val_loss, 'r') as f:
                    tmp_loss_dict = json.load(f)
                with open(save_val_loss, 'w') as f:
                    tmp_loss_dict.update(loss_dict_val)
                    loss_dict_val = {}
                    json.dump(tmp_loss_dict, f)
