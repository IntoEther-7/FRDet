# PROJECT: FRDet
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-11-17 20:59
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

if __name__ == '__main__':
    # 设置
    is_cuda = True
    torch.set_printoptions(sci_mode=False)
    random.seed(1096)

    # 设置参数
    torch.cuda.set_device(0)
    way = 5
    shot = 2
    query_batch = 16

    # 生成数据集
    root = '../FRNOD/datasets/coco'
    train_json = 'annotations/instances_train2017.json'
    test_json = 'annotations/instances_val2017.json'
    fsod = CocoDataset(root=root, ann_path=train_json, img_path='train2017',
                       way=way, shot=shot, query_batch=query_batch, is_cuda=is_cuda)

    # 生成模型
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

    # 移入cuda
    if is_cuda:
        model.cuda()

    # 设置是否从某些地方继续训练
    # continue_weight = 'weights_coco_c4/frnod2_6.pth'
    # print('!!!!!!!!!!!!!从{}继续训练!!!!!!!!!!!!!'.format(continue_weight))
    # weight = torch.load(continue_weight)
    # model.load_state_dict(weight['models'])
    continue_epoch = 2
    continue_mission = 7
    continue_epoch_done = True
    continue_mission_done = True
    # ------------------------------------------------------------------------------------------------------------------

    # 保存的相关参数
    save_root = 'results_coco_r50_5way_2shot'
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

    # 训练
    iteration = 0
    max_iteration = 60000
    stop_flag = False

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
            loss_this_iteration = {iteration + 1: loss_this_iteration}
            loss_dict_train.update(loss_this_iteration)

            postfix = {'iteration': '{}/{}'.format(iteration + 1, max_iteration),
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
            if (iteration + 1) % 100 == 0:  # 记得改
                with open(save_train_loss, 'r') as f:
                    tmp_loss_dict = json.load(f)
                with open(save_train_loss, 'w') as f:
                    tmp_loss_dict.update(loss_dict_train)
                    loss_dict_train = {}
                    json.dump(tmp_loss_dict, f)
                torch.save({'models': model.state_dict()},
                           os.path.join(save_weights, 'FRDet_{}.pth'.format(iteration + 1)))
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
