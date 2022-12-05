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
        way=5, shot=2, query_batch=16, is_cuda=True, lr=2e-02,
        # 设备参数
        random_seed=None, gpu_index=0,
        # 数据集参数
        root=None, json_path=None, img_path=None,
        # 模型
        model: FRDet = None,
        # 训练轮数
        max_epoch=20,
        # 继续训练, 如果没有seed可能很难完美续上之前的训练, 不过整个流程随机, 可能也可以
        continue_epoch=None, continue_iteration=None, continue_weight=None,
        # 保存相关的参数
        save_root=None,
        # loss权重
        loss_weights=None
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
    if loss_weights is None:
        loss_weights = {'loss_classifier': 0.995, 'loss_box_reg': 0.005,
                        'loss_objectness': 0.995, 'loss_rpn_box_reg': 0.005,
                        'loss_attention': 0.95, 'loss_aux': 0.05}
    assert root is not None, "root is None"
    assert json_path is not None, "json_path is none"
    assert img_path is not None, "img_path is none"
    assert (continue_iteration is None and continue_epoch is None) \
           or (continue_iteration is not None and continue_epoch is not None), \
        "continue_iteration and continue_epoch should be all None, or all not None"
    # 设置
    torch.set_printoptions(sci_mode=False)
    if random_seed is not None:
        random.seed(random_seed)

    # 设置参数
    torch.cuda.set_device(gpu_index)

    # 生成数据集
    dataset = CocoDataset(root=root, ann_path=json_path, img_path=img_path,
                          way=way, shot=shot, query_batch=query_batch, is_cuda=is_cuda)

    # 模型
    if model is None:
        model = FRDet(
            # box_predictor params
            way, shot, roi_size=7, num_classes=way + 1,
            # backbone
            backbone_name='resnet50', pretrained=False,
            returned_layers=None, trainable_layers=4,
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

    # 创建文件夹保存此次训练
    if not os.path.exists(save_weights):
        os.makedirs(save_weights)
    if not os.path.exists(save_results):
        os.makedirs(save_results)

    # 训练轮数
    if continue_epoch is not None and continue_iteration is not None:
        continue_weight = os.path.join(save_root, 'weights',
                                       'FRDet_{}_{}.pth'.format(continue_epoch, continue_iteration))
        weight = torch.load(continue_weight)
        model.load_state_dict(weight['models'])
        continue_done = False
    else:
        continue_epoch = 0
        continue_iteration = 0
        continue_done = True

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0005)

    warm_up_epoch = int(max_epoch * 0.1)
    warm_up_start_lr = lr / 10
    warm_up_end_lr = lr
    delta = (warm_up_end_lr - warm_up_start_lr) / (warm_up_epoch * len(dataset))
    lr_this_iteration = warm_up_start_lr

    fine_epoch = int(max_epoch * 0.7)
    val_losses = 0
    for epoch in range(max_epoch):
        if epoch + 1 < continue_epoch and continue_done is False:
            continue

        save_train_loss = os.path.join(save_results, 'train_loss_{}.json'.format(epoch + 1))
        save_val_loss = os.path.join(save_results, 'val_loss_{}.json'.format(epoch + 1))
        # 保存loss
        if not os.path.exists(save_train_loss):
            with open(save_train_loss, 'w') as f:
                json.dump({}, f)
        if not os.path.exists(save_val_loss):
            with open(save_val_loss, 'w') as f:
                json.dump({}, f)

        # 训练一个轮回
        dataset.initial()
        model.train()
        loss_dict_train = {}
        loss_dict_val = {}
        dataset.set_mode(is_training=True)
        pbar = tqdm(dataset)
        for index, item in enumerate(pbar):

            # warm up
            if epoch + 1 <= warm_up_epoch:
                lr_this_iteration += delta
                optimizer = torch.optim.SGD(model.parameters(), lr=lr_this_iteration, momentum=0.9, weight_decay=0.0005)
            elif epoch + 1 <= fine_epoch:
                lr_this_iteration = lr
                optimizer = torch.optim.SGD(model.parameters(), lr=lr_this_iteration, momentum=0.9, weight_decay=0.0005)
            else:
                lr_this_iteration = lr / 10
                optimizer = torch.optim.SGD(model.parameters(), lr=lr_this_iteration, momentum=0.9, weight_decay=0.0005)

            # lr_this_iteration = lr / 10
            # optimizer = torch.optim.SGD(model.parameters(), lr=lr_this_iteration, momentum=0.9, weight_decay=0.0005)

            iteration = index + 1
            if iteration < continue_iteration and continue_done is False:
                continue
            elif iteration == continue_iteration:
                continue_done = True
            loss_this_iteration = {}
            val_loss_this_iteration = {}
            support, bg, query, query_anns, cat_ids = item

            # 训练
            result = model.forward(support, query, bg, targets=query_anns)
            losses = 0

            sum_weights = 0
            for k, v in result.items():
                w = loss_weights[k]
                losses += v * w
                sum_weights += w
                loss_this_iteration.update({k: float(v)})
            tqdm.write('{:2} / {:3} / {:.6f} / {}'.format(epoch + 1, iteration, (float(losses)), result))
            losses = losses / sum_weights
            loss_this_iteration = {iteration: loss_this_iteration}
            loss_dict_train.update(loss_this_iteration)

            if torch.isnan(losses).any() or torch.isinf(losses).any():
                print('梯度炸了!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                sys.exit(0)
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            # 验证
            if (index + 1) % 5 == 0:
                support, bg, query, query_anns, cat_ids = dataset.get_val(
                    random.randint(1, len(dataset.val_iteration)) - 1)
                result = model.forward(support, query, bg, targets=query_anns)
                val_losses = 0
                sum_weights = 0
                for k, v in result.items():
                    w = loss_weights[k]
                    val_losses += v * w
                    sum_weights += w
                    val_loss_this_iteration.update({k: float(v)})
                val_losses = val_losses / sum_weights
                loss_this_epoch = {index + 1: val_loss_this_iteration}
                loss_dict_val.update(loss_this_epoch)
                # 信息展示
            postfix = {'epoch': '{:2}/{:2}'.format(epoch + 1, max_epoch),
                       'mission': '{:4}/{:4}'.format(index + 1, len(pbar)),
                       'catIds': cat_ids,
                       '模式': 'train',
                       'lr': optimizer.state_dict()['param_groups'][0]['lr'],
                       'train_loss': "%.6f" % float(losses),
                       'val_loss': "%.6f" % float(val_losses)}

            pbar.set_postfix(postfix)

            # 保存loss与权重
            if iteration % 100 == 0 or (index + 1) == len(dataset):  # 记得改
                with open(save_train_loss, 'r') as f:
                    tmp_loss_dict = json.load(f)
                with open(save_train_loss, 'w') as f:
                    tmp_loss_dict.update(loss_dict_train)
                    loss_dict_train = {}
                    json.dump(tmp_loss_dict, f)
                torch.save({'models': model.state_dict()},
                           os.path.join(save_weights, 'FRDet_{}_{}.pth'.format(epoch + 1, iteration)))

                with open(save_val_loss, 'r') as f:
                    tmp_loss_dict = json.load(f)
                with open(save_val_loss, 'w') as f:
                    tmp_loss_dict.update(loss_dict_val)
                    loss_dict_val = {}
                    json.dump(tmp_loss_dict, f)

            iteration += 1
