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
from utils.data.dataload import load_data, read_batch
from utils.data.dataset import FsodDataset
from utils.data.process import read_single_coco

if __name__ == '__main__':
    # 设置
    is_cuda = True
    torch.set_printoptions(sci_mode=False)
    # random.seed(1096)

    # 设置参数
    torch.cuda.set_device(1)
    way = 5
    shot = 2
    query_batch = 8

    # 训练参数
    epoch = 20
    fine_epoch = int(epoch * 0.7)
    pass

    # 生成数据集
    root = '../FRNOD/datasets/fsod'
    train_json = 'annotations/fsod_train.json'
    test_json = 'annotations/fsod_test.json'
    train_json = os.path.join(root, train_json)
    test_json = os.path.join(root, test_json)
    fsod = FsodDataset(root, train_json, support_shot=shot, dataset_img_path='images')
    if not (os.path.exists('weights_fsod_r50/results')):
        os.makedirs('weights_fsod_r50/results')

    # 整理类别信息
    cat_list = deepcopy(list(fsod.coco.cats.keys()))
    random.shuffle(cat_list)
    num_mission = len(cat_list) // way

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
        box_batch_size_per_image=64, box_positive_fraction=0.25,
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

    # 训练
    for e in range(0, epoch):  # 4 * 60000 * 8
        if continue_epoch is not None and continue_epoch_done is False:
            if e < continue_epoch - 1:
                tqdm.write('skip epoch {}'.format(e + 1))
                continue
            elif e == continue_epoch - 1:
                continue_epoch_done = True
        if e + 1 < fine_epoch:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.9, weight_decay=0.0005)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0002, momentum=0.9, weight_decay=0.0005)

        for i in range(num_mission):
            # 准备训练数据
            catIds = cat_list[i * way:(i + 1) * way]
            s, s_anns, q, q_anns, val, val_anns, bg = load_data(fsod, catIds, support_size=(320, 320), is_cuda=is_cuda)
            # 训练
            model.train()
            pbar = tqdm(range(len(q) // query_batch))
            for index in pbar:
                query = q[index * query_batch: (index + 1) * query_batch]
                target = q_anns[index * query_batch: (index + 1) * query_batch]
                for tmp_index, t in enumerate(target):
                    for tar in t:
                        bbox = tar[0]['bbox']
                        if bbox[2] <= 0.1 or bbox[3] <= 0.1:
                            tqdm.write('id为{}的标注存在问题, 对应image_id为{}, 跳过此张图像'.format(tar[0]['id'], tar[0]['image_id']))
                            target.pop(tmp_index)
                            query.pop(tmp_index)
                query, target = read_batch(query, target, label_ori=catIds, query_transforms=transforms.Compose(
                    [transforms.ToTensor()]), is_cuda=is_cuda)
                result = model.forward(s, query, bg, targets=target)
                losses = 0
                for loss in result.values():
                    losses += loss
                postfix = {'epoch': '{:3}/{:3}'.format(e + 1, epoch), 'mission': '{:3}/{:3}'.format(i + 1, num_mission),
                           'catIds': catIds,
                           '模式': 'trian', '损失': "%.6f" % float(loss)}
                pbar.set_postfix(postfix)
                if torch.isnan(loss).any():
                    print('梯度炸了!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    sys.exit(0)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
