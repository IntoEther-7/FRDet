# -*- coding:utf-8 -*-
"""
@Editor: Ether
@Time: 2023/1/10 10:02
@File: test_fcos
@Contact: 211307040003@hhu.edu.cn
@Version: 1.0
@Description: None
"""
import torch.cuda
from tqdm import tqdm

from fcos_pytorch.fr_fcos import FR_FCOS, collate_fn_train
from utils.dataset import CocoDataset, base_ids_voc1, novel_ids_voc1

root = '../FRNOD/datasets/voc/VOCdevkit/VOC2012'
json_path = 'cocoformatJson/voc_2012_train.json'
img_path = 'JPEGImages'
loss_weights0 = {'loss_classifier': 1, 'loss_box_reg': 1,
                 'loss_objectness': 1, 'loss_rpn_box_reg': 1,
                 'loss_attention': 1, 'loss_aux': 1}

if __name__ == '__main__':
    way = 5
    shot = 5
    torch.cuda.set_device(0)
    model = FR_FCOS(way, shot).cuda()

    dataset = CocoDataset(root=root, ann_path=json_path, img_path=img_path,
                          way=way, shot=shot, query_batch=16,
                          is_cuda=True, catIds=base_ids_voc1)
    dataset.initial()

    pbar = tqdm(dataset)
    for index, item in enumerate(pbar):
        support, bg, query, query_anns, cat_ids = item
        model.eval()
        losses, result = model.forward(query, targets=query_anns, support=support)
        # losses['total_loss'].backward()
        # model.eval()
        # losses, result = model.forward(query, targets=query_anns)
        tqdm.write('')
