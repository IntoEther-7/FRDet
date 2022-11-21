# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-10-16 14:29
import os.path
from copy import deepcopy

import PIL.Image
import torch
from pycocotools import coco
import random
from torchvision.transforms import transforms


# def n_way_k_shot(root, dataset: coco.COCO, catId: int, way: int, support_shot: int = 2,
#                  query_shot: int = 5):
#     sample_range = random.sample(dataset.cats.keys(), way + 1)
#     print(sample_range)
#     if catId in sample_range:
#         sample_range.remove(catId)
#     else:
#         sample_range = sample_range[:way]
#     print(sample_range)
#     pass


def one_way_k_shot(root, dataset: coco.COCO, dataset_img_path: str, catIds: list, catId: int, support_shot: int = 2,
                   quick_test=False):
    r"""
    针对某一个种类, 生成k-shot的support和query-shot的query, 并返回support, query, query的标注
    :param dataset: 数据集
    :param catId: 种类id
    :param support_shot: 几个支持集
    :param val_shot: 几个查询图像
    :return: support 已经经过裁剪的support(仅保留了box部分), query 查询图像, query_anns 查询图像的标注数据
    """
    support_imgIds, query_imgIds, val_imgIds = k_shot(dataset=dataset,
                                                      catId=catId,
                                                      support_shot=support_shot, quick_test=quick_test)

    support = []
    support_anns = []
    for imgInfo in dataset.loadImgs(ids=support_imgIds):
        imgPath = os.path.join(root, dataset_img_path, imgInfo['file_name'])
        support.append(imgPath)
        annIds = dataset.getAnnIds(imgIds=[imgInfo['id']], catIds=catIds)
        support_anns.append([dataset.loadAnns(annId) for annId in annIds])

    query = []
    query_anns = []  # type:list
    for imgInfo in dataset.loadImgs(ids=query_imgIds):
        imgPath = os.path.join(root, dataset_img_path, imgInfo['file_name'])
        # img = PIL.Image.open(imgPath).convert('RGB')
        query.append(imgPath)
        annIds = dataset.getAnnIds(imgIds=[imgInfo['id']], catIds=catIds)
        query_anns.append([dataset.loadAnns(annId) for annId in annIds])

    val = []
    val_anns = []
    for imgInfo in dataset.loadImgs(ids=val_imgIds):
        imgPath = os.path.join(root, dataset_img_path, imgInfo['file_name'])
        # img = PIL.Image.open(imgPath).convert('RGB')
        val.append(imgPath)
        annIds = dataset.getAnnIds(imgIds=[imgInfo['id']], catIds=catIds)
        val_anns.append([dataset.loadAnns(annId) for annId in annIds])
    return support, support_anns, query, query_anns, val, val_anns


def k_shot(dataset: coco.COCO, catId: int, support_shot: int = 2, quick_test=False):
    r"""
    对某一个类别, 生成支持集和查询集, 对于一类图像, 去除Support, query: val = 0.7: 0.3
    :param dataset: 利用cocoAPI生成的数据集
    :param catId: 类别的ID
    :param support_shot: 需要几个support实例, k-shot
    :return: 返回support_annIds(支持实例的ID -> 标注annID), query_imgIds(查询图像的ID)
    """
    # 该类别的所有图像
    imgIds_cat = dataset.getImgIds(catIds=[catId])
    # 选取其中部分图像
    imgIds_sample = deepcopy(imgIds_cat)
    left_num = len(imgIds_sample) - support_shot
    val_shot = int(left_num * 0.3)
    query_shot = left_num - val_shot
    random.shuffle(imgIds_sample)
    # 分离成支持图像和查询图像
    support_imgIds = imgIds_sample[:support_shot]
    query_imgIds = imgIds_sample[support_shot:support_shot + query_shot]
    val_imgIds = imgIds_sample[support_shot + query_shot:]

    if quick_test:
        query_imgIds = [query_imgIds[0]]
        val_imgIds = [val_imgIds[0]]
    return support_imgIds, query_imgIds, val_imgIds

# if __name__ == '__main__':
#     root = '../../datasets/fsod/'
#     train_json = os.path.join(root, 'annotations/fsod_train.json')
#     test_json = os.path.join(root, 'annotations/fsod_test.json')
#     fsod = coco.COCO(annotation_file=test_json)
#     catId = 20
#     random.seed(114514)
#     n_way_k_shot(root=root, dataset=fsod, catId=catId, way=5)
