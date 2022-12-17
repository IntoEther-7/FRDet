# PROJECT: FRDet
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-11-21 15:52
import os
import random

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data.dataset import Dataset, T_co
from torchvision.transforms import transforms
from tqdm import tqdm

# VOC   ---------------------------------------------------------------------
# base_ids, novel_ids = [1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20], [3, 6, 10, 14, 18] split1
# base_ids, novel_ids = [2, 3, 4, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 19, 20], [1, 5, 10, 13, 18] split2
# base_ids, novel_ids = [1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 19, 20], [4, 8, 14, 17, 18] split3
# {1: {'supercategory': 'none', 'id': 1, 'name': 'aeroplane'},
#  2: {'supercategory': 'none', 'id': 2, 'name': 'bicycle'},
#  3: {'supercategory': 'none', 'id': 3, 'name': 'bird'},
#  4: {'supercategory': 'none', 'id': 4, 'name': 'boat'},
#  5: {'supercategory': 'none', 'id': 5, 'name': 'bottle'},
#  6: {'supercategory': 'none', 'id': 6, 'name': 'bus'},
#  7: {'supercategory': 'none', 'id': 7, 'name': 'car'},
#  8: {'supercategory': 'none', 'id': 8, 'name': 'cat'},
#  9: {'supercategory': 'none', 'id': 9, 'name': 'chair'},
#  10: {'supercategory': 'none', 'id': 10, 'name': 'cow'},
#  11: {'supercategory': 'none', 'id': 11, 'name': 'diningtable'},
#  12: {'supercategory': 'none', 'id': 12, 'name': 'dog'},
#  13: {'supercategory': 'none', 'id': 13, 'name': 'horse'},
#  14: {'supercategory': 'none', 'id': 14, 'name': 'motorbike'},
#  15: {'supercategory': 'none', 'id': 15, 'name': 'person'},
#  16: {'supercategory': 'none', 'id': 16, 'name': 'pottedplant'},
#  17: {'supercategory': 'none', 'id': 17, 'name': 'sheep'},
#  18: {'supercategory': 'none', 'id': 18, 'name': 'sofa'},
#  19: {'supercategory': 'none', 'id': 19, 'name': 'train'},
#  20: {'supercategory': 'none', 'id': 20, 'name': 'tvmonitor'}}

# COCO  ---------------------------------------------------------------------
# base_ids = [8, 10, 11, 13, 14, 15, 22, 23, 24, 25,
#             27, 28, 31, 32, 33, 34, 35, 36, 37, 38,
#             39, 40, 41, 42, 43, 46, 47, 48, 49, 50,
#             51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
#             61, 65, 70, 73, 74, 75, 76, 77, 78, 79,
#             80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
# novel_ids =  [2, 5, 9, 16, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 7, 72, 63]
# {1: {'supercategory': 'person', 'id': 1, 'name': 'person'},
#  2: {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'},
#  3: {'supercategory': 'vehicle', 'id': 3, 'name': 'car'},
#  4: {'supercategory': 'vehicle', 'id': 4, 'name': 'motorcycle'},
#  5: {'supercategory': 'vehicle', 'id': 5, 'name': 'airplane'},
#  6: {'supercategory': 'vehicle', 'id': 6, 'name': 'bus'},
#  7: {'supercategory': 'vehicle', 'id': 7, 'name': 'train'},
#  8: {'supercategory': 'vehicle', 'id': 8, 'name': 'truck'},
#  9: {'supercategory': 'vehicle', 'id': 9, 'name': 'boat'},
#  10: {'supercategory': 'outdoor', 'id': 10, 'name': 'traffic light'},
#  11: {'supercategory': 'outdoor', 'id': 11, 'name': 'fire hydrant'},
#  13: {'supercategory': 'outdoor', 'id': 13, 'name': 'stop sign'},
#  14: {'supercategory': 'outdoor', 'id': 14, 'name': 'parking meter'},
#  15: {'supercategory': 'outdoor', 'id': 15, 'name': 'bench'},
#  16: {'supercategory': 'animal', 'id': 16, 'name': 'bird'},
#  17: {'supercategory': 'animal', 'id': 17, 'name': 'cat'},
#  18: {'supercategory': 'animal', 'id': 18, 'name': 'dog'},
#  19: {'supercategory': 'animal', 'id': 19, 'name': 'horse'},
#  20: {'supercategory': 'animal', 'id': 20, 'name': 'sheep'},
#  21: {'supercategory': 'animal', 'id': 21, 'name': 'cow'},
#  22: {'supercategory': 'animal', 'id': 22, 'name': 'elephant'},
#  23: {'supercategory': 'animal', 'id': 23, 'name': 'bear'},
#  24: {'supercategory': 'animal', 'id': 24, 'name': 'zebra'},
#  25: {'supercategory': 'animal', 'id': 25, 'name': 'giraffe'},
#  27: {'supercategory': 'accessory', 'id': 27, 'name': 'backpack'},
#  28: {'supercategory': 'accessory', 'id': 28, 'name': 'umbrella'},
#  31: {'supercategory': 'accessory', 'id': 31, 'name': 'handbag'},
#  32: {'supercategory': 'accessory', 'id': 32, 'name': 'tie'},
#  33: {'supercategory': 'accessory', 'id': 33, 'name': 'suitcase'},
#  34: {'supercategory': 'sports', 'id': 34, 'name': 'frisbee'},
#  35: {'supercategory': 'sports', 'id': 35, 'name': 'skis'},
#  36: {'supercategory': 'sports', 'id': 36, 'name': 'snowboard'},
#  37: {'supercategory': 'sports', 'id': 37, 'name': 'sports ball'},
#  38: {'supercategory': 'sports', 'id': 38, 'name': 'kite'},
#  39: {'supercategory': 'sports', 'id': 39, 'name': 'baseball bat'},
#  40: {'supercategory': 'sports', 'id': 40, 'name': 'baseball glove'},
#  41: {'supercategory': 'sports', 'id': 41, 'name': 'skateboard'},
#  42: {'supercategory': 'sports', 'id': 42, 'name': 'surfboard'},
#  43: {'supercategory': 'sports', 'id': 43, 'name': 'tennis racket'},
#  44: {'supercategory': 'kitchen', 'id': 44, 'name': 'bottle'},
#  46: {'supercategory': 'kitchen', 'id': 46, 'name': 'wine glass'},
#  47: {'supercategory': 'kitchen', 'id': 47, 'name': 'cup'},
#  48: {'supercategory': 'kitchen', 'id': 48, 'name': 'fork'},
#  49: {'supercategory': 'kitchen', 'id': 49, 'name': 'knife'},
#  50: {'supercategory': 'kitchen', 'id': 50, 'name': 'spoon'},
#  51: {'supercategory': 'kitchen', 'id': 51, 'name': 'bowl'},
#  52: {'supercategory': 'food', 'id': 52, 'name': 'banana'},
#  53: {'supercategory': 'food', 'id': 53, 'name': 'apple'},
#  54: {'supercategory': 'food', 'id': 54, 'name': 'sandwich'},
#  55: {'supercategory': 'food', 'id': 55, 'name': 'orange'},
#  56: {'supercategory': 'food', 'id': 56, 'name': 'broccoli'},
#  57: {'supercategory': 'food', 'id': 57, 'name': 'carrot'},
#  58: {'supercategory': 'food', 'id': 58, 'name': 'hot dog'},
#  59: {'supercategory': 'food', 'id': 59, 'name': 'pizza'},
#  60: {'supercategory': 'food', 'id': 60, 'name': 'donut'},
#  61: {'supercategory': 'food', 'id': 61, 'name': 'cake'},
#  62: {'supercategory': 'furniture', 'id': 62, 'name': 'chair'},
#  63: {'supercategory': 'furniture', 'id': 63, 'name': 'couch'},
#  64: {'supercategory': 'furniture', 'id': 64, 'name': 'potted plant'},
#  65: {'supercategory': 'furniture', 'id': 65, 'name': 'bed'},
#  67: {'supercategory': 'furniture', 'id': 67, 'name': 'dining table'},
#  70: {'supercategory': 'furniture', 'id': 70, 'name': 'toilet'},
#  72: {'supercategory': 'electronic', 'id': 72, 'name': 'tv'},
#  73: {'supercategory': 'electronic', 'id': 73, 'name': 'laptop'},
#  74: {'supercategory': 'electronic', 'id': 74, 'name': 'mouse'},
#  75: {'supercategory': 'electronic', 'id': 75, 'name': 'remote'},
#  76: {'supercategory': 'electronic', 'id': 76, 'name': 'keyboard'},
#  77: {'supercategory': 'electronic', 'id': 77, 'name': 'cell phone'},
#  78: {'supercategory': 'appliance', 'id': 78, 'name': 'microwave'},
#  79: {'supercategory': 'appliance', 'id': 79, 'name': 'oven'},
#  80: {'supercategory': 'appliance', 'id': 80, 'name': 'toaster'},
#  81: {'supercategory': 'appliance', 'id': 81, 'name': 'sink'},
#  82: {'supercategory': 'appliance', 'id': 82, 'name': 'refrigerator'},
#  84: {'supercategory': 'indoor', 'id': 84, 'name': 'book'},
#  85: {'supercategory': 'indoor', 'id': 85, 'name': 'clock'},
#  86: {'supercategory': 'indoor', 'id': 86, 'name': 'vase'},
#  87: {'supercategory': 'indoor', 'id': 87, 'name': 'scissors'},
#  88: {'supercategory': 'indoor', 'id': 88, 'name': 'teddy bear'},
#  89: {'supercategory': 'indoor', 'id': 89, 'name': 'hair drier'},
#  90: {'supercategory': 'indoor', 'id': 90, 'name': 'toothbrush'}}
base_ids_voc1, novel_ids_voc1 = [1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20], [3, 6, 10, 14, 18]
base_ids_voc2, novel_ids_voc2 = [2, 3, 4, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 19, 20], [1, 5, 10, 13, 18]
base_ids_voc3, novel_ids_voc3 = [1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 19, 20], [4, 8, 14, 17, 18]
base_ids_coco = [8, 10, 11, 13, 14, 15, 22, 23, 24, 25,
                 27, 28, 31, 32, 33, 34, 35, 36, 37, 38,
                 39, 40, 41, 42, 43, 46, 47, 48, 49, 50,
                 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                 61, 65, 70, 73, 74, 75, 76, 77, 78, 79,
                 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
novel_ids_coco = [2, 5, 9, 16, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 7, 72, 63]


class CocoDataset(Dataset):

    def __init__(self, root, ann_path, img_path,
                 way=5, shot=2, query_batch=16, is_cuda=False, catIds=None):
        super(CocoDataset, self).__init__()

        self.root = root
        self.ann_path = os.path.join(root, ann_path)
        self.img_path = os.path.join(root, img_path)
        self.way = way
        self.shot = shot
        self.query_batch = query_batch
        self.is_cuda = is_cuda

        self.coco = COCO(self.ann_path)
        if catIds is not None:
            self.split_base_novel(catIds)

        self.delete_bad_annotations()
        # 所有类划分为测试集和验证集
        self.train_img, self.val_img = self.split_train_val()

        self.train_mission = None
        self.val_mission = None
        self.train_iteration = None
        self.val_iteration = None

        # 将训练集和验证集划分mission和iteration
        self.initial()

        self.training = True
        self.iteration = self.train_iteration

    def split_base_novel(self, catIds):
        # 删除对象有, anns, catToImgs, cats, imgToAnns, imgs
        # 确定要删除的类
        delete_cat_ids = [i for i in list(self.coco.cats.keys()) if i not in catIds]
        # 先根据类别确定要删除的标注
        delete_anns_ids = self.coco.getAnnIds(catIds=delete_cat_ids)
        # 根据要删除的标注确定受影响的图像
        # 删除图像到标注的映射, 如果删除后, 图像没有标注, 则移除
        for ann_id in delete_anns_ids:
            img_id = self.coco.loadAnns(ann_id)[0]['image_id']
            ann = self.coco.loadAnns(ann_id)[0]
            self.coco.imgToAnns[img_id].remove(ann)
            if len(self.coco.imgToAnns[img_id]) == 0:
                self.coco.imgToAnns.pop(img_id)
                self.coco.imgs.pop(img_id)

        # 删除类到图像的映射
        # 删除类别信息
        for i in delete_cat_ids:
            self.coco.catToImgs.pop(i)
            self.coco.cats.pop(i)
        # 删除标注
        for i in delete_anns_ids:
            self.coco.anns.pop(i)

    def initial(self):
        # 将多个类划分为任务
        self.train_mission, self.val_mission = self.split_mission()
        # 将任务划分为每次迭代的数据
        self.train_iteration = self.split_iteration(self.train_mission)
        self.val_iteration = self.split_iteration(self.val_mission)
        self.set_mode(is_training=True)

    def delete_bad_annotations(self):
        r"""
        在划分数据集前先把一些错误标注的图像/标注根据情况删除
        :return:
        """
        delete_anns = []
        image_influenced = []
        delete_image = []
        for k, v in tqdm(self.coco.anns.items()):
            _, _, w, h = v['bbox']
            if w <= 1 or h <= 1:
                # 统计要删除的标注
                image_id = v['image_id']
                image_influenced.append(image_id)
                delete_anns.append(k)
                # self.coco.anns.pop(k)
                # tqdm.write('删除id为{}的标注'.format(k))
                # 删除图像对标注的映射
                self.coco.imgToAnns[image_id].remove(v)
                # tqdm.write('删除imgToAnns的映射')
                category_id = v['category_id']
                # 什么时候该删除cat2imgs的映射?
                # 图的标注没有该类
                if len(self.coco.getAnnIds(image_id, category_id)) == 0:
                    self.coco.catToImgs[category_id].remove(image_id)
                # 如果图像没有标注了, 删除图像
                if len(self.coco.getAnnIds(imgIds=[image_id])) == 0:
                    delete_image.append(image_id)
                    # tqdm.write('删除此图像')
        for k in delete_anns:
            self.coco.anns.pop(k)
        tqdm.write('删除标注{}\n影响的图像{}\n删除的图像{}'.format(delete_anns, image_influenced, delete_image))

    def set_mode(self, is_training):
        r"""
        设置训练模式, 根据情况返回训练样本
        :param is_training:
        :return:
        """
        if is_training:
            self.training = True
            self.iteration = self.train_iteration
        else:
            self.training = False
            self.iteration = self.val_iteration

    def __getitem__(self, index) -> T_co:
        this_iteration = self.iteration[index]
        support, bg, query, query_anns, query_anns_way = self.id2item(this_iteration)
        cat_ids = this_iteration['cat_ids']
        if self.is_cuda:
            support = [s.cuda() for s in support]
            bg = [b.cuda() for b in bg]
            query = [q.cuda() for q in query]
        return support, bg, query, (query_anns, query_anns_way), cat_ids

    def get_val(self, index) -> T_co:
        this_iteration = self.val_iteration[index]
        support, bg, query, query_anns, query_anns_way = self.id2item(this_iteration)
        cat_ids = this_iteration['cat_ids']
        if self.is_cuda:
            support = [s.cuda() for s in support]
            bg = [b.cuda() for b in bg]
            query = [q.cuda() for q in query]
        return support, bg, query, (query_anns, query_anns_way), cat_ids

    def __len__(self):
        return len(self.iteration)

    def split_train_val(self):
        r"""
        将数据集分为训练集和测试集, 对每个类别分别进行分, 不同类别中可能含有相同图像, 因为同一张图像可能包含多个类别
        :return: 不返回
        """
        train_img = {}
        val_img = {}
        for cat_id in self.coco.cats.keys():
            img_ids = self.coco.getImgIds(catIds=[cat_id])
            img_num = len(img_ids)
            random.shuffle(img_ids)
            train_list = img_ids[:int(img_num * 0.7)]
            val_list = img_ids[int(img_num * 0.7):]
            train_img.update({cat_id: train_list})
            val_img.update({cat_id: val_list})
        return train_img, val_img

    def split_mission(self):
        r"""
        随机way个类, 每个类shot实例, batch个query,
        先选取batch个query, 从剩下的里面选取way * shot个实例
        :param img_dict:
        :return:
        """
        mission_train = []
        mission_val = []
        cat_ids = list(self.coco.cats.keys())
        random.shuffle(cat_ids)
        mission_num = len(cat_ids) // self.way
        for i in range(mission_num):
            mission_train_img = []
            mission_val_img = []
            mission_cat = cat_ids[self.way * i: self.way * (i + 1)]
            for j in mission_cat:
                train_img = self.train_img[j]
                val_img = self.val_img[j]
                mission_train_img.extend(train_img)
                mission_val_img.extend(val_img)
            random.shuffle(mission_train_img)
            random.shuffle(mission_val_img)
            this_mission_train = {'cat_ids': mission_cat, 'img_ids': mission_train_img}
            this_mission_val = {'cat_ids': mission_cat, 'img_ids': mission_val_img}
            mission_train.append(this_mission_train)
            mission_val.append(this_mission_val)
        return mission_train, mission_val

    def split_iteration(self, mission):
        iteration = []
        for i in mission:
            cat_ids = i['cat_ids']
            img_ids = i['img_ids']
            num_iteration = len(img_ids) // self.query_batch + int(len(img_ids) % self.query_batch != 0)
            for j in range(num_iteration):
                img_this_iteration = img_ids[self.query_batch * j: self.query_batch * (j + 1)]
                cat_this_iteration = cat_ids
                ann_this_iteration = [self.coco.getAnnIds(imgIds=[img_id], catIds=cat_ids) for img_id in
                                      img_this_iteration]
                support_ids = []
                for k in cat_ids:
                    s_img = random.sample(self.train_img[k], self.shot)
                    anns = [self.coco.getAnnIds(imgIds=[s], catIds=[k]) for s in s_img]
                    for ann in anns:
                        support_ids.extend(random.sample(ann, 1))
                iteration.append({'support_label_ids': support_ids,
                                  'query_ids': img_this_iteration,
                                  'query_label_ids': ann_this_iteration,
                                  'cat_ids': cat_this_iteration})
        random.shuffle(iteration)
        return iteration

    def id2item(self, this_iteration):
        r"""
        读取数据, 并预处理
        :param this_iteration: 这个iteration的数据
        :return:
        """
        support_label_ids = this_iteration['support_label_ids']
        query_ids = this_iteration['query_ids']
        query_label_ids = this_iteration['query_label_ids']
        cat_ids = this_iteration['cat_ids']

        # support
        support, bg = self.crop_support_bg(support_label_ids, cat_ids)

        # query_anns
        skip_img_list = []  # 防止错误标注的图像进入训练
        query_labels = [self.coco.loadAnns(ids) for ids in query_label_ids]
        # [Dict{'boxes': tensor(n, 4), 'labels': tensor(n,)}, 'image_id': int, 'category_id': list(int), 'id': int]
        query_anns = []
        query_anns_group_by_way = [[] for i in cat_ids]

        for label_this_img in query_labels:
            boxes = []
            labels = []
            category_id = []
            ann_ids = []
            image_id = []
            # way
            boxes_way = [[] for i in cat_ids]
            labels_way = [[] for i in cat_ids]
            category_id_way = [[] for i in cat_ids]
            ann_ids_way = [[] for i in cat_ids]
            image_id_way = [[] for i in cat_ids]
            for ann in label_this_img:
                # 单个ann
                if ann['category_id'] in cat_ids:
                    # boxes
                    x1, y1, w, h = ann['bbox']
                    x2, y2 = x1 + w, y1 + h
                    boxes.append([x1, y1, x2, y2])
                    # category_id
                    category_id.append(ann['category_id'])
                    # labels
                    labels.append(cat_ids.index(ann['category_id']) + 1)
                    ann_ids.append(ann['id'])
                    image_id.append(ann['image_id'])
                    # 按照way分
                    way_index = cat_ids.index(ann['category_id'])
                    category_id_way[way_index].append(ann['category_id'])
                    boxes_way[way_index].append([x1, y1, x2, y2])
                    labels_way[way_index].append(way_index + 1)
                    ann_ids_way[way_index].append(ann['id'])
                    image_id_way[way_index].append(ann['image_id'])
            if self.is_cuda:
                query_anns.append({'boxes': torch.tensor(boxes).cuda(),
                                   'labels': torch.tensor(labels, dtype=torch.int64).cuda(),
                                   'image_id': image_id,
                                   'category_id': category_id,
                                   'ann_ids': ann_ids})
                for i, bw in enumerate(boxes_way):
                    query_anns_group_by_way[i].append({'boxes': torch.tensor(boxes_way[i]).cuda(),
                                                       'labels': torch.tensor(labels_way[i],
                                                                              dtype=torch.int64).cuda(),
                                                       'image_id': image_id_way[i],
                                                       'category_id': category_id_way[i],
                                                       'ann_ids': ann_ids_way[i]})

            else:
                query_anns.append({'boxes': torch.tensor(boxes),
                                   'labels': torch.tensor(labels, dtype=torch.int64),
                                   'image_id': image_id,
                                   'category_id': category_id,
                                   'ann_ids': ann_ids})
                for i, bw in enumerate(boxes_way):
                    query_anns_group_by_way[i].append({'boxes': torch.tensor(boxes_way[i]),
                                                       'labels': torch.tensor(labels_way[i],
                                                                              dtype=torch.int64),
                                                       'image_id': image_id_way[i],
                                                       'category_id': category_id_way[i],
                                                       'ann_ids': ann_ids_way[i]})

        # query
        query_info = self.coco.loadImgs(query_ids)
        query = []
        t_q = transforms.ToTensor()
        for info in query_info:
            path = os.path.join(self.img_path, info['file_name'])
            q = t_q(Image.open(path).convert('RGB'))
            query.append(q)

        return support, bg, query, query_anns, query_anns_group_by_way

    def crop_support_bg(self, support_label_ids, cat_ids, is_show=False):
        r"""
        处理支持集, 并得到背景
        :param support_label_ids:
        :param cat_ids:
        :param is_show:
        :return:
        """
        support_ann = self.coco.loadAnns(support_label_ids)
        support = []
        bg = []
        s_t = transforms.Compose([transforms.ToTensor(),
                                  transforms.Resize(size=(320, 320)),
                                  transforms.Pad(padding=16, fill=0, padding_mode='constant')])
        bg_t = transforms.ToTensor()
        for ann in support_ann:
            # info
            img_id = ann['image_id']
            img_info = self.coco.loadImgs(ids=[img_id])[0]
            img_path = os.path.join(self.img_path, img_info['file_name'])

            # crop support
            x, y, w, h = ann['bbox']
            img = Image.open(img_path).convert('RGB')
            img = img.crop((x, y, x + w, y + h))

            # get background
            bg_img = Image.open(img_path).convert('RGB')
            bg_img = bg_t(bg_img)
            bg_ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            for bg_ann in self.coco.loadAnns(ids=bg_ann_ids):
                if bg_ann['category_id'] in cat_ids:
                    x, y, w, h = bg_ann['bbox']
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    bg_img[:, y:y + h, x:x + w] = 0.

            # 可视化
            if is_show:
                img.save('{}.jpg'.format(img_id))
                bg_tt = transforms.ToPILImage()
                bg_tt(bg_img).save('{}_bg.jpg'.format(img_id))

            support.append(s_t(img))
            bg.append(bg_img)
        return support, bg


if __name__ == '__main__':
    # 生成数据集
    # root = '../../FRNOD/datasets/fsod'
    # train_json = 'annotations/fsod_train.json'
    # test_json = 'annotations/fsod_test.json'
    # fsod = CocoDataset(root=root, ann_path=test_json, img_path='images', way=5, shot=2, is_cuda=True)
    # for support, bg, query, query_anns, cat_ids in fsod:
    #     print()
    random.seed(1)
    # coco
    # root = '../../FRNOD/datasets/coco'
    # train_json = 'annotations/instances_train2017.json'
    # test_json = 'annotations/instances_val2017.json'
    # fsod = CocoDataset(root=root, ann_path=train_json, img_path='train2017', way=20, shot=5, is_cuda=True)

    # voc
    root = '../../FRNOD/datasets/voc/VOCdevkit/VOC2012'
    train_json = 'cocoformatJson/voc_2012_train.json'
    val_json = 'cocoformatJson/voc_2012_trainval.json'
    test_json = 'cocoformatJson/voc_2012_val.json'
    dataset = CocoDataset(root=root, ann_path=train_json, img_path='train2017',
                          way=5, shot=5, is_cuda=True, catIds=novel_ids_voc1)

    for support, bg, query, query_anns, cat_ids in tqdm(dataset):
        pass
