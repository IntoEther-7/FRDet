# PROJECT: FRDet
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-11-17 21:31
from torchvision.transforms import transforms

from utils.data.dataset import FsodDataset
from utils.data.process import pre_process_coco, transform_query, transform_anns, transform_support, cat_list


def load_data(dataset: FsodDataset, catIds, support_size=320, is_cuda=False):
    s, q, q_anns, val, val_anns = dataset.n_way_k_shot(catIds)
    # s, q, val: List[List[ImagePath]], 类别[某类的图像[路径1, 路径2, ...]]
    # ann: List[List[List[单个标注]]], 类别[某类的图像[图像的标注[标注1, 标注2, ...]]]
    s, q, q_anns, val, val_anns \
        = pre_process_coco(s, q, q_anns, val, val_anns,
                           support_transforms=transforms.Compose(
                               [transforms.ToTensor(),
                                transforms.Resize(support_size)]),
                           query_transforms=transforms.Compose(
                               [transforms.ToTensor()]),
                           is_cuda=is_cuda, random_sort=True)
    return s, q, q_anns, val, val_anns


def read_batch(q, qa, label_ori,
                     query_transforms=transforms.Compose([transforms.ToTensor()]), is_cuda=False):
    q = transform_query(q, query_transforms, is_cuda)
    qa = transform_anns(qa, is_cuda, label_ori)
    return q, qa


def pre_process_coco(support: list, query: list, query_anns: list, val: list, val_anns: list,
                     support_transforms=transforms.Compose([transforms.ToTensor(),
                                                            transforms.Resize((320, 320))]),
                     query_transforms=transforms.Compose([transforms.ToTensor()]), is_cuda=False, random_sort=False):
    r"""
    图像处理, 转换成tensor, s_c, s_n为tensor[shot, channel, 320, 320], q_c为[tensor, tensor, ...],
    gt_bboxes为[标注列表[每张图像的标注[每个盒子的参数]]],
    labels为[标注列表[每张图像的标签[每个盒子的标签]]]
    :param support: 支持图, [PIL.Image]
    :param query: 查询图,
    :param query_anns: 标注
    :param support_transforms:
    :param query_transforms:
    :param support_n:
    :return: 如果有s_n, 则返回s_c, s_n, q_c, gt_bboxes, labels, 否则返回s_c, q_c, gt_bboxes, labels
    """
    s_c_list = []
    for s in support:
        s_c_list.extend(transform_support(s, support_transforms, is_cuda))
    # q_c_list = [transform_query(q, query_transforms, is_cuda) for q in query]
    # q_anns_list = [transform_anns(query_anns[i], is_cuda, i + 1) for i in range(len(query_anns))]
    # q_c_list, q_anns_list = cat_list(q_c_list, q_anns_list, random_sort)
    q_c_list, q_anns_list = cat_list(query, query_anns, random_sort)
    # val_list = [transform_query(v, query_transforms, is_cuda) for v in val]
    # val_anns_list = [transform_anns(val_anns[i], is_cuda, i + 1) for i in range(len(val_anns))]
    # val_list, val_anns_list = cat_list(val_list, val_anns_list, random_sort)
    val_list, val_anns_list = cat_list(val, val_anns, random_sort)
    return s_c_list, q_c_list, q_anns_list, val_list, val_anns_list