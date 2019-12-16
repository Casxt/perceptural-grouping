import math
import torch
from typing import Collection

from io import BytesIO

from PIL import Image
from matplotlib.pyplot import imsave
from torchvision.transforms import transforms


def cat_in_out_gt(i, o, gt):
    gt = torch.cat((gt, gt, gt), 0)
    o = torch.cat((o, o, o), 0)
    return torch.cat((i, o, gt), 2)


def to_device(device, *tensors: Collection[torch.Tensor]):
    l = []
    for t in tensors:
        l.append(t.cuda(device))
    return l


def generator_img_by_node_feature(indeces: torch.Tensor, nodes_feature: torch.Tensor, node_prop: torch.Tensor,
                                  dist_map: torch.Tensor):
    """
    @param indeces: 1, 2000
    @param nodes_feature: 2000, 256
    @param node_prop: 3, 75, 100 has_edge, y, x
    @param dist_map: 2000, 2000 dist between node
    @return:
    """
    img = torch.zeros((600, 800))
    node_sets = []
    # 分组
    for index, node in enumerate(nodes_feature):
        min_dist_group = None
        min_dist = 1.
        # 找到距离最近的组
        for node_set in node_sets:
            dist = float(dist_map[node_set, :][:, index].min())
            if dist < min_dist:
                min_dist_group = node_set
                min_dist = dist
        if min_dist_group is not None:
            min_dist_group.append(index)
        else:
            node_sets.append([index])

    for i, node_set in enumerate(node_sets):
        for node_index in node_set:
            index = int(indeces[node_index])
            # 基本坐标
            ny, nx = (math.floor(index / 100)), (index % 100)
            # 偏移量
            by, bx = float(node_prop[1, ny, nx]), float(node_prop[2, ny, nx])
            by = (by + 1) / 2 * 8
            bx = (bx + 1) / 2 * 8
            y, x = round(ny * 8 + by), round(nx * 8 + bx)
            img[y - 1:y + 1, x - 1:x + 1] = i + 1
    buf = BytesIO()
    imsave(buf, img.numpy(), format='bmp')
    pil_im = Image.open(buf, 'r')
    return transforms.ToTensor()(pil_im).to(indeces.device)
