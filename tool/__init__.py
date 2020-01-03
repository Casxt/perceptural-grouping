from io import BytesIO
from typing import Collection

import math
import torch
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
    node grouping 结果可视化
    @param indeces: 1, node_num
    @param nodes_feature: node_num, 256
    @param node_prop: 3, 75, 100 has_edge, y, x
    @param dist_map: node_num, node_num dist between node
    @return:
    """
    img = torch.zeros((600, 800), device=indeces.device)
    node_sets = []
    # 分组
    for index, node in enumerate(nodes_feature):
        min_dist_group = None
        min_dist = 0.5
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
            by = ((by + 1) / 2) * 8
            bx = ((bx + 1) / 2) * 8
            y, x = round(ny * 8 + by), round(nx * 8 + bx)
            if 0 < y < 600 and 0 < x < 800:
                img[y, x] = i + 1
            else:
                pass  # print(f"out of bound y={y} x={x}")
    return render_color(img)


def render_raw_img(img: torch.Tensor, node_gt: torch.Tensor, edge_gt: torch.Tensor):
    """
    在原图上渲染视觉效果
    @param img: 3, h, w
    @param node_gt: 4 , 75 , 100   has_edge, y, x, id
    @param edge_gt: 1, h, w
    @return:
    """
    edge_img = img + edge_gt.expand(3, -1, -1)
    img = torch.zeros((600, 800), device=img.device)
    for y in range(0, 75):
        for x in range(0, 100):
            has_edge, by, bx, id = node_gt[:, y, x]
            by = ((int(by) + 1) / 2) * 8
            bx = ((int(bx) + 1) / 2) * 8
            dy, dx = round(y * 8 + by), round(x * 8 + bx)
            if 0 < y < 600 and 0 < x < 800:
                img[dy, dx] = id
    id_mask = (img > 0).expand(3, -1, -1)
    id_img = render_color(img)
    edge_img[id_mask] = id_img[id_mask]
    return edge_img


def render_color(input: torch.Tensor):
    buf = BytesIO()
    imsave(buf, input.cpu().numpy(), format='bmp')
    pil_im = Image.open(buf, 'r')
    return transforms.ToTensor()(pil_im).to(input.device)


def chunk(input: torch.Tensor, num_h, num_w):
    """
    与 pytorch.chunk 相似, 为了方便修改一些细节
    @param input: b, c, h, w
    @param num: how many chunks you need
    @return: list of chunk line by line
    """
    #  b, c, h / num , w
    b, c, h, w = input.shape
    if h % num_h != 0 or w % num_w != 0:
        raise Exception("num_h, num_w 必须可以整除输入的 h, w")
    h_len, w_len = int(h / num_h), int(w / num_w)
    chunks = []
    for y in range(num_h):
        for x in range(num_w):
            chunks.append(input[:, :, y * h_len:(y + 1) * h_len, x * w_len:(x + 1) * w_len])

    return chunks

