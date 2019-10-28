import torch
from typing import Collection


def cat_in_out_gt(i, o, gt):
    gt = torch.cat((gt, gt, gt), 0)
    o = torch.cat((o, o, o), 0)
    return torch.cat((i, o, gt), 2)


def to_device(device, *tensors: Collection[torch.Tensor]):
    l = []
    for t in tensors:
        l.append(t.cuda(device))
    return l
