from io import BytesIO
from typing import Collection

import torch
from PIL import Image
from matplotlib.pyplot import imsave
from torchvision.transforms import transforms

from .CitySpeaceEdgeGrouping import EdgeGroupingDataset


def to_device(device, *tensors: Collection[torch.Tensor]):
    l = []
    for t in tensors:
        l.append(t.cuda(device))
    return l

def render_color(input: torch.Tensor):
    buf = BytesIO()
    imsave(buf, input.cpu().numpy(), format='bmp')
    pil_im = Image.open(buf, 'r')
    return transforms.ToTensor()(pil_im).to(input.device)
