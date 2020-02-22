import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class EdgeGroupingDataset(Dataset):

    def __init__(self, dataset_path):
        self.files = []
        for (dirpath, dirnames, filenames) in os.walk(Path(dataset_path)):
            for filename in filenames:
                self.files.append(Path(dirpath, filename))

    def __len__(self):
        return len(self.files)

    @staticmethod
    def vision_transaction_matrix(gm):
        assert isinstance(gm, torch.Tensor)
        c, h, w = gm.shape
        transaction_matrix = gm.clone().view(c, h * w)
        for i in range(h * w):
            transaction_matrix[i, i] = 0
        same_group_with = tuple(
            map(lambda x: int(x), torch.argmax(transaction_matrix, dim=1)))
        groups = []
        has_group = set()
        for i in range(h * w):
            has_group.add(i)
            if same_group_with[i] in has_group:
                for g in groups:
                    if same_group_with[i] in g:
                        g.add(i)
                        break
            else:
                has_group.add(same_group_with[i])
                groups.append({i, same_group_with[i]})
        img = torch.zeros(size=(1, h * w), device=gm.device, dtype=gm.dtype)
        for i, g in enumerate(groups):
            img[0, list(g)] = i + 1
        return img.view(1, h, w)

    @staticmethod
    def most_pool(inp, kernel_size=8):
        assert isinstance(inp, torch.Tensor)
        c, h, w = inp.shape
        assert c == 1
        inp = inp[0]
        assert h % kernel_size == 0 and w % kernel_size == 0
        blocks = torch.zeros(size=(int(h / kernel_size), int(w / kernel_size)))
        for r in range(int(h / kernel_size)):
            for c in range(int(w / kernel_size)):
                block = inp[r * kernel_size:(r + 1) * kernel_size,
                        c * kernel_size:(c + 1) * kernel_size]
                block_elem, block_elem_count = torch.unique(
                    block, return_counts=True)
                blocks[r, c], max_num = 0, 0
                for i, elem in enumerate(block_elem):
                    if elem == 0:
                        continue
                    if block_elem_count[i] > max_num:
                        blocks[r, c] = elem
                        max_num = block_elem_count[i]
        return blocks.view((1, int(h / kernel_size), int(w / kernel_size)))

    @staticmethod
    def reassignment(inp):
        assert isinstance(inp, torch.Tensor)
        out = torch.zeros_like(inp)
        for i, val in enumerate(filter(lambda x: x != 0, torch.unique(inp))):
            out[inp == val] = i + 1
        return out

    def __getitem__(self, index):
        data = np.load(self.files[index])
        # image_path 是str不可以用于打包
        image_path = str(data['image_path'])
        image = torch.tensor(data['image'], dtype=torch.float)
        instance_masking = self.reassignment(torch.tensor(data['instance_masking'], dtype=torch.float))
        instance_edge = self.reassignment(torch.tensor(data['instance_edge'], dtype=torch.float))
        edge = torch.tensor(data['edge'], dtype=torch.float)
        grouping_matrix = torch.tensor(data['grouping_matrix'], dtype=torch.float)
        pool_edge = self.most_pool(edge)
        return image, instance_masking, instance_edge, edge, pool_edge, grouping_matrix
