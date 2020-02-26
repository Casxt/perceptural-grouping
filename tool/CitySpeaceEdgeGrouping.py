import os
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import Dataset


class EdgeGroupingDataset(Dataset):

    def __init__(self, dataset_path):
        self.files = []
        Path(dataset_path).absolute()
        for dir_name, dirs, files in os.walk(Path(dataset_path)):
            if len(files) > 0 and int(Path(dir_name).parts[-1]) in range(3, 11):
                self.files.extend([Path(dir_name, file) for file in files])

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
    def vision_transaction_matrix_trace(tm):
        assert isinstance(tm, torch.Tensor)
        c, h, w = tm.shape
        transaction_matrix = tm.clone().view(c, h * w)
        for i in range(h * w):
            transaction_matrix[i, i] = 0
        same_group_with = tuple(
            map(lambda x: int(x), torch.argmax(transaction_matrix, dim=1)))
        groups = []
        has_group = set()

        def trace_group(temp_group, next_i):
            if same_group_with[next_i] in temp_group:
                # 结局1 以新group身份进入groups
                groups.append(temp_group)
                return temp_group
            elif same_group_with[next_i] in has_group:
                # 结局2 合并入已有group
                group = tuple(
                    filter(lambda g: same_group_with[next_i] in g, groups))[0]
                group.update(temp_group)
                return group
            else:
                # 无法进入结局 继续迭代
                next_i = same_group_with[next_i]
                temp_group.add(next_i)
                return trace_group(temp_group, next_i)

        for i in range(h * w):
            if i in has_group:
                continue
            temp_group = {i}
            temp_group = trace_group(temp_group, i)
            has_group.update(temp_group)

        img = torch.zeros(size=(1, h * w))
        for i, g in enumerate(groups):
            img[0, list(g)] = i + 5
        return img.view(1, h, w)

    @staticmethod
    def vision_transaction_matrix_kmeans(tm, k):
        assert isinstance(tm, torch.Tensor)
        c, h, w = tm.shape
        transaction_matrix = tm.clone().view(c, h * w).permute(1, 0)
        print(transaction_matrix.shape)
        kmeans = KMeans(n_clusters=k, random_state=0).fit(
            transaction_matrix.numpy())
        labels = torch.tensor(kmeans.labels_)
        print(labels.shape, labels.max(), labels.min())
        return labels.view(1, h, w).numpy()

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
        instance_num = self.reassignment(torch.tensor(data['instance_num'], dtype=torch.int64))
        edge = torch.tensor(data['edge'], dtype=torch.float)
        nearby_matrix = torch.tensor(data['nearby_matrix'], dtype=torch.float)
        grouping_matrix = torch.tensor(data['grouping_matrix'], dtype=torch.float)
        pool_edge = self.most_pool(edge)
        return image, instance_masking, instance_edge, instance_num, edge, pool_edge, grouping_matrix, nearby_matrix
