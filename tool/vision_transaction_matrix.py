import torch
from torch import Tensor


def vision_transaction_matrix(tm: Tensor):
    c, h, w = tm.shape
    transaction_matrix = tm.clone().view(c, h * w)
    for i in range(h * w):
        transaction_matrix[i, i] = 0
    same_group_with = tuple(map(lambda x: int(x), torch.argmax(transaction_matrix[:], dim=1)))
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

    img = torch.zeros(size=(1, h * w))
    for i, g in enumerate(groups):
        img[0, list(g)] = i + 1
    return img.view(1, h, w)
