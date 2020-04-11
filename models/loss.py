import torch


def balance_bce_loss(output: torch.Tensor, target: torch.Tensor):
    pos_index = (target >= 0.5)
    neg_index = (target < 0.5)
    sum_num = int(output.nelement() / output.shape[0])
    # 逐样本计算正负样本数
    pos_num = pos_index.view(output.shape[0], sum_num).sum(dim=1).type(torch.float)
    neg_num = neg_index.view(output.shape[0], sum_num).sum(dim=1).type(torch.float)

    # 扩张回矩阵大小， 并进行clone，保证各个元素之前不会互相影响
    neg_num = (neg_num.view(output.shape[0], 1, 1, 1) / sum_num).expand(*output.shape).clone()
    pos_num = (pos_num.view(output.shape[0], 1, 1, 1) / sum_num).expand(*output.shape).clone()

    # 计算每个样本点的损失权重 正样本点权重为 负样本/全样本 负样本点权重 正样本/全样本
    pos_num[pos_index] = 0
    neg_num[neg_index] = 0
    weight = (pos_num + neg_num)
    return torch.nn.functional.binary_cross_entropy(output, target, weight, reduction='mean') * 100


def mask_bce_loss(output: torch.Tensor, target: torch.Tensor, pool_edge: torch.Tensor):
    b, c, h, w = output.shape
    loss = torch.tensor(0., device=output.device)
    output = output.view(b, c, c)
    target = target.view(b, c, c)
    pool_edge = pool_edge.view(b, c).gt(0)
    for idx, out, tar in zip(pool_edge, output, target):
        out = out[idx][:, idx]
        tar = tar[idx][:, idx]
        loss += torch.nn.functional.binary_cross_entropy(out, tar, reduction='mean')
    return loss / b


def k_loss(output: torch.Tensor, target: torch.Tensor):
    # print(torch.argmax(output, dim=1), target)
    print(output.shape, target.shape)
    return torch.nn.functional.cross_entropy(output, target, reduction='mean')
