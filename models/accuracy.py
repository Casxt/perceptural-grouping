import torch


def topk_accuracy(output: torch.Tensor, target: torch.Tensor, pool_edge: torch.Tensor, k):
    # 检查每一个像素是否具有正确指向
    b, c, h, w = output.shape
    output = output.view(b, c, c)
    target = target.view(b, c, c)
    correct = torch.tensor(0., dtype=output.dtype, device=output.device)
    total = torch.tensor(0., dtype=output.dtype, device=output.device)
    idx_mask = (pool_edge.view(b, c) > 0)
    for (idx, out, tar) in zip(idx_mask, output, target):
        out = out[idx][:, idx]
        tar = tar[idx][:, idx]
        _v, tpk = torch.topk(out, k=k, dim=0)
        for (rk, ro, rt) in zip(tpk, out, tar):
            correct += torch.round(ro[rk]).eq(rt[rk]).sum()
            total += rt.nelement()
    return correct / total


def k_accuracy(output: torch.Tensor, target: torch.Tensor):
    """
    分组数量k预测准确度
    @param output:
    @param target:
    @return:
    """
    # print(torch.argmax(output, dim=1), target)
    return torch.argmax(output, dim=1).eq(target).sum().float() / float(target.nelement())
