import torch
from torch import nn
from torch.nn import Conv2d, Sequential
from torchvision.models import resnet50


# v2 将register_forward_hook修改为原结构直出


class EdgeGroupingOnRestNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = 224
        self.output_size = int(self.input_size / 8)
        resnet = resnet50(pretrained=False)
        self.backend = Sequential(
            # 64, h, w
            # resnet.bn1,
            # resnet.relu,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
            # 2048, h/8, w/8
        )

        self.surface = Sequential(
            Conv2d(1, 64, kernel_size=(3, 3), padding=1, bias=False)
        )

        self.bottom = Sequential(
            Conv2d(2048, self.output_size ** 2, kernel_size=(3, 3), padding=1),
            Conv2d(self.output_size ** 2, self.output_size ** 2, kernel_size=(3, 3), padding=1, bias=False),
        )

        self.num_perd = Sequential(
            # b, c, 28, 28
            Conv2d(2048, 1024, kernel_size=(3, 3), padding=1, stride=2),
            # b, c, 14, 14
            Conv2d(1024, 512, kernel_size=(3, 3), padding=1, stride=2),
            # b, c, 7, 7
            nn.AdaptiveAvgPool2d((1, 1)),
            # b, c, 1, 1
            Conv2d(512, 10, kernel_size=(1, 1)),
            # b, 10, 1, 1
        )

        self._initialize_weights(self.surface,
                                 self.backend,
                                 self.bottom)

    def forward(self, x):
        x = self.surface(x)
        feature = self.backend(x)
        gm = self.bottom(feature)
        num = self.num_perd(feature)
        return gm, num

    def _initialize_weights(self, *parts):
        for part in parts:
            for m in part.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    @staticmethod
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
        return torch.nn.functional.binary_cross_entropy(output, target, weight, reduction='mean')

    @staticmethod
    def mask_loss(output: torch.Tensor, target: torch.Tensor, pool_edge: torch.Tensor):
        b, c, h, w = output.shape
        loss = torch.tensor(0., device=output.device)
        output = output.view(b, c, c)
        target = target.view(b, c, c)
        for b, edge in enumerate(pool_edge):
            idx = [i for i, v in enumerate(edge.view(c)) if v > 0]
            out = output[b][idx][:, idx]
            tar = target[b][idx][:, idx]
            loss += torch.nn.functional.binary_cross_entropy(torch.sigmoid(out), tar, reduction='mean')
        return loss / b

    @staticmethod
    def mask_sigmoid_loss(output: torch.Tensor, target: torch.Tensor, pool_edge: torch.Tensor):
        b, c, h, w = output.shape
        loss = torch.tensor(0., device=output.device)
        output = output.view(b, c, c)
        target = target.view(b, c, c)
        for b, edge in enumerate(pool_edge):
            idx = edge.view(c) > 0
            out = output[b][idx][:, idx]
            tar = target[b][idx][:, idx]
            for i in range(out.shape[1]):
                positive_mask = tar[:, i] > 0
                positive_mask[i] = False
                if positive_mask.sum() == 0:
                    print(f"skip a node with {(tar[:, i] > 0).sum()} elements")
                    continue
                res = out[:, i]
                positive_index = int(torch.argmax(res * positive_mask))
                loss += (torch.nn.functional.cross_entropy(res.view(1, -1),
                                                           torch.tensor([positive_index], device=output.device),
                                                           reduction='mean') / len(idx))
        return loss / b

    @staticmethod
    def k_loss(output: torch.Tensor, target: torch.Tensor):
        return torch.nn.functional.cross_entropy(output, target, reduction='mean')

    @staticmethod
    def topk_accuracy(output: torch.Tensor, target: torch.Tensor, k):
        # 检查每一个像素是否具有正确指向
        b, c, h, w = output.shape
        output = output.view(b, c, c)
        target = target.view(b, c, c)
        correct, total = 0, 0
        idx_mask = (target > 0)
        for (idx, out, tar) in zip(idx_mask, output, target):
            out = out[idx][:, idx]
            tar = tar[idx][:, idx]
            _v, topk = torch.topk(out, k=k, dim=0)
            for (rk, ro, rt) in zip(topk, tar, out):
                correct += torch.round(ro[rk]).eq(rt).sum()
                total += rt.nelement()
        return torch.tensor(correct / total, device=output.device)

    @staticmethod
    def k_accuracy(output: torch.Tensor, target: torch.Tensor):
        """
        分组数量k预测准确度
        @param output:
        @param target:
        @return:
        """
        return torch.argmax(output, dim=1).eq(target).sum() / target.nelement()
