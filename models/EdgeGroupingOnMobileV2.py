import torch
from torch import nn
from torch.nn import Conv2d, Sequential, Sigmoid
from torchvision.models import resnet50


# v2 将register_forward_hook修改为原结构直出


class EdgeGrouping(torch.nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet50(pretrained=False)
        self.backend = Sequential(
            # 64, h, w
            resnet.bn1,
            resnet.relu,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
            # 2048, h/8, w/8
        )

        self.surface = Sequential(
            Conv2d(1, 64, kernel_size=(3, 3), bias=False)
        )

        self.bottom = Sequential(
            Conv2d(2048, 1600, kernel_size=(3, 3), padding=1, bias=False),
            Sigmoid()
        )

        self._initialize_weights(self.surface,
                                 self.backend,
                                 self.bottom)

    def forward(self, x):
        x = self.surface(x)
        x = self.backend(x)
        x = self.bottom(x)
        return x

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
    def balance_loss(output: torch.Tensor, target: torch.Tensor):
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
        return torch.nn.functional.binary_cross_entropy(output, target, weight, reduction='sum') / target.shape[0] / 100

    @staticmethod
    def loss(output: torch.Tensor, target: torch.Tensor):
        return torch.nn.functional.binary_cross_entropy(output, target, reduction='sum') / target.shape[0] / 100

    @staticmethod
    def accuracy(output: torch.Tensor, target: torch.Tensor):
        output = torch.round(output)
        return 1 - torch.sum(output != target).float() / torch.sum(target)
