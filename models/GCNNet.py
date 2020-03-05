import torch
from torch import nn
from torch.nn import Conv2d, Sequential
from torchvision.models import resnet50

from models.tools import initialize_weights
from models.tools.GraphConvolution import InvertedGraphConvolution
from models.tools.InvertedResidual import InvertedResidual


class GroupNumPredict(torch.nn.Module):
    def __init__(self, node_num, out_dim):
        super().__init__()
        self.focus_layer = nn.Sequential(
            InvertedGraphConvolution(node_num, 2048, 4096, 1024, last_batch_normal=False)
        )

        self.pooling_layer = nn.Sequential(
            # b, c, 28, 28
            nn.BatchNorm2d(1024),
            Conv2d(1024, 1024, kernel_size=(3, 3), padding=1, stride=2),
            # b, c, 14, 14
            Conv2d(1024, 1024, kernel_size=(3, 3), padding=1, stride=2),
            nn.BatchNorm2d(1024),
            nn.ReLU6(),
            # b, c, 7, 7
            Conv2d(1024, 1024, kernel_size=(1, 7)),
            Conv2d(1024, 1024, kernel_size=(7, 1)),
            # b, c, 1, 1
            nn.BatchNorm2d(1024),
            nn.ReLU6(),
            Conv2d(1024, 1024, kernel_size=(1, 1)),
            Conv2d(1024, out_dim, kernel_size=(1, 1)),
            # b, out_dim, 1, 1
        )

    def forward(self, x: torch.Tensor, adjacent: torch.Tensor):
        b, c, h, w = x.shape
        x = x.view(b, -1, h * w)
        o = self.focus_layer(torch.cat([adjacent, x], dim=1))
        x = o[:, h * w:, :]
        x = x.view(b, -1, h, w)
        x = self.pooling_layer(x)
        x = x.view(b, -1)
        return x


class FocusGrouping(torch.nn.Module):
    def __init__(self, node_num, in_dim, out_dim):
        super().__init__()

        self.surface = nn.Sequential(
            InvertedResidual(in_dim, in_dim, 1, 2),
            InvertedResidual(in_dim, 1024, 1, 2),
            InvertedResidual(1024, 1024, 1, 4),
            InvertedResidual(1024, 1024, 1, 2),
            InvertedResidual(1024, out_dim, 1, 2),
        )

        self.focus_layer = nn.Sequential(
            InvertedGraphConvolution(node_num, out_dim, out_dim * 2, out_dim, last_batch_normal=False),
        )

    def forward(self, x: torch.Tensor, adjacent: torch.Tensor):
        b, c, h, w = x.shape
        x = self.surface(x)
        x = x.view(b, -1, h * w)
        o = self.focus_layer(torch.cat([adjacent, x], dim=1))
        x = o[:, h * w:, :]
        x = x.view(b, -1, h, w)
        # x = self.bottom(x)
        return x


class EdgeGroupingOnGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.max_instance_num = 8
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

        self.bottom = FocusGrouping(self.output_size ** 2, 2048, self.output_size ** 2)

        self.num_predict = GroupNumPredict(self.output_size ** 2, self.max_instance_num)

        initialize_weights(self.surface,
                           self.backend,
                           self.bottom,
                           self.num_predict)

    def forward(self, x, adjacent):
        x = self.surface(x)
        feature = self.backend(x)
        gm = self.bottom(feature, adjacent)
        num = self.num_predict(feature, adjacent)
        return torch.sigmoid(gm), num

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
        return torch.nn.functional.binary_cross_entropy(output, target, weight, reduction='mean') * 100

    @staticmethod
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

    @staticmethod
    def k_loss(output: torch.Tensor, target: torch.Tensor):
        # print(torch.argmax(output, dim=1), target)
        return torch.nn.functional.cross_entropy(output, target, reduction='mean')

    @staticmethod
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

    @staticmethod
    def k_accuracy(output: torch.Tensor, target: torch.Tensor):
        """
        分组数量k预测准确度
        @param output:
        @param target:
        @return:
        """
        # print(torch.argmax(output, dim=1), target)
        return torch.argmax(output, dim=1).eq(target).sum().float() / float(target.nelement())
