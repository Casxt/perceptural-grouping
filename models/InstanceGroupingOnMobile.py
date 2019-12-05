import torch
from torch import nn
from torchvision.models import mobilenet_v2
from typing import List, Tuple
import numpy as np


# v2 将register_forward_hook修改为原结构直出

# struct of vgg16.features
# 1 BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# 3 BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# 6 BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# 10 BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# 13 BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# 17 BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# pool not used here

class EdgeDetection(torch.nn.Module):
    # layers, inChannels, midChannels, outChannels
    mobile_output_layer: List[int] = [[3, 24, 48, 1],
                                      [6, 32, 64, 1],
                                      [10, 64, 128, 1],
                                      [13, 96, 192, 1],
                                      [17, 320, 512, 1]]
    mobile_outputs_convs: List[torch.nn.Sequential] = [None for i in mobile_output_layer]

    def __init__(self):
        super().__init__()
        self.mobile_blocks = (
            self.mobile_net_v2_b0, self.mobile_net_v2_b1, self.mobile_net_v2_b2, self.mobile_net_v2_b3,
            self.mobile_net_v2_b4,) = self.get_mobiel_block()

        # hook vgg output
        total_mobile_channel = 3
        for i, l in enumerate(self.mobile_output_layer):
            layers, inChannels, midChannels, outChannels = l
            self.mobile_outputs_convs[i] = nn.Sequential(
                nn.Conv2d(inChannels, midChannels, 1),
                nn.BatchNorm2d(midChannels),
                nn.ReLU6(inplace=True),
                nn.Conv2d(midChannels, midChannels, 3, groups=midChannels, padding=1),
                nn.ReLU6(inplace=True),
                nn.Conv2d(midChannels, outChannels, 1),
                nn.BatchNorm2d(outChannels)
            )
            self.add_module(f"mobile_outputs_convs[{i}]", self.mobile_outputs_convs[i])
            total_mobile_channel += outChannels

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(total_mobile_channel, total_mobile_channel * 2, 1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(total_mobile_channel * 2, total_mobile_channel * 2, 3, groups=total_mobile_channel * 2,
                      padding=1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(total_mobile_channel * 2, 1, 1),
        )

        # 区块特征提取
        self.edge_region_feature = nn.Sequential(
            # 600 * 800
            nn.Conv2d(6, 24, 1, bias=False),
            InvertedResidual(24, 24, 1, 2),
            # 600 * 800
            nn.Conv2d(24, 32, 3, stride=2, padding=1),
            # 300 * 400
            nn.Conv2d(32, 48, 3, stride=2, padding=1),
            # 150 * 200
            nn.Conv2d(48, 64, 3, stride=2, padding=1),
            # 64 * 75 * 100
            nn.BatchNorm2d(64),
            nn.ReLU6(inplace=True),
        )

        # 区块是否包含边缘置信度预测
        self.edge_region_predict = nn.Sequential(
            # 64 * 75 * 100
            InvertedResidual(64, 64, 1, 2),
            nn.Conv2d(64, 32, 1),
            nn.Conv2d(32, 3, 1),
            # 3 * 75 * 100
        )

        # GCN处理, 内部包含init, 不需要显示init
        self.node_feature_grouping = nn.Sequential(
            # 2000, 64 + 64 * 8 * 8 = 4160
            GraphConvolution(2000, 4160, 1024),
            GraphConvolution(2000, 1024, 1024),
            GraphConvolution(2000, 1024, 256),
            # 2000, 256
        )

        self._initialize_weights(self.fuse_conv, *self.mobile_outputs_convs, self.edge_region_feature,
                                 self.edge_region_predict)

    def forward(self, x):
        # b,c,h,w
        size = x.size()[2:4]

        # mobile block
        mobile_net_v2_output0 = self.mobile_net_v2_b0(x)
        mobile_net_v2_output1 = self.mobile_net_v2_b1(mobile_net_v2_output0)
        mobile_net_v2_output2 = self.mobile_net_v2_b2(mobile_net_v2_output1)
        mobile_net_v2_output3 = self.mobile_net_v2_b3(mobile_net_v2_output2)
        mobile_net_v2_output4 = self.mobile_net_v2_b4(mobile_net_v2_output3)

        mobile_outputs = (mobile_net_v2_output0, mobile_net_v2_output1, mobile_net_v2_output2,
                          mobile_net_v2_output3, mobile_net_v2_output4)

        # 边缘提取网络
        edge_outputs: list[torch.Tensor] = []
        for i, output in enumerate(mobile_outputs):
            output = self.mobile_outputs_convs[i](output)
            output = nn.functional.interpolate(output, size=size, mode="bilinear")
            edge_outputs.append(torch.sigmoid(output))

        edge_fuse_output = torch.sigmoid(self.fuse_conv(torch.cat([x, *edge_outputs], 1)))
        edge_outputs.append(edge_fuse_output)

        # 边缘特征
        feature_map_list = list(
            map(lambda f: nn.functional.interpolate(output, size=size, mode="bilinear"), mobile_outputs))
        feature_map_list.append(edge_fuse_output)
        feature_map = torch.cat(feature_map_list, 1)  # 64 * 600 * 800

        # 区块特征 b, 64, 75, 100
        edge_region_feature = self.edge_region_feature(feature_map)

        # 边缘特征图转节点特征, 使用先行后列的遍历方式, 将64个通道上每8*8大小区块的数据展平作为特征
        # feature_map 64 * 600 * 800 -> backend_node_feature 7500 * (64 * 8 * 8)
        block_size = 8
        backend_node_feature = torch.cat(
            tuple(feature_map[:, :, y:y + block_size, x:x + block_size]
                  .reshape(feature_map.shape[0], 1, feature_map.shape[1] * block_size * block_size)
                  for x in range(0, 800, block_size) for y in range(0, 600, block_size)),
            1)  # b, n, f

        # 区块预测网络, b, 3 * 75 * 100
        edge_region_predict = torch.sigmoid(self.edge_region_predict(edge_region_feature))

        # 区块特征图转节点特征, 使用先行后列的遍历方式, 将64个通道的数据作为特征
        # edge_region_feature 64 * 75 * 100 -> backend_node_feature 7500 * 64
        # edge_region_predict_and_feature = torch.cat((edge_region_predict, edge_region_feature), dim=1)
        block_feature = torch.cat(
            tuple(edge_region_feature[:, :, y, x]
                  .reshape(edge_region_feature.shape[0], 1, edge_region_feature.shape[1])
                  for x in range(0, 100) for y in range(0, 75)),
            1)  # b, n, f

        # 将边缘网络和区块网络的特征拼接
        node_feature = torch.cat((block_feature, backend_node_feature), 2)

        # 获得topk的节点id, topk_index (batch, node)
        _topk, topk_index = torch.topk(
            edge_region_predict[:, 0, :, :].reshape(edge_region_predict.shape[0], -1), k=2000)
        # 对节点index进行排序
        sorted_topk_index, _topk_index_index = topk_index.sort(dim=1)

        # 取出node map
        node_map = torch.cat(
            tuple(node_feature[batch, indices, :].unsqueeze(0) for batch, indices in enumerate(sorted_topk_index)),
            0)
        # 创建邻接图
        adjacent_map = self.generate_adjacent_map(sorted_topk_index, edge_region_predict.shape[3])
        node_output_feature = self.node_feature_grouping(
            torch.cat((node_map.unsqueeze(0), adjacent_map.unsqueeze(0)), 0))

        return (*edge_outputs, edge_region_predict, node_output_feature)

    @staticmethod
    def binary_cross_entropy_loss(input: torch.Tensor, target: torch.Tensor):
        """
        计算边缘损失
        :param input: b, 1, w, h
        :param target: b, 1, w, h
        :return:
        """
        pos_index = (target > 0)
        neg_index = (target == 0)

        pos_num = pos_index.sum().type(torch.float)
        neg_num = neg_index.sum().type(torch.float)
        sum_num = pos_num + neg_num

        # 计算每个样本点的损失权重
        weight = torch.empty(target.size()).to(input.device)
        weight[pos_index] = neg_num / sum_num
        weight[neg_index] = pos_num / sum_num
        return torch.nn.functional.binary_cross_entropy(input, target, weight, reduction='mean')

    def batch_binary_cross_entropy_loss(self, input: torch.Tensor, target: torch.Tensor):
        """
        计算边缘损失
        :param input: b, 1, w, h
        :param target: b, 1, w, h
        :return:
        """
        pos_index = (target > 0)
        neg_index = (target == 0)
        sum_num = input.shape[2] * input.shape[3]
        # 逐样本计算正负样本数
        pos_num = pos_index.view(input.shape[0], sum_num).sum(dim=1).type(torch.float)
        neg_num = neg_index.view(input.shape[0], sum_num).sum(dim=1).type(torch.float)

        # 扩张回矩阵大小
        neg_num = (neg_num.view(input.shape[0], 1, 1, 1) / sum_num).expand(*input.shape).clone()
        pos_num = (pos_num.view(input.shape[0], 1, 1, 1) / sum_num).expand(*input.shape).clone()

        # 计算每个样本点的损失权重 正样本点权重为 负样本/全样本 负样本点权重 正样本/全样本
        pos_num[pos_index] = neg_num[neg_index] = 0
        weight = (pos_num + neg_num) / sum_num

        # weight = torch.empty(target.size()).to(input.device)
        # weight[pos_index] = neg_num / sum_num
        # weight[neg_index] = pos_num / sum_num
        # return torch.nn.functional.binary_cross_entropy(input, target, weight, reduction='mean')
        return torch.nn.CrossEntropyLoss(weight)(input, target)


    def block_predict_loss(self, input: torch.Tensor, edge: torch.Tensor):
        """
        计算对于每一格block的边缘置信度损失，边缘位置损失
        @param input: c, 3, 75, 100,
        @param target: c, 3, 75, 100
        @return:
        """
        has_edge = input[:, 0, :, :]
        torch.nn.CrossEntropyLoss()(has_edge, edge)

    def get_mobiel_block(self):
        mobile_net_v2 = mobilenet_v2(pretrained=True).features[0:18]
        return (mobile_net_v2[0:4],
                mobile_net_v2[4:7],
                mobile_net_v2[7:11],
                mobile_net_v2[11:14],
                mobile_net_v2[14:18],
                )

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

    def generate_adjacent_map(self, sorted_topk_index: torch.Tensor, feature_map_width):
        # 距离矩阵 兼 邻接矩阵
        adjacent_map = (torch.zeros(sorted_topk_index.shape[0],
                                    sorted_topk_index.shape[1],
                                    sorted_topk_index.shape[1],
                                    requires_grad=False)
                        .to(sorted_topk_index.device))
        # 行坐标y
        row = (sorted_topk_index / feature_map_width).floor()
        # 列坐标x
        col = (sorted_topk_index % feature_map_width)

        # 生成表示点间距离的对称矩阵
        for i in range(1, adjacent_map.shape[2]):
            adjacent_map[:, i - 1, i:5] = adjacent_map[:, i:5, i - 1] = (
                    (row[:, i:5] - row[:, i - 1:i]) ** 2 +
                    (col[:, i:5] - col[:, i - 1:i]) ** 2
            )
        # 4格对角线长5.656, 3格对角线长4.242, 同时将0-5的范围留空
        adjacent_map[adjacent_map < 5 ** 2] = 0
        # 10格对角线长14.14121, 同时将1-14.5的范围留空
        adjacent_map[adjacent_map < 14.5 ** 2] = 0.5
        # 15格对角线长21.2132, 同时将1-21.5的范围留空
        adjacent_map[adjacent_map < 21.5 ** 2] = 0.8
        # 其他
        adjacent_map[adjacent_map >= 21 ** 2] = 1.
        # 翻转, 使距离为0的点权重为1, 以此类推
        adjacent_map = (adjacent_map - 1) * -1
        return adjacent_map


class ConvBNReLU(nn.Sequential):
    """
    卷积 + 归一化 + 激活
    """

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    """
    残差模块 用于加深深度
    出入口时已经包含归一化 和 激活
    """

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class GraphConvolution(nn.Module):

    def __init__(self, node_num, input_feature_num, output_feature_num, add_bias=True, dtype=torch.float64):
        super().__init__()
        # shapes
        self.graph_num = node_num
        self.input_feature_num = input_feature_num
        self.output_feature_num = output_feature_num
        self.add_bias = add_bias

        # params
        self.weight = nn.Parameter(torch.tensor(input_feature_num, self.output_feature_num, dtype=dtype))
        self.bias = nn.Parameter(torch.tensor(self.graph_num, self.output_feature_num, dtype=dtype))

        # init params
        self.params_reset()

    def params_reset(self):
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.merge, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bias, 0)

    def set_trainable(self, train=True):
        for param in self.parameters():
            param.requires_grad = train

    def forward(self, input):
        """
        :param node_feature: (batch, graph_num, in_feature_num)
        :param adjacent: (batch, graph_num, graph_num)
        :return:
        """
        node_feature, adjacent = input
        x = torch.matmul(adjacent, node_feature)
        x = torch.matmul(x, self.weight)
        if self.add_bias:
            x = x + self.bias
        if self.input_feature_num == self.output_feature_num:
            x = x + node_feature
        return x
