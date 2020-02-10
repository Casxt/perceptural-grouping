from typing import List

import torch
from torch import nn
from torchvision.models import mobilenet_v2


# v2 将register_forward_hook修改为原结构直出


class EdgeGrouping(torch.nn.Module):
    # layers, inChannels, midChannels, outChannels
    mobile_output_layer: List[int] = [[3, 24, 48, 1],
                                      [6, 32, 64, 1],
                                      [10, 64, 128, 1],
                                      [13, 96, 192, 1],
                                      [17, 320, 512, 1]]

    # mobile_outputs_convs: List[torch.nn.Sequential] = [None for i in mobile_output_layer]

    def __init__(self):
        super().__init__()
        self.node_num = 1000
        # mobile_net_v2_b0输出尺寸即为原尺寸的 1 / 4
        self.feature_map_size_factor = 1 / 4
        # 从特征图中提取的区块大小
        self.feature_map_block_size = 3
        # 区块预测输出尺寸即为原尺寸的 1 / 8
        self.region_map_size_factor = 1 / 8

        self.mobile_blocks = (
            self.mobile_net_v2_b0, self.mobile_net_v2_b1, self.mobile_net_v2_b2, self.mobile_net_v2_b3,
            self.mobile_net_v2_b4,) = self.get_mobiel_block()

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(536, 64, 1, bias=False),
            InvertedResidual(64, 16, 1, 2),
            InvertedResidual(16, 16, 1, 2),
            InvertedResidual(16, 1, 1, 2),
        )

        # 区块是否包含边缘置信度预测
        self.edge_region_predict = nn.Sequential(
            # 150 * 200
            nn.Conv2d(536, 256, 1, bias=False),
            InvertedResidual(256, 512, 2, 2),
            # 512 * 75 * 100
            InvertedResidual(512, 512, 1, 2),
            # 512 * 75 * 100
            InvertedResidual(512, 64, 1, 2),
            InvertedResidual(64, 64, 1, 2),
            InvertedResidual(64, 3, 1, 2),
            # 3 * 75 * 100
        )

        # GCN处理, 内部包含init, 不需要显示init
        self.node_feature_grouping = nn.Sequential(
            # self.node_num, 512
            InvertedGraphConvolution(self.node_num, 4824, 8192, 2048),
            InvertedGraphConvolution(self.node_num, 2048, 4096, 2048),
            InvertedGraphConvolution(self.node_num, 2048, 4096, 1024),
            InvertedGraphConvolution(self.node_num, 1024, 2048, 1024),
            InvertedGraphConvolution(self.node_num, 1024, 2048, 128),
            # self.node_num, 128
        )

        self._initialize_weights(self.fuse_conv,
                                 self.edge_region_predict)

    def forward(self, x, edges, block_gt):
        batch_size, _c, h, w = x.shape

        # mobile block
        mobile_net_v2_output0 = self.mobile_net_v2_b0(x)
        mobile_net_v2_output1 = self.mobile_net_v2_b1(mobile_net_v2_output0)
        mobile_net_v2_output2 = self.mobile_net_v2_b2(mobile_net_v2_output1)
        mobile_net_v2_output3 = self.mobile_net_v2_b3(mobile_net_v2_output2)
        mobile_net_v2_output4 = self.mobile_net_v2_b4(mobile_net_v2_output3)

        mobile_outputs = (mobile_net_v2_output0, mobile_net_v2_output1, mobile_net_v2_output2,
                          mobile_net_v2_output3, mobile_net_v2_output4)
        # 特征 up to 150 * 200
        feature_map_list = tuple(
            map(lambda f: nn.functional.interpolate(f, size=(
                int(h * self.feature_map_size_factor), int(w * self.feature_map_size_factor))),
                mobile_outputs))

        feature_map = torch.cat(feature_map_list, dim=1)  # 536 * 150 * 200

        # 提取边缘 up to 600 * 800
        edge_fuse_output = edges

        rand = torch.abs(torch.rand_like(block_gt[:, 0], device=x.device))
        rand = rand / rand.max() / 10
        edge_region_predict = block_gt[:, 0:3]
        edge_region_predict[:, 0] += rand
        edge_region_predict[:, 0] /= edge_region_predict[:, 0].max()
        # 获得topk的节点id, topk_index (batch, node)
        _topk, topk_index = torch.topk(
            edge_region_predict[:, 0, :, :].reshape(edge_region_predict.shape[0], -1), k=self.node_num)

        # 对节点index进行排序, sorted_topk_index代表每个样本中边缘置信度最高的 self.node_num 个点的一维坐标
        # sorted_topk_index: b, self.node_num
        sorted_topk_index, _topk_index_index = topk_index.sort(dim=1)

        # 根据sorted_topk_index取出node map
        node_map = torch.empty((batch_size, self.node_num,
                                feature_map.shape[1] * self.feature_map_block_size * self.feature_map_block_size),
                               device=x.device)
        edge_region_h, edge_region_w = edge_region_predict.shape[2:4]
        for batch, indices in enumerate(sorted_topk_index):
            for node_index, pos_index in enumerate(indices):
                # 计算坐标缩放比例
                factor = self.feature_map_size_factor / self.region_map_size_factor
                # 计算在 region map 中的区块左上角坐标
                edge_region_y, edge_region_x = int(pos_index / edge_region_w), int(
                    pos_index % edge_region_w)
                # 计算在 feature map 中的区块左中心坐标
                y = round((edge_region_y + (
                        float(edge_region_predict[batch, 1, edge_region_y, edge_region_x]) + 1.) / 2.) * factor)
                x = round((float(edge_region_x) + (
                        float(edge_region_predict[batch, 2, edge_region_y, edge_region_x]) + 1.) / 2.) * factor)
                # feature map 中的区块左中心坐标 转换为左上角坐标, 修正坐标使得框选总在 feature map 内部, 此处会有舍入误差
                y = max(0, y - int(self.feature_map_block_size / 2))
                x = max(0, x - int(self.feature_map_block_size / 2))
                if y + self.feature_map_block_size > edge_region_h + 1:
                    y = edge_region_h + 1 - self.feature_map_block_size
                if x + self.feature_map_block_size > edge_region_w + 1:
                    x = edge_region_w + 1 - self.feature_map_block_size
                node_map[batch, node_index] = feature_map[batch, :, y:y + self.feature_map_block_size,
                                              x:x + self.feature_map_block_size].reshape(-1)

        # 创建邻接图, int包裹shape防止device error
        adjacent_map = self.generate_adjacent_map(sorted_topk_index, edge_region_predict.shape[3])

        # b, self.node_num, self.node_num + 256, adjacent_map和node_map维度不同，需要横向拼接
        #   a  b  c  f1 f2
        # a 0  *  *  -  -
        # b *  0  +  -  -
        # c *  +  0  -  -
        # 处理完成后通过[:, :, adjacent_map_width:]取出feature
        node_feature_grouping_inp = torch.cat((adjacent_map, node_map), dim=2)
        node_output_feature = self.node_feature_grouping(node_feature_grouping_inp)

        # 取出 feature
        node_output_feature = node_output_feature[:, :, adjacent_map.shape[1]:]

        return (edge_fuse_output, edge_region_predict, sorted_topk_index,
                node_output_feature)

    def image_edge_loss(self, input: torch.Tensor, target: torch.Tensor):
        """
        计算边缘损失
        :param input: b, 1, w, h
        :param target: b, 1, w, h
        :return:
        """
        pos_index = (target >= 0.5)
        neg_index = (target < 0.5)
        sum_num = input.shape[2] * input.shape[3]
        # 逐样本计算正负样本数
        pos_num = pos_index.view(input.shape[0], sum_num).sum(dim=1).type(torch.float)
        neg_num = neg_index.view(input.shape[0], sum_num).sum(dim=1).type(torch.float)

        # 扩张回矩阵大小， 并进行clone，保证各个元素之前不会互相影响
        neg_num = (neg_num.view(input.shape[0], 1, 1, 1) / sum_num).expand(*input.shape).clone()
        pos_num = (pos_num.view(input.shape[0], 1, 1, 1) / sum_num).expand(*input.shape).clone()

        # 计算每个样本点的损失权重 正样本点权重为 负样本/全样本 负样本点权重 正样本/全样本
        pos_num[pos_index] = 0
        neg_num[neg_index] = 0
        weight = (pos_num + neg_num) / sum_num
        return torch.nn.functional.binary_cross_entropy(input, target, weight, reduction='sum') / input.shape[0]

    def node_edge_loss(self, input: torch.Tensor, target: torch.Tensor):
        """
        计算节点边缘预测损失
        :param input: b, 1, w, h
        :param target: b, 1, w, h
        :return:
        """
        return torch.nn.functional.binary_cross_entropy(input, target, reduction='mean')

    def block_position_loss(self, input: torch.Tensor, target: torch.Tensor):
        """
        计算对于每一格block边缘位置损失
        @param input: c, 2, 75, 100,
        @param target: c, 2, 75, 100
        @return: dist: loss
        """
        dist = (input - target).pow(2)
        dist = dist[:, 0, :, :] + dist[:, 1, :, :]
        return dist.sum() / input.shape[0] / input.shape[2] / input.shape[3]

    def node_grouping_loss_v2(self, indices: torch.Tensor, nodes_feature: torch.Tensor, instances_map: torch.Tensor,
                              alpha=0.5,
                              beta=2):
        """
        计算对于每一格block边缘位置损失
        @param beta: 约束负样本间距
        @param alpha: 约束正样本间距
        @param indices: b, self.node_num
        @param nodes_feature: b, self.node_num, 128
        @param instances_map: b, 1, 75, 100 每格类型的标注
        @return: loss, a number
        """

        # 距离矩阵 兼 邻接矩阵 b, self.node_num, self.node_num
        dist_map = (torch.zeros(indices.shape[0],
                                indices.shape[1],
                                indices.shape[1],
                                requires_grad=False,
                                device=indices.device))

        triplet_loss = torch.empty(indices.shape[0], device=indices.device)

        map_length = dist_map.shape[2]
        # 生成表示点间距离的对称矩阵
        # 生成表示点间距离的对称矩阵
        #   a  b  c
        # a 0  *  *
        # b *  0  +
        # c *  +  0
        # 1. 计算 a到 b, c 的距离 *
        # 2. 计算 b到c 的距离 +
        for i in range(1, dist_map.shape[2]):
            dist_map[:, i - 1, i:map_length] = dist_map[:, i:map_length, i - 1] = (
                    nodes_feature[:, i:map_length, :] - nodes_feature[:, i - 1:i, :]
            ).pow(2).sum(dim=2)
        total_node_set = set(range(self.node_num))
        for batch, index in enumerate(indices):
            # inp = nodes_feature[batch]
            ins = instances_map[batch, 0]
            dist = dist_map[batch]
            # 每组包含n个node的index, index 为node在 self.node_num 个nodes中的顺序，用于在dist_map中索引该node
            node_sets = {int(t): [] for t in ins.unique()}
            # 将node分组
            for i, pos in enumerate(index):
                # 行坐标y, 列坐标x, pos 是 long 型 不需要 .floor()
                y, x = pos / instances_map.shape[3], (pos % instances_map.shape[3])
                node_sets[int(ins[y, x])].append(i)

            # 计算每个batch中每组的triple let loss
            batch_triplet_losses = []

            # 不一定每个instances id 都有内容, 筛选出 node_set 长度大于1的 node_set
            for this_ins, node_set in filter(lambda kv: 0 < len(kv[1]) < 1999, node_sets.items()):
                # 取出其他组的nodes
                other_group_nodes = tuple(total_node_set - set(node_set))
                this_dist = dist[node_set, :]
                # 找到不同组的小于最小间距的点 , 注意[node_set, :][:, node_set]不能写成[node_set, node_set]
                negative = this_dist[:, other_group_nodes] - beta
                negative_mask = (negative < 0)
                negative_mask_sum = negative_mask.sum()
                if negative_mask_sum > 0:
                    negative_loss = negative[negative_mask].sum() * -1 / negative_mask_sum
                else:
                    negative_loss = 0
                # 找到同组的大于最大间距的点
                positive = this_dist[:, node_set] - alpha
                positive_mask = (positive > 0)
                positive_mask_sum = positive_mask.sum()
                if positive_mask_sum > 0:
                    positive_loss = positive[positive_mask].sum() / positive_mask_sum
                else:
                    positive_loss = 0
                batch_triplet_losses.append(positive_loss + negative_loss)
            triplet_loss[batch] = sum(batch_triplet_losses)  # / len(node_sets)

        return triplet_loss.sum() / triplet_loss.shape[0], dist_map

    def get_mobiel_block(self):
        """
        获取mobile net的各个block
        @return:
        """
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

    def generate_adjacent_map(self, sorted_topk_index: torch.Tensor, feature_map_width: torch.Size):
        # 距离矩阵 兼 邻接矩阵 b, self.node_num, self.node_num
        adjacent_map = (torch.zeros(sorted_topk_index.shape[0],
                                    sorted_topk_index.shape[1],
                                    sorted_topk_index.shape[1],
                                    requires_grad=False,
                                    device=sorted_topk_index.device))
        # 行坐标y, 列坐标x, 均为tensor, 注意使用.floor()而不是math.floor(), sorted_topk_index 是Long型，不需要floor
        # b, self.node_num  b, self.node_num
        row, col = (sorted_topk_index / int(feature_map_width)), (sorted_topk_index % int(feature_map_width))
        map_length = int(adjacent_map.shape[2])
        # 生成表示点间距离的对称矩阵
        #   a  b  c
        # a 0  *  *
        # b *  0  +
        # c *  +  0
        # 1. 计算 a到 b, c 的距离
        # 2. 计算 b到c 的距离
        for i in range(1, map_length):
            adjacent_map[:, i - 1, i:map_length] = adjacent_map[:, i:map_length, i - 1] = (
                    torch.pow(row[:, i:map_length] - row[:, i - 1:i], 2) +
                    torch.pow(col[:, i:map_length] - col[:, i - 1:i], 2)
            )
        # 点间距离小于 4格对角线长5.656, 3格对角线长4.242,8格对角线长8.4, 同时将0-5的范围留空
        adjacent_map[adjacent_map < 4 ** 2] = 0
        # adjacent_map[adjacent_map < 8.5 ** 2] = 0.4
        # 10格对角线长14.14121, 同时将1-14.5的范围留空
        # adjacent_map[adjacent_map < 14.5 ** 2] = 0.5
        # 15格对角线长21.2132, 同时将1-21.5的范围留空
        # adjacent_map[adjacent_map < 21.5 ** 2] = 0.8
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


class InvertedGraphConvolution(nn.Module):
    def __init__(self, node_num, input_feature_num, mid_feature_num, output_feature_num):
        super().__init__()
        # shapes
        self.graph_num = node_num
        self.input_feature_num = input_feature_num
        self.output_feature_num = output_feature_num
        self.layer = nn.Sequential(
            GraphConvolution(node_num, input_feature_num, mid_feature_num, add_bias=False, batch_normal=True),
            GraphConvolution(node_num, mid_feature_num, mid_feature_num, batch_normal=True),
            nn.ReLU6(inplace=True),
            GraphConvolution(node_num, mid_feature_num, output_feature_num, batch_normal=True)
        )

    def forward(self, inp: torch.Tensor):
        x = self.layer(inp)
        if self.input_feature_num == self.output_feature_num:
            b, g, t = inp.shape
            x[:, :, g:t] += inp[:, :, g:t]
        return x


class GraphConvolution(nn.Module):

    def __init__(self, node_num, input_feature_num, output_feature_num, add_bias=True, dtype=torch.float,
                 batch_normal=True):
        super().__init__()
        # shapes
        self.graph_num = node_num
        self.input_feature_num = input_feature_num
        self.output_feature_num = output_feature_num
        self.add_bias = add_bias
        self.batch_normal = batch_normal

        # params
        self.weight = nn.Parameter(torch.empty(input_feature_num, self.output_feature_num, dtype=dtype))
        if add_bias:
            self.bias = nn.Parameter(torch.empty(self.graph_num, self.output_feature_num, dtype=dtype))
        else:
            self.bias = nn.Parameter(torch.empty(1, 1, dtype=dtype))

        if batch_normal:
            self.norm = nn.InstanceNorm1d(node_num)
        # init params
        self.params_reset()

    def params_reset(self):
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bias, 0)

    def set_trainable(self, train=True):
        for param in self.parameters():
            param.requires_grad = train

    def forward(self, inp: torch.Tensor):
        """
        @param inp : adjacent: (batch, graph_num, graph_num) cat node_feature: (batch, graph_num, in_feature_num) -> (batch, graph_num, graph_num + in_feature_num)
        @return:
        """
        b, g, t = inp.shape
        adjacent, node_feature = inp[:, :, 0:g], inp[:, :, g:t]
        x = torch.matmul(adjacent, node_feature)
        x = torch.matmul(x, self.weight)
        if self.add_bias:
            x = x + self.bias
        # short cut
        # if self.input_feature_num == self.output_feature_num:
        #     x = x + node_feature

        if self.batch_normal:
            x = self.norm(x)

        return torch.cat((adjacent, x), dim=2)
