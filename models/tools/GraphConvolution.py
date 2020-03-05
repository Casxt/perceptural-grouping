import torch
from torch import nn


class InvertedGraphConvolution(nn.Module):
    def __init__(self, node_num, input_feature_num, mid_feature_num, output_feature_num, first_add_bias=False,
                 last_batch_normal=True):
        super().__init__()
        # shapes
        # self.graph_num = node_num
        self.input_feature_num = input_feature_num
        self.output_feature_num = output_feature_num
        self.layer = nn.Sequential(
            GraphConvolution(node_num, input_feature_num, mid_feature_num, add_bias=first_add_bias, batch_normal=False),
            nn.ReLU6(inplace=True),
            GraphConvolution(node_num, mid_feature_num, mid_feature_num, batch_normal=False),
            GraphConvolution(node_num, mid_feature_num, output_feature_num, batch_normal=False)
        )

    def forward(self, inp: torch.Tensor):
        x = self.layer(inp)
        if self.input_feature_num == self.output_feature_num:
            b, c, n = inp.shape
            x[:, n:, :] += inp[:, n:, :]
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
        self.weight = nn.Parameter(torch.empty(self.output_feature_num, input_feature_num, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(self.output_feature_num, self.graph_num, dtype=dtype))
        if batch_normal:
            self.norm = nn.InstanceNorm1d(node_num)

    def set_trainable(self, train=True):
        for param in self.parameters():
            param.requires_grad = train

    def forward(self, inp: torch.Tensor):
        """
        @param inp : adjacent: (batch, graph_num, graph_num) cat node_feature: (batch, graph_num, in_feature_num) -> (batch, graph_num, graph_num + in_feature_num)
        @return:
        """
        b, c, n = inp.shape
        adjacent, node_feature = inp[:, 0:n, :], inp[:, n:, :]
        x = torch.matmul(self.weight, node_feature)
        x = torch.matmul(x, adjacent)
        if self.add_bias:
            x = x + self.bias
        if self.batch_normal:
            x = self.norm(x)

        return torch.cat((adjacent, x), dim=1)
