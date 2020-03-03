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
            GraphConvolution(node_num, input_feature_num, mid_feature_num, add_bias=first_add_bias, batch_normal=True),
            nn.ReLU6(inplace=True),
            GraphConvolution(node_num, mid_feature_num, mid_feature_num, batch_normal=False),
            GraphConvolution(node_num, mid_feature_num, output_feature_num, batch_normal=last_batch_normal)
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
        # self.params_reset()

    # def params_reset(self):
    #     nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
    #     nn.init.constant_(self.bias, 0)

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
        if self.batch_normal:
            x = self.norm(x)

        return torch.cat((adjacent, x), dim=2)
