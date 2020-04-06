import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init, Conv2d, Sequential

from models.tools.InvertedResidual import InvertedResidual


class HGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, input_size, hidden_size, kernel_size, batchnorm=True, timesteps=8):
        super().__init__()
        self.padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.timesteps = timesteps
        self.batchnorm = batchnorm

        self.u1_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        self.u2_gate = nn.Conv2d(hidden_size, hidden_size, 1)

        self.w_gate_inh = nn.Parameter(torch.empty(hidden_size, hidden_size, kernel_size, kernel_size))
        self.w_gate_exc = nn.Parameter(torch.empty(hidden_size, hidden_size, kernel_size, kernel_size))

        self.alpha = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.gamma = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.kappa = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.w = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.mu = nn.Parameter(torch.empty((hidden_size, 1, 1)))

        if self.batchnorm:
            # self.bn = nn.ModuleList([nn.GroupNorm(25, 25, eps=1e-03) for i in range(32)])
            self.bn = nn.ModuleList([nn.BatchNorm2d(25, eps=1e-03) for i in range(32)])
        else:
            self.n = nn.Parameter(torch.randn(self.timesteps, 1, 1))

        init.orthogonal_(self.w_gate_inh)
        init.orthogonal_(self.w_gate_exc)

        #        self.w_gate_inh = nn.Parameter(self.w_gate_inh.reshape(hidden_size , hidden_size , kernel_size, kernel_size))
        #        self.w_gate_exc = nn.Parameter(self.w_gate_exc.reshape(hidden_size , hidden_size , kernel_size, kernel_size))
        self.w_gate_inh.register_hook(lambda grad: (grad + torch.transpose(grad, 1, 0)) * 0.5)
        self.w_gate_exc.register_hook(lambda grad: (grad + torch.transpose(grad, 1, 0)) * 0.5)
        #        self.w_gate_inh.register_hook(lambda grad: print("inh"))
        #        self.w_gate_exc.register_hook(lambda grad: print("exc"))

        init.orthogonal_(self.u1_gate.weight)
        init.orthogonal_(self.u2_gate.weight)

        for bn in self.bn:
            init.constant_(bn.weight, 0.1)

        init.constant_(self.alpha, 0.1)
        init.constant_(self.gamma, 1.0)
        init.constant_(self.kappa, 0.5)
        init.constant_(self.w, 0.5)
        init.constant_(self.mu, 1)

        init.uniform_(self.u1_gate.bias.data, 1, 8.0 - 1)
        self.u1_gate.bias.data.log()
        self.u2_gate.bias.data = -self.u1_gate.bias.data

    def forward(self, input_, prev_state2, timestep=0):

        if timestep == 0:
            prev_state2 = torch.empty_like(input_)
            init.xavier_normal_(prev_state2)

        # import pdb; pdb.set_trace()
        i = timestep
        if self.batchnorm:
            g1_t = torch.sigmoid(self.bn[i * 4 + 0](self.u1_gate(prev_state2)))
            c1_t = self.bn[i * 4 + 1](F.conv2d(prev_state2 * g1_t, self.w_gate_inh, padding=self.padding))

            next_state1 = F.relu(input_ - F.relu(c1_t * (self.alpha * prev_state2 + self.mu)))
            # next_state1 = F.relu(input_ - c1_t*(self.alpha*prev_state2 + self.mu))

            g2_t = torch.sigmoid(self.bn[i * 4 + 2](self.u2_gate(next_state1)))
            c2_t = self.bn[i * 4 + 3](F.conv2d(next_state1, self.w_gate_exc, padding=self.padding))

            h2_t = F.relu(self.kappa * next_state1 + self.gamma * c2_t + self.w * next_state1 * c2_t)
            # h2_t = F.relu(self.kappa*next_state1 + self.kappa*self.gamma*c2_t + self.w*next_state1*self.gamma*c2_t)

            prev_state2 = (1 - g2_t) * prev_state2 + g2_t * h2_t

        else:
            g1_t = F.sigmoid(self.u1_gate(prev_state2))
            c1_t = F.conv2d(prev_state2 * g1_t, self.w_gate_inh, padding=self.padding)
            next_state1 = F.tanh(input_ - c1_t * (self.alpha * prev_state2 + self.mu))
            g2_t = F.sigmoid(self.bn[i * 4 + 2](self.u2_gate(next_state1)))
            c2_t = F.conv2d(next_state1, self.w_gate_exc, padding=self.padding)
            h2_t = F.tanh(
                self.kappa * (next_state1 + self.gamma * c2_t) + (self.w * (next_state1 * (self.gamma * c2_t))))
            prev_state2 = self.n[timestep] * ((1 - g2_t) * prev_state2 + g2_t * h2_t)

        return prev_state2


class HGRU(nn.Module):

    def __init__(self, time_steps=8):
        super().__init__()
        self.max_instance_num = 8
        self.input_size = 224
        self.output_size = int(self.input_size / 8)

        self.time_steps = time_steps

        self.conv0 = InvertedResidual(1, 25, 1, 5)

        # 2 倍降采样
        self.down_resolution = InvertedResidual(25, 25, 2, 5)
        # 2 倍升采样
        self.up_resolution = lambda x: nn.functional.interpolate(x, scale_factor=2)
        # # 混合
        # self.mixed_layer = Conv2d(50, 25, kernel_size=(1, 1))

        self.hGru_full_resolution = HGRUCell(25, 25, 9)
        self.hGru_half_resolution = HGRUCell(25, 25, 9)
        self.hGru_mixed = HGRUCell(25, 25, 9)

        self.read_out = Sequential(
            # 224
            InvertedResidual(25, 128, 2, 2),
            InvertedResidual(128, 256, 2, 2),
            InvertedResidual(256, self.output_size ** 2, 2, 4),
            # 28
            InvertedResidual(self.output_size ** 2, self.output_size ** 2, 1, 2),
            # 28
        )

        self.num_perd = Sequential(
            # 224
            InvertedResidual(25, 64, 2, 2),
            InvertedResidual(64, 128, 2, 2),
            InvertedResidual(128, 256, 2, 2),
            InvertedResidual(256, 512, 2, 2),
            InvertedResidual(512, 1024, 2, 2),
            # b, c, 7, 7
            nn.MaxPool2d((1, 1)),
            # b, c, 1, 1
            Conv2d(1024, self.max_instance_num, kernel_size=(1, 1)),
            # b, 10, 1, 1
        )

    def forward(self, x, adjacent):
        (full_resolution_internal_state,
         half_resolution_internal_state,
         mixed_resolution_internal_state) = None, None, None

        x = self.conv0(x)
        x = torch.pow(x, 2)
        full_inp = x
        half_inp = self.down_resolution(x)
        for i in range(self.time_steps):
            full_resolution_internal_state = self.hGru_full_resolution(full_inp, full_resolution_internal_state,
                                                                       timestep=i)

            half_resolution_internal_state = self.hGru_half_resolution(half_inp, half_resolution_internal_state,
                                                                       timestep=i)
            # 拼接
            mixed_resolution_inp = full_resolution_internal_state + self.up_resolution(half_resolution_internal_state)
            mixed_resolution_internal_state = self.hGru_mixed(mixed_resolution_inp, mixed_resolution_internal_state,
                                                              timestep=i)
        gm = self.read_out(mixed_resolution_internal_state)
        num = self.num_perd(mixed_resolution_internal_state)
        return torch.sigmoid(gm), num
