import torch
from torch import nn
from torchvision.models import mobilenet_v2
from typing import List


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
    # mobile_outputs: List[torch.Tensor] = [None for i in mobile_output_layer]
    mobile_outputs_convs: List[torch.nn.Sequential] = [None for i in mobile_output_layer]

    def __init__(self):
        super().__init__()
        self.mobile_net_v2 = mobilenet_v2(pretrained=True).features[0:18]

        self.mobile_net_v2_b0 = self.mobile_net_v2[0:4]
        self.mobile_net_v2_b1 = self.mobile_net_v2[4:7]
        self.mobile_net_v2_b2 = self.mobile_net_v2[7:11]
        self.mobile_net_v2_b3 = self.mobile_net_v2[11:14]
        self.mobile_net_v2_b4 = self.mobile_net_v2[14:18]

        self.mobile_blocks = (
            self.mobile_net_v2_b0, self.mobile_net_v2_b1, self.mobile_net_v2_b2, self.mobile_net_v2_b3,
            self.mobile_net_v2_b4,)

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
        self._initialize_weights(self.fuse_conv, *self.mobile_outputs_convs)

    def forward(self, x):
        outputs: list[torch.Tensor] = []
        size = x.size()[2:4]

        mobile_net_v2_output0 = self.mobile_net_v2_b0(x)
        mobile_net_v2_output1 = self.mobile_net_v2_b1(mobile_net_v2_output0)
        mobile_net_v2_output2 = self.mobile_net_v2_b2(mobile_net_v2_output1)
        mobile_net_v2_output3 = self.mobile_net_v2_b3(mobile_net_v2_output2)
        mobile_net_v2_output4 = self.mobile_net_v2_b4(mobile_net_v2_output3)

        # process vgg_output one by one
        for i, output in enumerate((mobile_net_v2_output0, mobile_net_v2_output1, mobile_net_v2_output2,
                                    mobile_net_v2_output3, mobile_net_v2_output4)):
            output = self.mobile_outputs_convs[i](output)
            output = nn.functional.interpolate(output, size=size, mode="bilinear")
            outputs.append(torch.sigmoid(output))

        fuse = self.fuse_conv(torch.cat([x, *outputs], 1))
        outputs.append(torch.sigmoid(fuse))
        return tuple(outputs)

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
    def binary_cross_entropy_loss(input: torch.Tensor, target: torch.Tensor):
        # log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        # target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        # 正负样本统计
        pos_index = (target > 0)
        neg_index = (target == 0)

        pos_num = pos_index.sum().type(torch.float)
        neg_num = neg_index.sum().type(torch.float)
        sum_num = pos_num + neg_num

        # 计算每个样本点的损失权重
        weight = torch.empty(target.size()).cuda()
        weight[pos_index] = neg_num / sum_num
        weight[neg_index] = pos_num / sum_num
        return torch.nn.functional.binary_cross_entropy(input, target, weight, reduction='mean')
