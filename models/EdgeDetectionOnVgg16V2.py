import torch
from torch import nn
from torchvision.models.vgg import vgg16
from typing import List, Tuple


# v2 将register_forward_hook修改为原结构直出

# struct of vgg16.features
# Block0 [0, 3]
# Block1 [4, 8]
# Block2 [9, 15]
# Block3 [16, 22]
# Block4 [23, 29]
# pool not used here

class EdgeDetection(torch.nn.Module):
    vgg_output_layer: List[int] = [3, 8, 15, 22, 29]

    vgg_output: List[torch.Tensor] = [None, None, None, None, None]

    def __init__(self):
        super().__init__()
        vgg16_backen = vgg16(pretrained=True, init_weights=False).features[0:30]  # drop the last pooling layer

        self.vgg16_b0 = vgg16_backen[0:4]
        self.vgg16_b1 = vgg16_backen[4:9]
        self.vgg16_b2 = vgg16_backen[9:16]
        self.vgg16_b3 = vgg16_backen[16:23]
        self.vgg16_b4 = vgg16_backen[23:30]

        # use process vgg output and raw c,h,w input
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(5 * 1 + 3, 256, 1),
            nn.Conv2d(256, 256, 3, groups=256, padding=1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(256, 1, 1),
        )

        self.vgg_output_conv0 = nn.Sequential(
            nn.Conv2d(64, 128, 1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(128, 128, 3, groups=128, padding=1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(128, 1, 1))

        self.vgg_output_conv1 = nn.Sequential(
            nn.Conv2d(128, 256, 1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(256, 256, 3, groups=256, padding=1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(256, 1, 1))

        self.vgg_output_conv2 = nn.Sequential(
            nn.Conv2d(256, 256, 1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(256, 256, 3, groups=256, padding=1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(256, 1, 1))

        self.vgg_output_conv3 = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.Conv2d(256, 256, 3, groups=256, padding=1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(256, 1, 1))

        self.vgg_output_conv4 = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.Conv2d(256, 256, 3, groups=256, padding=1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(256, 1, 1))

        # save vgg_output_conv to list
        self.vgg_output_convs: List[torch.Conv2d] = [self.vgg_output_conv0, self.vgg_output_conv1,
                                                     self.vgg_output_conv2, self.vgg_output_conv3,
                                                     self.vgg_output_conv4]
        self.vgg_blocks: List[torch.Conv2d] = [self.vgg16_b0, self.vgg16_b1, self.vgg16_b2, self.vgg16_b3,
                                               self.vgg16_b4]
        self._initialize_weights(self.fuse_conv, *self.vgg_output_convs)

    def forward(self, x) -> Tuple:
        outputs: list[torch.Tensor] = []
        size = x.size()[2:4]
        vgg_output0 = self.vgg16_b0(x)
        vgg_output1 = self.vgg16_b1(vgg_output0)
        vgg_output2 = self.vgg16_b2(vgg_output1)
        vgg_output3 = self.vgg16_b3(vgg_output2)
        vgg_output4 = self.vgg16_b4(vgg_output3)
        # process vgg_output one by one
        for i, output in enumerate((vgg_output0, vgg_output1, vgg_output2, vgg_output3, vgg_output4)):
            output = self.vgg_output_convs[i](output)
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
