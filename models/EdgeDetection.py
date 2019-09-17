import torch
from torch import nn
from torchvision.models.vgg import vgg16
import torch.nn.functional as F
from typing import List


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
        self.vgg16 = vgg16(pretrained=False, init_weights=False).features[0:30]  # drop the last pooling layer

        # hook vgg output
        for i, l in enumerate(self.vgg_output_layer):
            self.vgg16[l].register_forward_hook(self.get_hook(i))

        # use process vgg output
        self.fuse_conv = nn.Conv2d(len(self.vgg_output), 1, 1)
        self.vgg_output_conv0 = nn.Conv2d(64, 1, 1)
        self.vgg_output_conv1 = nn.Conv2d(128, 1, 1)
        self.vgg_output_conv2 = nn.Conv2d(256, 1, 1)
        self.vgg_output_conv3 = nn.Conv2d(512, 1, 1)
        self.vgg_output_conv4 = nn.Conv2d(512, 1, 1)

        # save vgg_output_conv to list
        self.vgg_output_convs: List[torch.Conv2d] = [self.vgg_output_conv0, self.vgg_output_conv1,
                                                     self.vgg_output_conv2, self.vgg_output_conv3,
                                                     self.vgg_output_conv4]

        self._initialize_weights()

    def forward(self, x):
        outputs: list[torch.Tensor] = []
        size = x.size()[2:4]
        self.vgg16(x)  # run vgg and trigger hook

        # process vgg_output one by one
        for i, output in enumerate(self.vgg_output):
            output = self.vgg_output_convs[i](output)
            output = F.interpolate(output, size=size)
            outputs.append(torch.sigmoid(output))

        fuse = self.fuse_conv(torch.cat(outputs, 1))
        outputs.append(fuse)
        return outputs

    def get_hook(self, layer):
        def hook(module, input_tensor, output_tensor):
            self.vgg_output[layer] = output_tensor

        return hook

    def _initialize_weights(self):
        for m in self.modules():
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
