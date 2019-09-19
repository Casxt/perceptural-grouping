from pathlib import Path

import torch
from torch.utils.data import DataLoader

from BSDS500 import BSDS500
from models import EdgeDetection

#
epochs = 200
batchSize = 32

dataSet = BSDS500(Path("/root/perceptual_grouping/dataset/BSDS500"))
val = DataLoader(dataSet.get_val(), shuffle=False, pin_memory=True, batch_size=batchSize)
train = DataLoader(dataSet.get_train(), shuffle=True, pin_memory=True, batch_size=batchSize)

net: torch.nn.Module = EdgeDetection().cuda()

# define the optimizer
lr = 1e-4
lrDecay = 1e-1
optimizer = torch.optim.RMSprop(net.parameters(), lr=lr)
for epoch in range(epochs):
    for batch_idx, batch in enumerate(train):
        imgs, gts = batch
        net.train()
        anss = net(imgs)
        loss = torch.zeros([0]).cuda()
        for i, ans in enumerate(anss):
            loss += EdgeDetection.binary_cross_entropy_loss(ans, gts)
        loss.backward()
