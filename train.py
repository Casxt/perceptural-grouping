import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from BSDS500 import BSDS500
from models import EdgeDetection

#
epochs = 2000
batchSize = 16
save = Path("/root/perceptual_grouping/vgg/weight")
writer = SummaryWriter("/root/perceptual_grouping/vgg/log/no_pretrain_lr0001")

dataSet = BSDS500(Path("/root/perceptual_grouping/dataset/BSDS500"))

net: torch.nn.Module = EdgeDetection().cuda()

# define the optimizer
lr = 1e-4
lrDecay = 1e-1
optimizer = torch.optim.RMSprop(net.parameters(), lr=lr)

step = 0
for epoch in range(epochs):
    net.train()
    loss = torch.tensor([0.]).cuda()

    # 传入图像大小不同，只能一张一张训练
    train = DataLoader(dataSet.get_train(), shuffle=True, pin_memory=True, batch_size=1)
    for index, batch in enumerate(train):
        t = time.time()
        optimizer.zero_grad()
        imgs, gts = batch
        imgs, gts = imgs.cuda(), gts.cuda()
        anss = net(imgs)
        for i, ans in enumerate(anss):
            loss += EdgeDetection.binary_cross_entropy_loss(ans, gts)

        if index != 0 and index % batchSize == 0:
            loss.backward()
            optimizer.step()
            writer.add_scalar("loss", loss.detach().cpu().numpy()[0] / batchSize, step)
            writer.add_image("tarin_image", anss[-1][0], step)
            print(
                f"epoch{epoch}    step{index}   samples {index}/{len(train.dataset)}" +
                f"    spend{(time.time() - t) / batchSize}s    loss{loss.detach().cpu().numpy()[0] / batchSize}")
            loss = torch.tensor([0.]).cuda()
        step += 1

    net.eval()
    loss = torch.tensor([0.]).cuda()
    anss = None
    val = DataLoader(dataSet.get_val(), shuffle=False, pin_memory=True, batch_size=1)
    for index, batch in enumerate(val):
        imgs, gts = batch
        imgs, gts = imgs.cuda(), gts.cuda()
        anss = net(imgs)
        for i, ans in enumerate(anss):
            loss += EdgeDetection.binary_cross_entropy_loss(ans, gts)

    valLoss = loss.detach().cpu().numpy()[0] / len(val)
    print(f"epoch{epoch} val loss{valLoss}")

    # img = anss[-1].data.cpu().numpy()[0][0] * 255.0
    # img = img.astype(np.uint8)
    # Image.fromarray(img, 'L').save(Path(save, f"{epoch}-Loss{valLoss}.jpg"), "jpeg")
    writer.add_image("val_image", anss[-1][0], step)
    writer.add_scalar("val_loss", valLoss, step)
    torch.save(net.state_dict(), Path(save, f"{epoch}-Loss{valLoss}.weight"))
    writer.flush()
