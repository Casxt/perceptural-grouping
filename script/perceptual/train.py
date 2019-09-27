import sys
from os import path

sys.path.append(path.join(path.join(path.dirname(__file__), '..'), ".."))
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# import tool
from tool.BSDS500 import BSDS500

from models import PerceptualEdgeDetectionV2 as EdgeDetection
from torch.optim.lr_scheduler import ReduceLROnPlateau

#
epochs = 2000
batchSize = 8
# no_pretrain_lr0001 no_pretrain_lr1
# pretrain_lr0001
subPath = Path("pretrain_vgg_perceptua4b_lr01_fuse3")
save = Path("/root/perceptual_grouping/weight/perceptual/", subPath)
save.mkdir(parents=True) if not save.exists() else None

writer = SummaryWriter(Path("/root/perceptual_grouping/log/perceptual/", subPath))

dataSet = BSDS500(Path("/root/perceptual_grouping/dataset/BSDS500"))

net: torch.nn.Module = EdgeDetection().cuda()

# define the optimizer
optimizer = torch.optim.RMSprop([
    {'params': net.vgg16.parameters(), "lr": 0},
    {'params': net.fuse_conv.parameters(), "lr": 0},
    {'params': net.vgg_output_conv0.parameters(), "lr": 0},
    {'params': net.vgg_output_conv1.parameters(), "lr": 0},
    {'params': net.vgg_output_conv2.parameters(), "lr": 0},
    {'params': net.vgg_output_conv3.parameters(), "lr": 0},
    {'params': net.vgg_output_conv4.parameters(), "lr": 0},
    {'params': net.perceptual_fuse_1.parameters()},
    {'params': net.perceptual_fuse_2.parameters()},
    {'params': net.perceptual_fuse_3.parameters()},
], lr=1e-2)

scheduler = ReduceLROnPlateau(optimizer)

step = 0
for epoch in range(epochs):
    net.train()
    loss = [torch.tensor([0.]).cuda()] * 6

    if epoch == 20:
        for p in optimizer.param_groups:
            p["lr"] = 1e-3
    elif epoch == 40:
        for p in optimizer.param_groups:
            p["lr"] = 1e-4
    # 传入图像大小不同，只能一张一张训练
    train = DataLoader(dataSet.get_train(), shuffle=True, pin_memory=True, batch_size=1)
    for index, batch in enumerate(train):
        t = time.time()
        imgs, gts = batch
        imgs, gts = imgs.cuda(), gts.cuda()
        anss = net(imgs)
        for i, ans in enumerate(anss):
            loss[i] += EdgeDetection.binary_cross_entropy_loss(ans, gts)

        if index != 0 and index % batchSize == 0:
            for i, l in enumerate(loss):
                writer.add_scalar(f"train_loss_conv{i}", l.detach().cpu().numpy()[0] / batchSize, step)
            totalLoss = sum(loss)
            writer.add_scalar("train_loss", totalLoss.detach().cpu().numpy()[0] / batchSize, step)
            writer.add_image("tarin_image", anss[-1][0], step)
            print(
                f"epoch{epoch}    step{index}   samples {index}/{len(train.dataset)}" +
                f"    spend{(time.time() - t) / batchSize}s    loss{totalLoss.detach().cpu().numpy()[0] / batchSize}")
            optimizer.zero_grad()
            totalLoss.backward()
            optimizer.step()
            # torch.cuda.empty_cache()
            loss = [torch.tensor([0.]).cuda()] * 6
        step += 1

    net.eval()
    loss = [torch.tensor([0.], requires_grad=False).cuda()] * 6
    anss, gts = None, None
    val = DataLoader(dataSet.get_val(), shuffle=False, pin_memory=True, batch_size=1)
    for index, batch in enumerate(val):
        imgs, gts = batch
        imgs, gts = imgs.cuda(), gts.cuda()
        anss = net(imgs)
        for i, ans in enumerate(anss):
            loss[i] += EdgeDetection.binary_cross_entropy_loss(ans, gts).detach()

    # img = anss[-1].data.cpu().numpy()[0][0] * 255.0
    # img = img.astype(np.uint8)
    # Image.fromarray(img, 'L').save(Path(save, f"{epoch}-Loss{valLoss}.jpg"), "jpeg")
    writer.add_image("val_image", anss[-1][0], step)
    for i, l in enumerate(loss):
        writer.add_scalar(f"val_loss_conv{i}", l.detach().cpu().numpy()[0] / batchSize, step)

    totalLoss = sum(loss)
    valLoss = totalLoss.detach().cpu().numpy()[0] / len(val)
    print(f"epoch{epoch} val loss{valLoss}")
    writer.add_scalar("val_loss", valLoss, step)

    torch.save(net.state_dict(), Path(save, f"{epoch}-Loss{valLoss}.weight"))
    writer.flush()

    scheduler.step(valLoss, epoch)
