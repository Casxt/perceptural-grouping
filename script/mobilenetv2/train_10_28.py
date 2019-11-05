import sys
from os import path

import cv2

sys.path.append(path.join(path.join(path.dirname(__file__), '..'), ".."))
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tool import cat_in_out_gt, to_device
from tool.CitySpeace import CitySpace
from models import EdgeDetectionOnMobileV2 as EdgeDetection

# cityspace 数据集， 一切默认
device = 0
epochs = 2000
batchSize = 16
workernum = 12
subPath = Path("mobilenetv2/plain_4")
save = Path("/root/perceptual_grouping/weight", subPath)
save.mkdir(parents=True) if not save.exists() else None

writer = SummaryWriter(Path("/root/perceptual_grouping/log", subPath))

# dataSet = BSDS500(Path("/root/perceptual_grouping/dataset/BSDS500"))
dataSet = CitySpace(Path("/root/perceptual_grouping/dataset/cityspace"))
net: torch.nn.Module = EdgeDetection().cuda(device)
writer.add_graph(net, torch.rand(1, 3, 600, 800).cuda(device), True)
# define the optimizer
optimizer = torch.optim.RMSprop([
    {'params': net.mobile_net_v2_b0.parameters(), "lr": 0},
    {'params': net.mobile_net_v2_b1.parameters(), "lr": 0},
    {'params': net.mobile_net_v2_b2.parameters(), "lr": 0},
    {'params': net.mobile_net_v2_b3.parameters(), "lr": 0},
    {'params': net.mobile_net_v2_b4.parameters(), "lr": 0},
    {'params': net.fuse_conv.parameters()},
    {'params': net.mobile_outputs_convs[0].parameters()},
    {'params': net.mobile_outputs_convs[1].parameters()},
    {'params': net.mobile_outputs_convs[2].parameters()},
    {'params': net.mobile_outputs_convs[3].parameters()},
    {'params': net.mobile_outputs_convs[4].parameters()}], lr=1e-3)

# writer.add_graph(net, torch.rand(1, 3, 836, 2035).cuda(), False)

step, val_step = 0, 0
for epoch in range(epochs):
    net.train()
    losses = [torch.tensor([0.]).cuda(device)] * 6

    if epoch == 2:
        for p in optimizer.param_groups:
            p["lr"] = 1e-4

    train = DataLoader(dataSet.get_train(), shuffle=True, pin_memory=False, num_workers=workernum, batch_size=batchSize)
    for index, batch in enumerate(train):
        losses = [torch.tensor([0.]).cuda(device)] * 6
        imgs, gts, edges = to_device(device, *batch)
        start_time = time.time()
        anss = net(imgs)
        used_time = time.time() - start_time

        for i, ans in enumerate(anss):
            losses[i] += EdgeDetection.binary_cross_entropy_loss(ans, edges)
            writer.add_scalar(f"tarin_loss_conv{i}", losses[i].detach().cpu().numpy()[0] / len(imgs), step)

        total_loss = sum(losses)
        total_loss_val = total_loss.detach().cpu().numpy()[0]
        writer.add_scalar("tarin_loss", total_loss_val / len(imgs), step)
        writer.add_image("tarin_image", cat_in_out_gt(imgs[0], anss[-1][0], edges[0]), step)
        print(f"epoch{epoch}    step{index}   samples {step}/{len(train.dataset)}",
              f"spend {format(used_time / len(imgs), '.6f')}s",
              f"loss {format(total_loss_val / len(imgs), '.9f')}", sep="    ")


        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        step += len(imgs)
        writer.flush()

    with torch.no_grad():
        net.eval()
        losses = [torch.tensor([0.]).cuda(device)] * 6
        anss, gts, edges = None, None, None
        val = DataLoader(dataSet.get_val(), shuffle=False, pin_memory=False, num_workers=workernum,
                         batch_size=batchSize)
        total_time = 0
        for index, batch in enumerate(val):
            step_losses = [torch.tensor([0.]).cuda(device)] * 6
            imgs, gts, edges = to_device(device, *batch)

            start_time = time.time()
            anss = net(imgs)
            used_time = time.time() - start_time

            total_time += used_time
            val_step += len(imgs)
            for i, ans in enumerate(anss):
                step_losses[i] = EdgeDetection.binary_cross_entropy_loss(ans, edges).detach()
                losses[i] += step_losses[i]
            print(f"epoch{epoch}    val_step{index}   val_samples {val_step}/{len(val.dataset)}",
                  f"val_spend {format(used_time / len(imgs), '.6f')}s",
                  f"val_loss {format(sum(step_losses) / len(imgs), '.9f')}", sep="    ")
            writer.add_image("val_image_step", cat_in_out_gt(imgs[0], anss[-1][0], edges[0]), val_step)

        for i, l in enumerate(losses):
            writer.add_scalar(f"val_loss_conv{i}", l.detach().cpu().numpy()[0] / len(val.dataset), val_step)

        total_loss = sum(losses).detach().cpu().numpy()[0] / len(val.dataset)
        print(f"epoch {epoch}",
              f"val_spend {total_time / len(val.dataset)}s",
              f"val_loss {format(total_loss, '.9f')}", sep="    ")
        writer.add_scalar("val_loss", total_loss, val_step)
        writer.add_image("val_image", cat_in_out_gt(imgs[0], anss[-1][0], edges[0]), val_step)
        torch.save(net.state_dict(), Path(save, f"epoch{epoch}-step{step}-val_step{val_step}-Loss{total_loss}.weight"))
        writer.flush()
