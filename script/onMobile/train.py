import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tool.BSDS500 import BSDS500
from models import EdgeDetectionOnMobile
from torch.optim.lr_scheduler import ReduceLROnPlateau

#
epochs = 1000
batchSize = 16
# no_pretrain_lr0001 no_pretrain_lr1
# pretrain_lr0001
subPath = Path("mobile/pretrain_lr01_normal_conv_tarin_all")
save = Path("/root/perceptual_grouping/weight/", subPath)

save.mkdir(parents=True) if not save.exists() else None

writer = SummaryWriter(Path("/root/perceptual_grouping/log/", subPath))

dataSet = BSDS500(Path("/root/perceptual_grouping/dataset/BSDS500"))

net: torch.nn.Module = EdgeDetectionOnMobile()
net.cuda()
# define the optimizer
mobile_params = net.mobile_net_v2.parameters()
our_params = filter(lambda p: p not in mobile_params, net.parameters())
print(mobile_params, our_params)
optimizer = torch.optim.RMSprop([
    {"params": net.parameters(), "lr": 1e-2},
    # {"params": mobile_params, "lr": 0},
], lr=1e-2)
scheduler = ReduceLROnPlateau(optimizer)

step = 0
for epoch in range(epochs):

    if epoch == 20:
        for p in optimizer.param_groups:
            p['lr'] = 1e-2

    net.train()
    loss = torch.tensor([0.]).cuda()

    # 传入图像大小不同，只能一张一张训练
    train = DataLoader(dataSet.get_train(), shuffle=True, pin_memory=True, batch_size=1)
    for index, batch in enumerate(train):
        t = time.time()
        imgs, gts = batch
        imgs, gts = imgs.cuda(), gts.cuda()
        anss = net(imgs)
        # for i, ans in enumerate(anss):
        loss += EdgeDetectionOnMobile.binary_cross_entropy_loss(anss, gts)

        if index != 0 and index % batchSize == 0:
            writer.add_scalar("loss", loss.detach().cpu().numpy()[0] / batchSize, step)
            writer.add_image("tarin_image", anss[0], step)
            print(
                f"epoch{epoch}    step{index}   samples {index}/{len(train.dataset)}" +
                f"    spend{(time.time() - t) / batchSize}s    loss{loss.detach().cpu().numpy()[0] / batchSize}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            loss = torch.tensor([0.]).cuda()
        step += 1

    net.eval()
    loss = torch.tensor([0.], requires_grad=False).cuda()
    anss, gts = None, None
    val = DataLoader(dataSet.get_val(), shuffle=False, pin_memory=True, batch_size=1)
    for index, batch in enumerate(val):
        imgs, gts = batch
        imgs, gts = imgs.cuda(), gts.cuda()
        anss = net(imgs)
        # for i, ans in enumerate(anss):
        loss += EdgeDetectionOnMobile.binary_cross_entropy_loss(anss, gts).detach()

    valLoss = loss.detach().cpu().numpy()[0] / len(val)
    print(f"epoch{epoch} val loss{valLoss}")

    # img = anss[-1].data.cpu().numpy()[0][0] * 255.0
    # img = img.astype(np.uint8)
    # Image.fromarray(img, 'L').save(Path(save, f"{epoch}-Loss{valLoss}.jpg"), "jpeg")
    writer.add_image("val_image", anss[0], step)
    writer.add_scalar("val_loss", valLoss, step)
    torch.save(net.state_dict(), Path(save, f"{epoch}-Loss{valLoss}.weight"))
    writer.flush()
