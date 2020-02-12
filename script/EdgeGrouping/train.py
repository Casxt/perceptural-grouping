import sys
from os import path
from random import shuffle

sys.path.append(path.join(path.join(path.dirname(__file__), '..'), ".."))

import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tool import to_device, render_color, vision_transaction_matrix
from tool.CitySpeaceEdge import CitySpace
from models import EdgeGrouping

# cityspace 数据集， 一切默认
device = 0
epochs = 2000
batchSize = 8
workernum = 8
subPath = Path("edge_grouping/fourth_try")
save = Path("/root/perceptual_grouping/weight", subPath)
save.mkdir(parents=True) if not save.exists() else None

writer = SummaryWriter(Path("/root/perceptual_grouping/log", subPath))

dataSet = CitySpace(Path("/root/perceptual_grouping/dataset/cityspace"))
net: EdgeGrouping = EdgeGrouping().cuda(device)

optimizer = torch.optim.Adam([
    {'params': net.bottom.parameters()},
    {'params': net.surface.parameters()},
    {'params': net.backend.parameters()},
], lr=1e-3, betas=(0.9, 0.999), )


def forward(image, edge, bgt, tm, net: EdgeGrouping):
    res = net(edge)
    loss = net.balance_loss(res, tm)
    acc = net.accuracy(res, tm)
    return res, loss, acc, {'loss': loss.cpu(), "acc": acc.cpu()}


def loging(perfix, epoch, step, used_time, dataset_size, batch_size, tensorboard=True, **addentional):
    print(f"{perfix} epoch{epoch}", f"step{step}", f"samples {step % dataset_size}/{dataset_size}",
          f"net spend {format(used_time / batch_size, '.6f')}s",
          sep="    ", end="    ")

    for name in addentional:
        print(f"{perfix}_{name} {format(res[name], '.9f')}")
        writer.add_scalar(f"{perfix}_{name}", res[name], step) if tensorboard else None
    writer.flush() if tensorboard else None


def vision(perfix, step, image, edge, bgt, tm, output):
    # 可视化
    ins = bgt[0, 3]
    edge_ins = torch.zeros_like(ins)
    edge_ins_prop = vision_transaction_matrix(output[0])[0]
    uni = list(torch.unique(ins))
    shuffle(uni)
    for i, num in enumerate(uni):
        edge_ins[ins == num] = i + 5
    edge_ins[bgt[0, 0] == 0] = 0
    edge_ins_prop[bgt[0, 0] == 0] = 0

    writer.add_image(f"{perfix}_node_result", render_color(edge_ins_prop), step)
    writer.add_image(f"{perfix}_node_ground_truth", render_color(edge_ins), step)
    writer.add_image(f"{perfix}_input_edge", render_color(edge[0, 0]), step)
    writer.flush()


step, val_step = 0, 0
for epoch in range(epochs):
    net.train()
    train = DataLoader(dataSet.get_train(), shuffle=True, num_workers=workernum, batch_size=batchSize)
    start_time = time.time()
    for index, batch in enumerate(train):
        image, edge, bgt, tm, gt = to_device(device, *batch)
        output, loss, acc, res = forward(image, edge, bgt, tm, net)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        used_time = time.time() - start_time

        loging('train', epoch, step, used_time, dataset_size=len(train.dataset), batch_size=len(edge), **res)
        vision('train', step, image, edge, bgt, tm, output)

        step += len(edge)
        start_time = time.time()

    with torch.no_grad():
        net.eval()
        val = DataLoader(dataSet.get_val(), shuffle=False, pin_memory=False, num_workers=workernum,
                         batch_size=batchSize)
        total_loss, total_acc = torch.tensor(0, dtype=torch.float), torch.tensor(0, dtype=torch.float)
        start_time = used_time = time.time()
        bgt, index = None, None
        for index, batch in enumerate(val):
            image, edge, bgt, tm, gt = to_device(device, *batch)
            output, loss, acc, res = forward(image, edge, bgt, tm, net)
            total_loss += res['loss']
            total_acc += res['acc']
        used_time = time.time() - start_time
        loging('val', epoch, epoch, used_time, dataset_size=1, batch_size=len(val.dataset),
               **{'loss': total_loss / index, 'acc': total_acc / index})
        vision('val', epoch, image, edge, bgt, tm, output)
