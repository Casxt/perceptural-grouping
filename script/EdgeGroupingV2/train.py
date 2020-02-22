import sys
from os import path

sys.path.append(path.join(path.join(path.dirname(__file__), '..'), ".."))

import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tool import to_device, render_color
from tool.CitySpeaceEdgeGrouping import EdgeGroupingDataset
from models import EdgeGroupingOnRestNet

# cityspace 数据集， 一切默认
device = 0
epochs = 2000
batchSize = 16
workernum = 8
subPath = Path("edge_grouping_v2/second_try")
save = Path("/root/edge_grouping_v2/weight", subPath)
save.mkdir(parents=True) if not save.exists() else None

writer = SummaryWriter(Path("/root/perceptual_grouping/log", subPath))

dataset_path = Path("/root/perceptual_grouping/dataset/edge_grouping")
net: EdgeGroupingOnRestNet = EdgeGroupingOnRestNet().cuda(device)

optimizer = torch.optim.Adam([
    {'params': net.bottom.parameters()},
    {'params': net.surface.parameters()},
    {'params': net.backend.parameters()},
], lr=5e-4, betas=(0.9, 0.999), )


def forward(image, instance_masking, instance_edge, edge, pool_edge, grouping_matrix, net: EdgeGroupingOnRestNet):
    res = net(edge, pool_edge)
    loss = net.mask_sigmoid_loss(res, grouping_matrix, pool_edge)
    acc = net.accuracy(res, grouping_matrix, pool_edge)
    return res, loss, acc, {'loss': loss.cpu(), "acc": acc.cpu()}


def loging(perfix, epoch, step, used_time, dataset_size, batch_size, tensorboard=True, **addentional):
    print(f"{perfix} epoch{epoch}", f"step{step}", f"samples {step % dataset_size}/{dataset_size}",
          f"net spend {format(used_time / batch_size, '.6f')}s",
          sep="    ", end="    ")

    for name in addentional:
        print(f"{perfix}_{name} {format(res[name], '.9f')}")
        writer.add_scalar(f"{perfix}_{name}", res[name], step) if tensorboard else None
    writer.flush() if tensorboard else None


def vision(perfix, step, image, instance_masking, instance_edge, edge, pool_edge, grouping_matrix, output):
    # 可视化
    # 使用mask遮罩不属于edge的部分
    b, c, h, w = grouping_matrix.shape
    edge = (pool_edge > 0).to(torch.int)
    # 注意下方尺度变换, 各个维度的位置及顺序已经经过测试, 切勿乱改
    mask = edge.view(b, c, 1).expand(b, -1, c).view(b, c, h, w)
    image = image[0]
    pool_edge = pool_edge[0, 0]
    instance_edge = render_color(instance_edge[0, 0])
    grouping_matrix = render_color(
        EdgeGroupingDataset.vision_transaction_matrix((grouping_matrix * mask)[0])[0] * pool_edge)
    output = render_color(EdgeGroupingDataset.vision_transaction_matrix(output[0])[0] * pool_edge)
    writer.add_image(f"{perfix}_image", torch.cat([image, instance_edge], dim=2), step)
    writer.add_image(f"{perfix}_grouping", torch.cat([grouping_matrix, output], dim=2), step)
    writer.flush()


step, val_step = 0, 0
for epoch in range(epochs):
    if epoch == 5:
        for p in optimizer.param_groups:
            p["lr"] = 1e-4
    net.train()
    dataset = EdgeGroupingDataset(Path(dataset_path, "train"))
    train = DataLoader(dataset, shuffle=True, num_workers=workernum, batch_size=batchSize)
    start_time = time.time()
    for index, batch in enumerate(train):
        image, instance_masking, instance_edge, edge, pool_edge, grouping_matrix = to_device(device, *batch)
        output, loss, acc, res = forward(image, instance_masking, instance_edge, edge, pool_edge, grouping_matrix, net)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        used_time = time.time() - start_time
        loging('train', epoch, step, used_time, dataset_size=len(train.dataset), batch_size=len(edge), **res)
        if index % 10 == 0:
            vision('train', step, image, instance_masking, instance_edge, edge, pool_edge, grouping_matrix, output)
        step += len(edge)
        start_time = time.time()

    with torch.no_grad():
        net.eval()
        dataset = EdgeGroupingDataset(Path(dataset_path, "val"))
        val = DataLoader(dataset, shuffle=False, pin_memory=False, num_workers=workernum,
                         batch_size=batchSize)
        total_loss, total_acc = torch.tensor(0, dtype=torch.float), torch.tensor(0, dtype=torch.float)
        start_time = used_time = time.time()
        image, instance_masking, instance_edge, edge, pool_edge, grouping_matrix, output, index = None, None, None, None, None, None, None, None
        for index, batch in enumerate(val):
            image, instance_masking, instance_edge, edge, pool_edge, grouping_matrix = to_device(device, *batch)
            output, loss, acc, res = forward(image, instance_masking, instance_edge, edge, pool_edge, grouping_matrix,
                                             net)
            total_loss += res['loss']
            total_acc += res['acc']

            if index % 10 == 0:
                loging('val_step', epoch, epoch * len(val.dataset) + index, used_time, dataset_size=len(train.dataset),
                       batch_size=len(batch),
                       **{'loss': total_loss / index, 'acc': total_acc / index})
                vision('val_step', epoch * len(val.dataset) + index, image, instance_masking, instance_edge, edge,
                       pool_edge, grouping_matrix, output)
        index += 1
        used_time = time.time() - start_time
        loging('val', epoch, epoch, used_time, dataset_size=len(train.dataset), batch_size=len(val.dataset),
               **{'loss': total_loss / index, 'acc': total_acc / index})
        vision('val', epoch, image, instance_masking, instance_edge, edge, pool_edge, grouping_matrix, output)
        torch.save(net.state_dict(), Path(save, f"epoch{epoch}-loss{total_loss / index}-acc{total_acc / index}.weight"))
