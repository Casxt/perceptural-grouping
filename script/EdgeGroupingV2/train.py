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
subPath = Path("edge_grouping_v2/third_try")
save = Path("/root/edge_grouping_v2/weight", subPath)
save.mkdir(parents=True) if not save.exists() else None

writer = SummaryWriter(Path("/root/perceptual_grouping/log", subPath))

dataset_path = Path("/root/perceptual_grouping/dataset/edge_grouping")
net: EdgeGroupingOnRestNet = EdgeGroupingOnRestNet().cuda(device)

optimizer = torch.optim.Adam([
    {'params': net.bottom.parameters()},
    {'params': net.surface.parameters()},
    {'params': net.backend.parameters()},
], lr=1e-3, betas=(0.9, 0.999), )


def forward(image, instance_masking, instance_edge, instance_num, edge, pool_edge, grouping_matrix, nearby_matrix,
            net: EdgeGroupingOnRestNet):
    gm, k = net(edge)
    gm_loss = net.balance_bce_loss(gm, grouping_matrix)
    k_loss = net.mask_sigmoid_loss(k, grouping_matrix)
    gm_acc = net.topk_accuracy(gm, grouping_matrix, k=5)
    k_acc = net.k_accuracy(k, instance_num)
    return gm, k, gm_loss + k_loss, gm_acc, {'k_loss': k_loss.cpu(), 'gm_loss': gm_loss.cpu(), "top5_acc": gm_acc.cpu(),
                                             'k_acc': k_acc.cpu()}


def loging(perfix, epoch, step, used_time, dataset_size, batch_size, tensorboard=True, **addentional):
    print(f"{perfix} epoch{epoch}", f"step{step}", f"samples {step % dataset_size}/{dataset_size}",
          f"net spend {format(used_time / batch_size, '.6f')}s",
          sep="    ", end="    ")

    for name in addentional:
        print(f"{perfix}_{name} {format(res[name], '.9f')}")
        writer.add_scalar(f"{perfix}_{name}", res[name], step) if tensorboard else None
    writer.flush() if tensorboard else None


def vision(perfix, step, image, instance_masking, instance_edge, edge, pool_edge, grouping_matrix, output, k):
    # 可视化
    # 使用mask遮罩不属于edge的部分
    b, c, h, w = grouping_matrix.shape
    edge = (pool_edge > 0).to(torch.int)
    # 注意下方尺度变换, 各个维度的位置及顺序已经经过测试, 切勿乱改
    mask = edge.view(b, c, 1).expand(b, -1, c).view(b, c, h, w)
    k = torch.argmax(k[0])
    image = image[0]
    pool_edge = pool_edge[0, 0]
    instance_edge = render_color(instance_edge[0, 0])
    vision_gm_kmeans = render_color(
        EdgeGroupingDataset.vision_transaction_matrix_kmeans((grouping_matrix * mask)[0], k)[0] * pool_edge)
    vision_gm_trace = render_color(
        EdgeGroupingDataset.vision_transaction_matrix_trace((grouping_matrix * mask)[0])[0] * pool_edge)
    # output = render_color(EdgeGroupingDataset.vision_transaction_matrix(output[0])[0] * pool_edge)
    writer.add_image(f"{perfix}_image", torch.cat([image, instance_edge], dim=2), step)
    writer.add_image(f"{perfix}_grouping", torch.cat([grouping_matrix, vision_gm_kmeans, vision_gm_trace], dim=2), step)
    writer.flush()


step, val_step = 0, 0
for epoch in range(epochs):
    if epoch == 3:
        for p in optimizer.param_groups:
            p["lr"] = 1e-4
    net.train()
    dataset = EdgeGroupingDataset(Path(dataset_path, "train"))
    train = DataLoader(dataset, shuffle=True, num_workers=workernum, batch_size=batchSize)
    start_time = time.time()
    for index, batch in enumerate(train):
        image, instance_masking, instance_edge, instance_num, edge, pool_edge, grouping_matrix, nearby_matrix = to_device(
            device, *batch)
        output, k, loss, acc, res = forward(image, instance_masking, instance_edge, instance_num, edge, pool_edge,
                                            grouping_matrix, nearby_matrix, net)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        used_time = time.time() - start_time
        loging('train', epoch, step, used_time, dataset_size=len(train.dataset), batch_size=len(edge), **res)
        if index % 10 == 0:
            vision('train', step, image, instance_masking, instance_edge, edge, pool_edge, grouping_matrix, output, k)
        step += len(edge)
        start_time = time.time()

    with torch.no_grad():
        net.eval()
        dataset = EdgeGroupingDataset(Path(dataset_path, "val"))
        val = DataLoader(dataset, shuffle=False, pin_memory=False, num_workers=workernum,
                         batch_size=batchSize)
        total_loss, total_acc = torch.tensor(0, dtype=torch.float), torch.tensor(0, dtype=torch.float)
        start_time = used_time = time.time()
        image, instance_masking, instance_edge, instance_num, edge, pool_edge, grouping_matrix, nearby_matrix, output, index = None, None, None, None, None, None, None, None
        for index, batch in enumerate(val):
            image, instance_masking, instance_edge, instance_num, edge, pool_edge, grouping_matrix, nearby_matrix = to_device(
                device, *batch)
            output, k, loss, acc, res = forward(image, instance_masking, instance_edge, instance_num, edge, pool_edge,
                                                grouping_matrix, nearby_matrix,
                                                net)
            total_loss += loss
            total_acc += acc

            if index % 10 == 0:
                loging('val_step', epoch, epoch * len(val.dataset) + index, used_time, dataset_size=len(train.dataset),
                       batch_size=len(batch),
                       **{'loss': total_loss / index, 'acc': total_acc / index})
                vision('val_step', epoch * len(val.dataset) + index, image, instance_masking, instance_edge, edge,
                       pool_edge, grouping_matrix, output, k)
        index += 1
        used_time = time.time() - start_time
        loging('val', epoch, epoch, used_time, dataset_size=len(train.dataset), batch_size=len(val.dataset),
               **{'loss': total_loss / index, 'acc': total_acc / index})
        vision('val', epoch, image, instance_masking, instance_edge, edge, pool_edge, grouping_matrix, output)
        torch.save(net.state_dict(), Path(save, f"epoch{epoch}-loss{total_loss / index}-acc{total_acc / index}.weight"))
