import sys
from os import path

from models.Transform import subsequent_mask

sys.path.append(path.join(path.join(path.dirname(__file__), '..'), ".."))

import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tool import to_device, render_color
from tool.CitySpeaceEdgeGrouping import EdgeGroupingDataset
from models import EdgeGroupingOnTransform, make_image_model
from models.loss import k_loss, mask_bce_loss
from models.accuracy import k_accuracy, topk_accuracy

# cityspace 数据集， 一切默认
device = 0
epochs = 2000
batchSize = 6
workernum = 8
subPath = Path("hgru/first_try")
save = Path("/root/perceptual_grouping/weight", subPath)
save.mkdir(parents=True) if not save.exists() else None

writer = SummaryWriter(Path("/root/perceptual_grouping/log", subPath))

dataset_path = Path("/root/perceptual_grouping/dataset/edge_grouping")
net: EdgeGroupingOnTransform = make_image_model(input_channel=64, output_channel=784, max_len=784 * 512).cuda(device)

optimizer = torch.optim.Adam([
    {'params': net.parameters()},
], lr=5e-4, betas=(0.9, 0.999), )


def chunk_image(data, block_size):
    """
    将b, 1, ,h, w 图像分割为小方块, 返回 b, n, block_size, block_size 尺寸数据
    由于使用cat函数, 先对dim=3 做chunk, 再对dim=2 做chunk
    如果直接获取返回, 则应该先对dim=2 做chunk, 再对dim=3 做chunk
    """
    b, c, h, w = data.shape
    assert c == 1, "image channel should be 1"
    d2 = torch.chunk(data, w / block_size, dim=3)
    d2 = torch.cat(d2, dim=1)
    d3 = torch.chunk(d2, h / block_size, dim=2)
    return torch.cat(d3, dim=1)


def forward(net: EdgeGroupingOnTransform, image, instance_masking, instance_edge, instance_num, edge, pool_edge,
            grouping_matrix, nearby_matrix,
            adjacent_matrix):
    src = chunk_image(edge, 8)
    src = src.reshape((pool_edge.shape[0], -1, 8 * 8))
    # 添加开始占位符
    src = torch.cat([torch.zeros(1, 1, 4), src], dim=1)
    memory = net.encode(src, None)
    print("memory size:", memory.shape)
    output = torch.zeros(1, 1, 784).type_as(src.data)
    for i in range(784):
        print("decode pos:", i)
        out = net.decode(memory, None,
                         output,
                         subsequent_mask(output.size(1)).type_as(src.data))
        # 取出预测序列的最后一个元素
        out = out[:, -1:, :]
        out = net.generator(out)
        output = torch.cat([output, out], dim=1)
    gm = output[:, 1:, :].reshape(grouping_matrix.shape)
    k = instance_num
    gm_l = mask_bce_loss(gm, grouping_matrix, pool_edge)
    k_l = k_loss(k, instance_num)
    gm_acc = topk_accuracy(gm, grouping_matrix, pool_edge, k=5)
    k_acc = k_accuracy(k, instance_num)
    return gm, k, gm_l + k_l, gm_acc, {'k_loss': k_l.cpu(), 'gm_loss': gm_l.cpu(), "top5_acc": gm_acc.cpu(),
                                       'k_acc': k_acc.cpu()}


def loging(perfix, epoch, step, used_time, dataset_size, batch_size, tensorboard=True, **addentional):
    print(f"{perfix} epoch{epoch}", f"step{step}", f"samples {step % dataset_size}/{dataset_size}",
          f"net spend {format(used_time / batch_size, '.6f')}s",
          sep="    ", end="    ")

    for name in addentional:
        print(f"{perfix}_{name} {format(res[name], '.9f')}", end="    ")
        writer.add_scalar(f"{perfix}_{name}", res[name], step) if tensorboard else None
    print("")
    writer.flush() if tensorboard else None


def vision(perfix, step, image, instance_masking, instance_edge, instance_num, edge, pool_edge, grouping_matrix, output,
           k):
    # 可视化
    # 使用mask遮罩不属于edge的部分
    b, c, h, w = grouping_matrix.shape
    edge = (pool_edge[0, 0] > 0).to(torch.int)
    # 注意下方尺度变换, 各个维度的位置及顺序已经经过测试, 切勿乱改
    mask = edge.view(c, 1).expand(c, c).view(c, h, w)
    k = max(1, int(torch.argmax(k[0])))
    image = image[0]
    pool_edge = pool_edge[0, 0]
    instance_edge = render_color(instance_edge[0, 0])
    instance_num = instance_num[0]
    mask_gm = grouping_matrix[0] * mask
    mask_predict_gm = output[0] * mask
    outgm_kmeans_with_outk = render_color(
        EdgeGroupingDataset.vision_transaction_matrix_kmeans(mask_predict_gm, k)[0] * pool_edge)
    outgm_kmeans_with_gtk = render_color(
        EdgeGroupingDataset.vision_transaction_matrix_kmeans(mask_predict_gm, instance_num)[0] * pool_edge)

    gtgm_kmeans_with_gtk = render_color(
        EdgeGroupingDataset.vision_transaction_matrix_kmeans(mask_gm, instance_num)[0])
    writer.add_image(f"{perfix}_image", torch.cat([image, instance_edge], dim=2), step)
    writer.add_image(f"{perfix}_grouping",
                     torch.cat([gtgm_kmeans_with_gtk, outgm_kmeans_with_gtk, outgm_kmeans_with_outk], dim=2),
                     step)
    writer.flush()


step, val_step = 0, 0
for epoch in range(epochs):
    if epoch == 4:
        for p in optimizer.param_groups:
            p["lr"] = 5e-4
    elif epoch == 8:
        for p in optimizer.param_groups:
            p["lr"] = 1e-4
    elif epoch == 15:
        for p in optimizer.param_groups:
            p["lr"] = 5e-5

    net.train()
    dataset = EdgeGroupingDataset(Path(dataset_path, "train"))
    train = DataLoader(dataset, shuffle=True, num_workers=workernum, batch_size=batchSize)
    start_time = time.time()
    for index, batch in enumerate(train):
        image, instance_masking, instance_edge, instance_num, edge, pool_edge, grouping_matrix, nearby_matrix, adjacent_matrix = to_device(
            device, *batch)
        output, k, loss, acc, res = forward(net, image, instance_masking, instance_edge, instance_num, edge, pool_edge,
                                            grouping_matrix, nearby_matrix, adjacent_matrix)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        used_time = time.time() - start_time
        loging('train', epoch, step, used_time, dataset_size=len(train.dataset), batch_size=len(edge), **res)
        if index % 10 == 0:
            vision('train', step, image, instance_masking, instance_edge, instance_num, edge, pool_edge,
                   grouping_matrix, output, k)
        step += len(edge)
        start_time = time.time()

    with torch.no_grad():
        net.eval()
        dataset = EdgeGroupingDataset(Path(dataset_path, "val"))
        val = DataLoader(dataset, shuffle=False, pin_memory=False, num_workers=workernum,
                         batch_size=batchSize)
        total_res = dict()
        start_time = used_time = time.time()
        (image, instance_masking, instance_edge, instance_num, edge,
         pool_edge, grouping_matrix, nearby_matrix, output, k, index) = [None] * 11
        for index, batch in enumerate(val):
            image, instance_masking, instance_edge, instance_num, edge, pool_edge, grouping_matrix, nearby_matrix, adjacent_matrix = to_device(
                device, *batch)
            output, k, loss, acc, res = forward(net, image, instance_masking, instance_edge, instance_num, edge,
                                                pool_edge,
                                                grouping_matrix, nearby_matrix, adjacent_matrix)
            for key, value in res.items():
                total_res[key] = total_res.get(key, torch.zeros_like(value)) + value

            if index % 10 == 0:
                loging('val_step', epoch, epoch * len(val.dataset) + index, used_time, dataset_size=len(train.dataset),
                       batch_size=len(batch), **{k: v / index for k, v in total_res.items()})
                vision('val_step', epoch * len(val.dataset) + index, image, instance_masking, instance_edge,
                       instance_num, edge,
                       pool_edge, grouping_matrix, output, k)
        index += 1
        used_time = time.time() - start_time
        loging('val', epoch, epoch, used_time, dataset_size=len(train.dataset), batch_size=len(val.dataset),
               **{k: v / index for k, v in total_res.items()})
        vision('val', epoch, image, instance_masking, instance_edge, instance_num, edge, pool_edge, grouping_matrix,
               output, k)
        torch.save(
            net.state_dict(),
            Path(save,
                 f"epoch{epoch}-loss{(total_res['k_loss'] + total_res['gm_loss']) / index}-acc{total_res['k_acc'] / index}.weight")
        )
