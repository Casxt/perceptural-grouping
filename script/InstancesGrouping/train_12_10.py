import sys
from os import path

sys.path.append(path.join(path.join(path.dirname(__file__), '..'), ".."))

import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tool import cat_in_out_gt, to_device
from tool.CitySpeaceV2 import CitySpace
from models import InstanceGrouping

# cityspace 数据集， 一切默认
device = 1
epochs = 2000
batchSize = 1
workernum = 1
subPath = Path("instance_grouping/first_try")
save = Path("/root/perceptual_grouping/weight", subPath)
save.mkdir(parents=True) if not save.exists() else None

writer = SummaryWriter(Path("/root/perceptual_grouping/log", subPath))

# dataSet = BSDS500(Path("/root/perceptual_grouping/dataset/BSDS500"))
dataSet = CitySpace(Path("/root/perceptual_grouping/dataset/cityspace"))
net: InstanceGrouping = InstanceGrouping().cuda(device)
# writer.add_graph(net, torch.rand(1, 3, 600, 800).cuda(device), True)
# print("add_graph")
# exit()
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
    {'params': net.mobile_outputs_convs[4].parameters()},
    {'params': net.edge_region_feature.parameters()},
    {'params': net.edge_region_predict.parameters()},
    {'params': net.node_feature_grouping.parameters()}
], lr=1e-3)

# writer.add_graph(net, torch.rand(1, 3, 836, 2035).cuda(), False)

step, val_step = 0, 0
for epoch in range(epochs):
    net.train()

    if epoch == 2:
        for p in optimizer.param_groups:
            p["lr"] = 1e-4

    train = DataLoader(dataSet.get_train(), shuffle=True, pin_memory=True, num_workers=workernum, batch_size=batchSize)
    for index, batch in enumerate(train):
        losses = [torch.tensor([0.]).cuda(device)] * 6
        imgs, gts, edges, block_gt = to_device(device, *batch)
        start_time = time.time()
        e_1, e_2, e_3, e_4, e_5, e_6, \
        edge_region_predict, sorted_topk_index, node_output_feature = net(imgs)
        anss = (e_1, e_2, e_3, e_4, e_5, e_6)
        used_time = time.time() - start_time

        # edge loss
        # for i, ans in enumerate(anss):
        #     losses[i] += net.batch_binary_cross_entropy_loss(ans, edges)
        #     writer.add_scalar(f"tarin_edge_loss_conv{i}", losses[i].detach().cpu().numpy()[0] / len(imgs), step)
        losses[5] += net.batch_binary_cross_entropy_loss(e_6, edges)
        # node edge loss, 注意此处0：1 保证维度是b, 1, r, c 否则维度将变为b, r, c
        node_edge_loss = net.batch_binary_cross_entropy_loss(edge_region_predict[:, 0:1], block_gt[:, 0:1])
        # node pos loss
        node_pos_loss = net.block_position_loss(edge_region_predict[:, 1:3], block_gt[:, 1:3])
        # node feature loss
        node_feature_loss = net.node_grouping_loss(sorted_topk_index, node_output_feature, block_gt[:, 3:4])
        # 只约束edge的fuse层输出
        edge_loss = sum(losses)
        total_loss = edge_loss + node_edge_loss + node_pos_loss + node_feature_loss
        total_loss_val = total_loss.detach().cpu().numpy()[0]
        writer.add_scalar("train_edge_loss", total_loss_val, step)
        writer.add_image("train_image", cat_in_out_gt(imgs[0], anss[-1][0], edges[0]), step)

        print(f"train epoch{epoch}", f"step{index}", f"samples {step}/{len(train.dataset)}",
              f"spend {format(used_time / len(imgs), '.6f')}s",
              f"loss {format(total_loss_val, '.9f')}",
              f"edge_loss {format(edge_loss, '.9f')}",
              f"node_edge_loss {format(node_edge_loss, '.9f')}",
              f"node_pos_loss {format(node_pos_loss, '.9f')}",
              f"node_feature_loss {format(node_feature_loss, '.9f')}", sep="    ")

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        step += len(imgs)
        writer.flush()

    with torch.no_grad():
        net.eval()
        total_edge_loss = torch.tensor(0.).cuda(device)
        total_node_edge_loss = torch.tensor(0.).cuda(device)
        total_node_feature_loss = torch.tensor(0.).cuda(device)
        total_node_pos_loss = torch.tensor(0.).cuda(device)

        anss, gts, edges = None, None, None
        val = DataLoader(dataSet.get_val(), shuffle=False, pin_memory=False, num_workers=workernum,
                         batch_size=batchSize)
        total_time = 0
        for index, batch in enumerate(val):
            step_edge_losses = [torch.tensor([0.]).cuda(device)] * 6
            imgs, gts, edges, block_gt = to_device(device, *batch)

            start_time = time.time()
            e_1, e_2, e_3, e_4, e_5, e_6, \
            edge_region_predict, sorted_topk_index, node_output_feature = net(imgs)
            anss = (e_1, e_2, e_3, e_4, e_5, e_6)
            used_time = time.time() - start_time

            total_time += used_time
            val_step += len(imgs)
            # for i, ans in enumerate(anss):
            #     step_losses[i] = net.batch_binary_cross_entropy_loss(ans, edges).detach()
            #     total_edge_loss += step_losses[i]

            step_edge_losses[5] = net.batch_binary_cross_entropy_loss(e_6, edges).detach()

            # node edge loss, 注意此处0：1 保证维度是b, 1, r, c 否则维度将变为b, r, c
            node_edge_loss = net.batch_binary_cross_entropy_loss(edge_region_predict[:, 0:1], block_gt[:, 0:1])
            total_node_edge_loss += node_edge_loss
            # node pos loss
            node_pos_loss = net.block_position_loss(edge_region_predict[:, 1:3], block_gt[:, 1:3])
            total_node_pos_loss += node_pos_loss
            # node feature loss
            node_feature_loss = net.node_grouping_loss(sorted_topk_index, node_output_feature, block_gt[:, 3:4])
            total_node_feature_loss += node_feature_loss

            print(f"valid epoch{epoch}", f"val_step{index}", f"val_samples {val_step}/{len(val.dataset)}",
                  f"spend {format(used_time / len(imgs), '.6f')}s",
                  f"loss {format(sum(step_edge_losses) + node_edge_loss + node_pos_loss + node_feature_loss, '.9f')}",
                  f"edge_loss {format(sum(step_edge_losses), '.9f')}",
                  f"node_edge_loss {format(node_edge_loss, '.9f')}",
                  f"node_pos_loss {format(node_pos_loss, '.9f')}",
                  f"node_feature_loss {format(node_feature_loss / len(imgs), '.9f')}", sep="    ")

            writer.add_image("val_image_step", cat_in_out_gt(imgs[0], anss[-1][0], edges[0]), val_step)

        # for i, l in enumerate(losses):
        #     writer.add_scalar(f"val_loss_conv{i}", l.detach().cpu().numpy()[0] / len(val.dataset), val_step)

        total_loss = (total_edge_loss +
                      total_node_edge_loss +
                      total_node_feature_loss +
                      total_node_pos_loss)

        print(f"valid epoch{epoch}", f"val_step{index}", f"val_samples {val_step}/{len(val.dataset)}",
              f"loss {format(total_loss, '.9f')}",
              f"edge_loss {format(total_edge_loss, '.9f')}",
              f"node_edge_loss {format(total_node_edge_loss, '.9f')}",
              f"node_pos_loss {format(total_node_feature_loss, '.9f')}",
              f"node_feature_loss {format(total_node_pos_loss / len(imgs), '.9f')}", sep="    ")

        writer.add_scalar("val_loss", total_loss, val_step)
        writer.add_scalar("val_edge_loss", total_edge_loss, val_step)
        writer.add_scalar("val_node_edge_loss", total_node_edge_loss, val_step)
        writer.add_scalar("val_node_feature_loss", total_node_feature_loss, val_step)
        writer.add_scalar("val_node_pos_loss", total_node_pos_loss, val_step)

        writer.add_image("val_image", cat_in_out_gt(imgs[0], anss[-1][0], edges[0]), val_step)

        torch.save(net.state_dict(), Path(save, f"epoch{epoch}-step{step}-val_step{val_step}-Loss{total_loss}.weight"))
        writer.flush()
