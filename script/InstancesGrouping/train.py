import sys
from os import path

sys.path.append(path.join(path.join(path.dirname(__file__), '..'), ".."))

import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tool import to_device, generator_img_by_node_feature, render_raw_img
from tool.CitySpeaceV2 import CitySpace
from models import InstanceGroupingV2 as InstanceGrouping

# cityspace 数据集， 一切默认
device = 0
epochs = 2000
batchSize = 12
workernum = 30
subPath = Path("instance_grouping/seventh_try")
save = Path("/root/perceptual_grouping/weight", subPath)
save.mkdir(parents=True) if not save.exists() else None

writer = SummaryWriter(Path("/root/perceptual_grouping/log", subPath))

dataSet = CitySpace(Path("/root/perceptual_grouping/dataset/cityspace"))
net: InstanceGrouping = InstanceGrouping().cuda(device)

optimizer = torch.optim.RMSprop([
    # {'params': net.mobile_net_v2_b0.parameters(), "lr": 0},
    # {'params': net.mobile_net_v2_b1.parameters(), "lr": 0},
    # {'params': net.mobile_net_v2_b2.parameters(), "lr": 0},
    # {'params': net.mobile_net_v2_b3.parameters(), "lr": 0},
    # {'params': net.mobile_net_v2_b4.parameters(), "lr": 0},
    # {'params': net.fuse_conv.parameters()},
    # {'params': net.edge_region_predict.parameters()},
    {'params': net.node_feature_grouping.parameters()}
], lr=1e-4)

step, val_step = 0, 0
for epoch in range(epochs):
    net.train()

    if epoch == 5:
        for p in optimizer.param_groups:
            p["lr"] = 1e-4
    elif epoch == 15:
        for p in optimizer.param_groups:
            p["lr"] = 1e-3
    elif epoch == 25:
        for p in optimizer.param_groups:
            p["lr"] = 1e-4

    train = DataLoader(dataSet.get_train(), shuffle=True, num_workers=workernum, batch_size=batchSize)
    for index, batch in enumerate(train):
        imgs, gts, edges, block_gt, raw_imgs = to_device(device, *batch)
        start_time = time.time()
        edge_predict, edge_region_predict, sorted_topk_index, node_output_feature = net(imgs, edges, block_gt)
        used_time = time.time() - start_time

        edge_loss = net.image_edge_loss(edge_predict, edges)
        # node edge loss, 注意此处0：1 保证维度是b, 1, r, c 否则维度将变为b, r, c
        node_edge_loss = net.node_edge_loss(edge_region_predict[:, 0:1], block_gt[:, 0:1])
        # node pos loss
        node_pos_loss = net.block_position_loss(edge_region_predict[:, 1:3], block_gt[:, 1:3])
        # node feature loss
        node_feature_loss, dist_map = net.node_grouping_loss_v2(sorted_topk_index, node_output_feature,
                                                                block_gt[:, 3:4])
        # 只约束edge的fuse层输出
        total_loss = edge_loss + node_edge_loss + node_pos_loss + node_feature_loss

        print(f"train epoch{epoch}", f"step{index}", f"samples {step % len(train.dataset)}/{len(train.dataset)}",
              f"spend {format(used_time / len(imgs), '.6f')}s",
              f"loss {format(total_loss, '.9f')}",
              f"edge_loss {format(edge_loss, '.9f')}",
              f"node_edge_loss {format(node_edge_loss, '.9f')}",
              f"node_pos_loss {format(node_pos_loss, '.9f')}",
              f"node_feature_loss {format(node_feature_loss, '.9f')}", sep="    ")

        writer.add_scalar("train_loss", total_loss, step)
        writer.add_scalar("train_edge_loss", edge_loss, step)
        writer.add_scalar("train_node_edge_loss", node_edge_loss, step)
        writer.add_scalar("train_node_pos_loss", node_pos_loss, step)
        writer.add_scalar("train_node_feature_loss", node_feature_loss, step)
        # 可视化
        vis = generator_img_by_node_feature(sorted_topk_index[0].cpu().detach(), node_output_feature[0].cpu().detach(),
                                            edge_region_predict[0].cpu().detach(),
                                            dist_map[0].cpu().detach())
        vis_edge_region_predict = torch.nn.functional.interpolate(
            edge_region_predict[0:1, 0:1].cpu().detach(), size=(600, 800)
        )[0].expand(3, -1, -1)

        writer.add_image("train_node_vision",
                         torch.cat(
                             (render_raw_img(raw_imgs[0].cpu().detach(), block_gt[0].cpu().detach(),
                                             edges[0].cpu().detach()), vis,
                              vis_edge_region_predict),
                             dim=2),
                         step)
        writer.add_image("train_node_edge",
                         torch.cat((block_gt[0, 0:1].cpu(), edge_region_predict[0, 0:1].cpu()), dim=2), step)
        writer.add_image("train_image",
                         torch.cat((raw_imgs[0].cpu().detach() + edges[0].cpu().detach().expand(3, -1, -1),
                                    edge_predict[0].cpu().detach().expand(3, -1, -1)), dim=2),
                         step)

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

        edge_predict, gts, edges, raw_imgs = None, None, None, None
        dist_map, edge_region_predict, sorted_topk_index, node_output_feature = None, None, None, None
        val = DataLoader(dataSet.get_val(), shuffle=False, pin_memory=False, num_workers=workernum,
                         batch_size=batchSize)
        total_time, index = 0, 0
        for index, batch in enumerate(val):
            imgs, gts, edges, block_gt, raw_imgs = to_device(device, *batch)

            start_time = time.time()
            edge_predict, edge_region_predict, sorted_topk_index, node_output_feature = net(imgs)
            used_time = time.time() - start_time

            total_time += used_time
            val_step += len(imgs)

            edge_loss = net.image_edge_loss(edge_predict, edges)
            total_edge_loss += edge_loss
            # node edge loss, 注意此处0：1 保证维度是b, 1, r, c 否则维度将变为b, r, c
            node_edge_loss = net.node_edge_loss(edge_region_predict[:, 0:1], block_gt[:, 0:1])
            total_node_edge_loss += node_edge_loss
            # node pos loss
            node_pos_loss = net.block_position_loss(edge_region_predict[:, 1:3], block_gt[:, 1:3])
            total_node_pos_loss += node_pos_loss
            # node feature loss
            node_feature_loss, dist_map = net.node_grouping_loss_v2(sorted_topk_index, node_output_feature,
                                                                    block_gt[:, 3:4])
            total_node_feature_loss += node_feature_loss

            print(f"valid epoch{epoch}", f"val_step{index}", f"val_samples {val_step}/{len(val.dataset)}",
                  f"spend {format(used_time / len(imgs), '.6f')}s",
                  f"loss {format(edge_loss + node_edge_loss + node_pos_loss + node_feature_loss, '.9f')}",
                  f"edge_loss {format(edge_loss, '.9f')}",
                  f"node_edge_loss {format(node_edge_loss, '.9f')}",
                  f"node_pos_loss {format(node_pos_loss, '.9f')}",
                  f"node_feature_loss {format(node_feature_loss / len(imgs), '.9f')}", sep="    ")

            writer.add_image("val_step_image",
                             torch.cat((raw_imgs[0].detach() + edges[0].detach().expand(3, -1, -1),
                                        edge_predict[0].detach().expand(3, -1, -1)),
                                       dim=2),
                             val_step)
            # 可视化
            vis = generator_img_by_node_feature(sorted_topk_index[0], node_output_feature[0], edge_region_predict[0],
                                                dist_map[0])
            vis_edge_region_predict = torch.nn.functional.interpolate(
                edge_region_predict[0:1, 0:1], size=(600, 800)
            )[0].expand(3, -1, -1)
            writer.add_image("val_step_node_vision",
                             torch.cat(
                                 (render_raw_img(raw_imgs[0].detach(), block_gt[0].detach(), edges[0].detach()), vis,
                                  vis_edge_region_predict), dim=2), step)

        total_edge_loss /= (index + 1)
        total_node_edge_loss /= (index + 1)
        total_node_feature_loss /= (index + 1)
        total_node_pos_loss /= (index + 1)
        total_loss = (total_edge_loss +
                      total_node_edge_loss +
                      total_node_feature_loss +
                      total_node_pos_loss)

        print(f"valid epoch{epoch}", f"val_step{index}", f"val_samples {val_step}/{len(val.dataset)}",
              f"loss {format(total_loss, '.9f')}",
              f"edge_loss {format(total_edge_loss, '.9f')}",
              f"node_edge_loss {format(total_node_edge_loss, '.9f')}",
              f"node_pos_loss {format(total_node_feature_loss, '.9f')}",
              f"node_feature_loss {format(total_node_pos_loss, '.9f')}", sep="    ")

        writer.add_scalar("val_loss", total_loss, val_step)
        writer.add_scalar("val_edge_loss", total_edge_loss, val_step)
        writer.add_scalar("val_node_edge_loss", total_node_edge_loss, val_step)
        writer.add_scalar("val_node_feature_loss", total_node_feature_loss, val_step)
        writer.add_scalar("val_node_pos_loss", total_node_pos_loss, val_step)

        writer.add_image("val_image", torch.cat((raw_imgs[0].detach() + edges[0].detach().expand(3, -1, -1),
                                                 edge_predict[0].detach().expand(3, -1, -1)),
                                                dim=2), val_step)
        # 可视化
        vis = generator_img_by_node_feature(sorted_topk_index[0].detach(), node_output_feature[0].detach(),
                                            edge_region_predict[0].detach(),
                                            dist_map[0].detach())
        vis_edge_region_predict = \
            torch.nn.functional.interpolate(edge_region_predict[0:1, 0:1].detach(), size=(600, 800),
                                            mode="bilinear",
                                            align_corners=False)[0].expand(3, -1, -1)
        writer.add_image("val_node_vision",
                         torch.cat(
                             (render_raw_img(raw_imgs[0].detach(), block_gt[0].detach(), edges[0].detach()), vis,
                              vis_edge_region_predict),
                             dim=2), step)

        torch.save(net.state_dict(), Path(save, f"epoch{epoch}-step{step}-val_step{val_step}-Loss{total_loss}.weight"))
        writer.flush()
