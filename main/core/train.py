import torch
import util
from tqdm import tqdm


def train(config, reid_net, train_loader, criterion, optimizer, scheduler, device, epoch, logger, accelerator, *args, **kwargs):
    scheduler.step(epoch)
    reid_net.train()
    meter = util.MultiItemAverageMeter()
    for epoch, data in enumerate(tqdm(train_loader)):
        img, pid, camid, clotheid = data

        if config.MODEL.MODULE == "Lucky":
            B, C, H, W = img.size()
            total_loss = 0

            res0_featmap, res1_featmap, res2_featmap, res3_featmap, res4_featmap = reid_net(img)

            # Global
            global_feat = reid_net.module.global_pool(res4_featmap).view(B, reid_net.module.GLOBAL_DIM)
            global_bn_feat = reid_net.module.global_bn_neck(global_feat)
            global_cls_score = reid_net.module.global_classifier(global_bn_feat)
            global_id_loss = criterion.ce_ls(global_cls_score, pid)
            meter.update({"global_id_loss": global_id_loss.item()})
            total_loss += global_id_loss
            global_tri_loss = criterion.tri(global_feat, pid)
            meter.update({"global_tri_loss": global_tri_loss.item()})
            total_loss += global_tri_loss

            # Hierarchical layer
            hier_featmap_list = [res1_featmap, res2_featmap, res3_featmap]
            for i in range(3):
                hier_featmap = hier_featmap_list[i]
                hier_feat = reid_net.module.hier_pool_list[i](hier_featmap).view(B, reid_net.module.HIER_DIM[i])
                hier_bn_feat = reid_net.module.hier_bn_neck_list[i](hier_feat)
                hier_cls_score = reid_net.module.hier_classifier_list[i](hier_bn_feat)
                hier_id_loss = criterion.ce_ls(hier_cls_score, pid)
                meter.update({"hier_id_loss_{}".format(i): hier_id_loss.item()})
                total_loss += 0.1 * hier_id_loss
                hier_tri_loss = criterion.tri(hier_feat, pid)
                meter.update({"hier_tri_loss_{}".format(i): hier_tri_loss.item()})
                total_loss += 0.1 * hier_tri_loss

            optimizer.zero_grad()
            accelerator.backward(total_loss)
            optimizer.step()

    return meter
