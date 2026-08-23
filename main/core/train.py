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

            outputs = reid_net(img)

            # Global
            global_feat = reid_net.module.global_pool(outputs["res4_featmap"]).view(B, -1)
            global_bn_feat = reid_net.module.global_bn_neck(global_feat)
            global_cls_score = reid_net.module.global_classifier(global_bn_feat)
            global_id_loss = criterion.ce_ls(global_cls_score, pid)
            meter.update({"global_id_loss": global_id_loss.item()})
            total_loss += global_id_loss
            global_tri_loss = criterion.tri(global_feat, pid)
            meter.update({"global_tri_loss": global_tri_loss.item()})
            total_loss += global_tri_loss

            # Hierarchical
            res2_featmap = reid_net.module.l2_pool(outputs["res2_featmap"]).view(B, -1)
            res2_bn_feat = reid_net.module.l2_bn_neck(res2_featmap)
            res2_cls_score = reid_net.module.l2_classifier(res2_bn_feat)
            res2_id_loss = criterion.ce_ls(res2_cls_score, pid)
            meter.update({"res2_id_loss": res2_id_loss.item()})
            total_loss += res2_id_loss

            res3_featmap = reid_net.module.l3_pool(outputs["res3_featmap"]).view(B, -1)
            res3_bn_feat = reid_net.module.l3_bn_neck(res3_featmap)
            res3_cls_score = reid_net.module.l3_classifier(res3_bn_feat)
            res3_id_loss = criterion.ce_ls(res3_cls_score, pid)
            meter.update({"res3_id_loss": res3_id_loss.item()})
            total_loss += res3_id_loss

            # 反事实换衣特征
            res2_featmap = reid_net.module.l2_pool(outputs["res2_featmap_aug"]).view(B, -1)
            res2_bn_feat = reid_net.module.l2_bn_neck(res2_featmap)
            res2_cls_score = reid_net.module.l2_classifier(res2_bn_feat)
            res2_id_loss = criterion.ce_ls(res2_cls_score, pid)
            meter.update({"res2_id_loss": res2_id_loss.item()})
            total_loss += res2_id_loss

            res3_featmap = reid_net.module.l3_pool(outputs["res3_featmap_aug"]).view(B, -1)
            res3_bn_feat = reid_net.module.l3_bn_neck(res3_featmap)
            res3_cls_score = reid_net.module.l3_classifier(res3_bn_feat)
            res3_id_loss = criterion.ce_ls(res3_cls_score, pid)
            meter.update({"res3_id_loss": res3_id_loss.item()})
            total_loss += res3_id_loss

            optimizer.zero_grad()
            accelerator.backward(total_loss)
            optimizer.step()

    return meter
