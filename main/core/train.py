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

            backbone_feat_map, global_feat, global_bn_feat = reid_net(img)

            # Global
            global_cls_score = reid_net.module.global_classifier(global_bn_feat)
            global_id_loss = criterion.ce_ls(global_cls_score, pid)
            meter.update({"global_id_loss": global_id_loss.item()})
            total_loss += global_id_loss
            global_tri_loss = criterion.tri(global_feat, pid)
            meter.update({"global_tri_loss": global_tri_loss.item()})
            total_loss += global_tri_loss

            optimizer.zero_grad()
            accelerator.backward(total_loss)
            optimizer.step()

    return meter
