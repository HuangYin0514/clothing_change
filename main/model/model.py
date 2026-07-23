import torch
import torch.nn as nn

from .layer import BN_Neck, GeneralizedMeanPoolingP, Linear_Classifier
from .net import resnet50, resnet50_ibn_a
from .process import HGE


# Backbone_R50 ------------------------------
class Backbone_R50(nn.Module):
    def __init__(self, backbone_type):
        super().__init__()

        resnet = None
        if backbone_type == "resnet50":
            resnet = resnet50(pretrained=True)
        if backbone_type == "resnet50_ibn_a":
            resnet = resnet50_ibn_a(pretrained=True)

        # Modifiy backbone
        resnet.layer4[0].downsample[0].stride = (1, 1)
        resnet.layer4[0].conv2.stride = (1, 1)

        # Backbone structure
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)

        self.layer1 = resnet.layer1  # 3 blocks
        self.layer2 = resnet.layer2  # 4 blocks
        self.layer3 = resnet.layer3  # 6 blocks
        self.layer4 = resnet.layer4  # 3 blocks

        self.hge_res1 = HGE(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn_res1 = nn.BatchNorm2d(256)
        self.hge_res2 = HGE(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn_res2 = nn.BatchNorm2d(512)
        self.hge_res3 = HGE(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.bn_res3 = nn.BatchNorm2d(1024)

        self.act = nn.ReLU()

    def forward(self, img):
        # Layer 0
        out = self.layer0(img)
        res0_featmap = out

        # Layer 1
        out = self.layer1(out)
        hge1_out = self.hge_res1(out)
        hge1_out = self.bn_res1(hge1_out)
        hge1_out = self.act(hge1_out)
        out = out + hge1_out
        res1_featmap = out

        # Layer 2
        out = self.layer2(out)
        hge2_out = self.hge_res2(out)
        hge2_out = self.bn_res2(hge2_out)
        hge2_out = self.act(hge2_out)
        out = out + hge2_out
        res2_featmap = out

        # Layer 3
        out = self.layer3(out)
        hge3_out = self.hge_res3(out)
        hge3_out = self.bn_res3(hge3_out)
        hge3_out = self.act(hge3_out)
        out = out + hge3_out
        res3_featmap = out

        # Layer 4
        out = self.layer4(out)
        res4_featmap = out

        return res0_featmap, res1_featmap, res2_featmap, res3_featmap, res4_featmap


class ReID_Net(nn.Module):

    def __init__(self, config, num_pid):
        super().__init__()
        self.config = config

        BACKBONE_TYPE = config.MODEL.BACKBONE_TYPE

        # ------------- Backbone -----------------------
        self.backbone = Backbone_R50(BACKBONE_TYPE)

        # ------------- Global -----------------------
        self.GLOBAL_DIM = 2048
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_bn_neck = BN_Neck(self.GLOBAL_DIM)
        self.global_classifier = Linear_Classifier(self.GLOBAL_DIM, num_pid)

        # ------------- Hierarchical layer -----------------------
        hier_pool_list = nn.ModuleList()
        hier_bn_neck_list = nn.ModuleList()
        hier_classifier_list = nn.ModuleList()
        self.HIER_DIM = [256, 512, 1024]
        for i in range(3):
            hier_pool_list.append(nn.AdaptiveAvgPool2d(1))
            hier_bn_neck_list.append(BN_Neck(self.HIER_DIM[i]))
            hier_classifier_list.append(Linear_Classifier(self.HIER_DIM[i], num_pid))
        self.hier_pool_list = hier_pool_list
        self.hier_bn_neck_list = hier_bn_neck_list
        self.hier_classifier_list = hier_classifier_list

    def heatmap(self, img):
        B, C, H, W = img.shape
        res0_featmap, res1_featmap, res2_featmap, res3_featmap, res4_featmap = self.backbone(img)
        return res4_featmap

    def forward(self, img):
        B, C, H, W = img.shape

        # ------------- Global -----------------------
        res0_featmap, res1_featmap, res2_featmap, res3_featmap, res4_featmap = self.backbone(img)

        if not self.training:
            eval_feat_meter = []

            # ------------- Global -----------------------
            global_feat = self.global_pool(res4_featmap).view(B, self.GLOBAL_DIM)
            global_bn_feat = self.global_bn_neck(global_feat)
            eval_feat_meter.append(global_bn_feat)

            eval_feat = torch.cat(eval_feat_meter, dim=1)
            return eval_feat

        return res0_featmap, res1_featmap, res2_featmap, res3_featmap, res4_featmap
