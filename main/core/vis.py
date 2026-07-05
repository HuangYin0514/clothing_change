####################
# 参考代码
# https://www.kaggle.com/code/huangyin0514/cc-reid/output
####################
import random
from pathlib import Path

import cv2
import numpy as np
import reid
import torch
import util

from .test import get_data, get_distmat

# ====================== 全局常量统一管理 ======================
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.224], dtype=np.float32)
EPS = 1e-12
ALPHA_HEATMAP = 0.5
GRID_SPACING = 10
QUERY_EXTRA_SPACING = 20
BORDER_WIDTH = 2
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
IMG_W = 128
IMG_H = 256
TOPK_RANK = 10
RAND_MIN = 100000
RAND_MAX = 999999
PRINT_STEP = 100


# ====================== 通用工具函数 ======================
def tensor_to_bgr(img_tensor: torch.Tensor) -> np.ndarray:
    """标准化张量 [C,H,W] -> OpenCV BGR uint8 图像 HWC"""
    img = img_tensor.detach().cpu()
    for t, m, s in zip(img, IMAGENET_MEAN, IMAGENET_STD):
        t.mul_(s).add_(m).clamp_(0, 1)
    img_np = (img.numpy() * 255).astype(np.uint8)
    img_np = np.transpose(img_np, (1, 2, 0))
    return img_np[:, :, ::-1]


def norm_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """热力图归一化到0~255，防止除零错误"""
    h_min, h_max = heatmap.min(), heatmap.max()
    heatmap = 255 * (heatmap - h_min) / (h_max - h_min + EPS)
    return heatmap.astype(np.uint8)


def concat_three_imgs(ori: np.ndarray, cam: np.ndarray, mix: np.ndarray, h: int, w: int) -> np.ndarray:
    """原图、热力图、叠加图横向拼接一张大图"""
    total_w = 3 * w + 2 * GRID_SPACING
    grid = np.ones((h, total_w, 3), dtype=np.uint8) * 255
    grid[:, :w, :] = ori
    s1 = w + GRID_SPACING
    e1 = 2 * w + GRID_SPACING
    grid[:, s1:e1, :] = cam
    s2 = 2 * w + 2 * GRID_SPACING
    grid[:, s2:, :] = mix
    return grid


def add_img_border(img: np.ndarray, is_match: bool) -> np.ndarray:
    """给单张图添加匹配边框，统一缩放尺寸"""
    border_color = COLOR_GREEN if is_match else COLOR_RED
    img = cv2.copyMakeBorder(img, BORDER_WIDTH, BORDER_WIDTH, BORDER_WIDTH, BORDER_WIDTH, cv2.BORDER_CONSTANT, value=border_color)
    return cv2.resize(img, (IMG_W, IMG_H))


# ====================== 顶层入口函数 ======================
def visualization(config, reid_net, train_loader, query_loader, gallery_loader, logger, device):
    """可视化总入口"""
    visualization_heatmap(config, reid_net, query_loader, device, logger)
    visualization_rank(config, reid_net, train_loader, query_loader, gallery_loader, logger, device)


# ====================== GradCAM热力图可视化======================
def visualization_heatmap(config, reid_net, heatmap_loader, device, logger, *args, **kwargs):
    logger(f"==> Start Heatmap Results Visualization...")

    actmap_dir = util.make_dirs(Path(config.SAVE.OUTPUT_PATH) / "actmap")
    reid_net.eval()

    for idx, data in enumerate(heatmap_loader):
        if idx % PRINT_STEP == 0:
            logger(f"CAM Progress: {idx}/{len(heatmap_loader)}")

        img, pid, camid, clotheid = data
        img = img.to(device)
        B, C, H, W = img.shape
        cam = util.GradCAMpp(reid_net, target_layer=reid_net.module.backbone)

        for b_idx in range(B):
            single_img = img[b_idx : b_idx + 1]
            cam_map = cam(single_img)

            # 原图转BGR
            ori_bgr = tensor_to_bgr(img[b_idx])
            # 热力图上色
            cam_norm = norm_heatmap(cam_map)
            cam_color = cv2.applyColorMap(cam_norm, cv2.COLORMAP_JET)
            # 叠加融合
            mixed = (1 - ALPHA_HEATMAP) * ori_bgr + ALPHA_HEATMAP * cam_color
            mixed = np.clip(mixed, 0, 255).astype(np.uint8)
            # 拼接网格图
            grid_img = concat_three_imgs(ori_bgr, cam_color, mixed, H, W)

            # 保存文件
            pid_val = pid[b_idx].item()
            camid_val = camid[b_idx].item()
            rand_suffix = random.randint(RAND_MIN, RAND_MAX)
            save_path = actmap_dir / f"{pid_val}_{camid_val}_{rand_suffix}.jpg"
            cv2.imwrite(str(save_path), grid_img)


# ====================== Rank检索结果可视化======================
def visualization_rank(config, reid_net, train_loader, query_loader, gallery_loader, logger, device, *args, **kwargs):
    logger(f"==> Start Ranked Results Visualization...")
    reid_net.eval()
    ranked_dir = util.make_dirs(Path(config.SAVE.OUTPUT_PATH) / "rank")
    distmat = None

    if config.DATA.TRAIN_DATASET == "ltcc":
        with torch.no_grad():
            qf, q_pids, q_camids, q_clothids = get_data(query_loader, reid_net, device)
            gf, g_pids, g_camids, g_clothids = get_data(gallery_loader, reid_net, device)
        distmat = get_distmat(qf, gf, dist="cosine")

        CMC_CC, mAP_CC = reid.evaluate_ltcc(distmat, q_pids, g_pids, q_camids, g_camids, q_clothids, g_clothids, mode="CC")
        logger(f"CC mode | mAP: {mAP_CC * 100:.2f}% | R-1: {CMC_CC[0] * 100:.2f}% | Top20 CMC: {CMC_CC[:20]}")

    query_set, gallery_set = query_loader.dataset, gallery_loader.dataset
    num_q, num_g = distmat.shape
    logger(f"[Rank Visual] Query: {num_q} | Gallery: {num_g} | Top-{TOPK_RANK}")
    indices = np.argsort(distmat, axis=1)

    for q_idx in range(num_q):
        q_tensor, q_pid, q_camid, q_cloid = query_set[q_idx]
        q_bgr = tensor_to_bgr(q_tensor)
        q_bgr = cv2.resize(q_bgr, (IMG_W, IMG_H))
        q_bgr = cv2.copyMakeBorder(q_bgr, BORDER_WIDTH, BORDER_WIDTH, BORDER_WIDTH, BORDER_WIDTH, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        q_bgr = cv2.resize(q_bgr, (IMG_W, IMG_H))

        # 构建画布
        total_cols = TOPK_RANK + 1
        grid_w = total_cols * IMG_W + TOPK_RANK * GRID_SPACING + QUERY_EXTRA_SPACING
        grid_img = np.ones((IMG_H, grid_w, 3), dtype=np.uint8) * 255
        grid_img[:, :IMG_W, :] = q_bgr

        rank_idx = 1
        match_cnt = 0
        for g_idx in indices[q_idx]:
            g_tensor, g_pid, g_camid, g_cloid = gallery_set[g_idx]
            # 过滤同相机同ID干扰样本
            if q_pid == g_pid and q_camid == g_camid:
                continue
            # LTCC换衣模式：跳过同服装
            if q_cloid == g_cloid:
                continue

            is_match = g_pid == q_pid
            if is_match:
                match_cnt += 1

            g_bgr = tensor_to_bgr(g_tensor)
            g_bgr = cv2.resize(g_bgr, (IMG_W, IMG_H))
            g_bgr = add_img_border(g_bgr, is_match)

            # 计算绘图偏移
            offset = rank_idx * IMG_W + rank_idx * GRID_SPACING + QUERY_EXTRA_SPACING
            grid_img[:, offset : offset + IMG_W, :] = g_bgr

            rank_idx += 1
            if rank_idx > TOPK_RANK:
                break

        # 保存图片
        save_name = f"{q_pid}_{random.randint(RAND_MIN, RAND_MAX)}.jpg"
        cv2.imwrite(str(ranked_dir / save_name), grid_img)

        if (q_idx + 1) % PRINT_STEP == 0:
            logger(f"[Rank] Processed {q_idx+1}/{num_q} queries")
