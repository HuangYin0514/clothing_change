#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import glob
import os
import warnings

import matplotlib.pyplot as plt
from PIL import Image

warnings.filterwarnings("ignore")


# -------------------------- 参数配置（argparse） --------------------------
def get_args():
    parser = argparse.ArgumentParser(description="批量读取文件夹图片并网格可视化")
    # 必选参数：图片目录
    parser.add_argument("--img_dir", type=str, required=True, help="图片文件夹路径")
    # 可选参数
    parser.add_argument("--max_imgs", type=int, default=100, help="最多加载图片数量，默认100")
    parser.add_argument("--rows", type=int, default=10, help="网格行数，默认10")
    parser.add_argument("--cols", type=int, default=5, help="网格列数，默认5")
    return parser.parse_args()


# 支持的图片后缀
IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]


def display_images_from_dir(image_dir, max_images=100, grid_rows=10, grid_cols=5):
    # 收集所有图片
    image_paths = []
    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext), recursive=False))
        image_paths.extend(glob.glob(os.path.join(image_dir, ext.upper()), recursive=False))

    # 去重并截断数量
    image_paths = list(set(image_paths))[:max_images]
    num_images = len(image_paths)
    total_slots = grid_rows * grid_cols

    if num_images == 0:
        print(f"错误：目录 {image_dir} 未找到任何图片！")
        return

    print(f"共找到 {num_images} 张图片，网格 {grid_rows}×{grid_cols}，最多展示 {total_slots} 张")

    plt.rcParams["figure.figsize"] = (20, 20)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for idx, img_path in enumerate(image_paths):
        if idx >= total_slots:
            break
        try:
            img = Image.open(img_path)
            ax = plt.subplot(grid_rows, grid_cols, idx + 1)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"{idx+1}", fontsize=8)
        except Exception as e:
            print(f"跳过损坏图片 {img_path}，错误：{e}")
            continue

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = get_args()
    display_images_from_dir(image_dir=args.img_dir, max_images=args.max_imgs, grid_rows=args.rows, grid_cols=args.cols)
