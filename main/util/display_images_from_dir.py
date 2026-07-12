#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

warnings.filterwarnings("ignore")

OUTPUT_SAVE_PATH = "/kaggle/working/clothing_change/image_grid_output.png"


def get_args():
    parser = argparse.ArgumentParser(description="批量读取文件夹图片并网格可视化，保存图片到/kaggle/working/clothing_change")
    parser.add_argument("--img_dir", type=str, required=True, help="图片文件夹路径")
    parser.add_argument("--max_imgs", type=int, default=100, help="最多加载图片数量，默认100")
    parser.add_argument("--rows", type=int, default=10, help="网格行数，默认10")
    parser.add_argument("--cols", type=int, default=5, help="网格列数，默认5")
    return parser.parse_args()


IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]


def display_images_from_dir(image_dir, max_images=100, grid_rows=10, grid_cols=5):
    image_paths = []
    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext), recursive=False))
        image_paths.extend(glob.glob(os.path.join(image_dir, ext.upper()), recursive=False))

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

    # ========== 关键修复：自动创建父文件夹 ==========
    save_dir = os.path.dirname(OUTPUT_SAVE_PATH)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    plt.savefig(OUTPUT_SAVE_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"网格图片已保存至: {OUTPUT_SAVE_PATH}")


if __name__ == "__main__":
    args = get_args()
    display_images_from_dir(image_dir=args.img_dir, max_images=args.max_imgs, grid_rows=args.rows, grid_cols=args.cols)
