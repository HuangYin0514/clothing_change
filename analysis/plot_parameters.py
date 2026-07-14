from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 外部配置（保留原有项目配置）
from config_plot import *
from matplotlib.ticker import MultipleLocator

# ===================== 全局常量 =====================
OUTPUT_DIR = Path("./analysis/results/ablation")
FIG_ROW = 1
FIG_COL = 1
FIG_W = 4 * FIG_COL
FIG_H = 3 * FIG_ROW
DPI = LATAX_DPI
SHOW_PLOT = False  # True开启弹窗预览，False直接保存

# 配色统一管理
COLOR_MAP = "#FD625E"
COLOR_RANK1 = "#71D4EB"


def plot_single_line_chart(x_data: list, map_data: list, x_label: str, save_name: str, y_lim: tuple = None, y_major_tick: int = None):
    """
    单轴折线图（仅mAP，对应 lossWeight 绘图）
    """
    fig, ax = plt.subplots(FIG_ROW, FIG_COL, figsize=(FIG_W, FIG_H), dpi=DPI)

    ax.plot(x_data, map_data, label="mAP", marker="o", markersize=MARKERSIZE, zorder=2)

    ax.set_xlabel(x_label)
    ax.set_ylabel("mAP (%)")
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    if y_major_tick is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_major_tick))

    ax.legend(framealpha=1)
    plt.tight_layout()

    save_path = OUTPUT_DIR / save_name
    plt.savefig(save_path, bbox_inches="tight")
    print(f"图像已保存到: {save_path.resolve()}")

    if SHOW_PLOT:
        plt.show()
    plt.close(fig)


def plot_dual_axis_line_chart(
    x_data: list,
    map_data: list,
    rank1_data: list,
    x_label: str,
    save_name: str,
):
    """
    双Y轴折线图：左轴mAP，右轴Rank-1
    """
    fig, ax1 = plt.subplots(FIG_ROW, FIG_COL, figsize=(FIG_W, FIG_H), dpi=DPI)
    ax2 = ax1.twinx()

    l1 = ax1.plot(x_data, map_data, color=COLOR_MAP, marker="o", markersize=MARKERSIZE, label="mAP", zorder=2)
    l2 = ax2.plot(x_data, rank1_data, color=COLOR_RANK1, marker="o", markersize=MARKERSIZE, label="Rank-1", zorder=2)

    ax1.set_xlabel(x_label)
    ax1.set_ylabel("mAP (%)")
    ax2.set_ylabel("Rank-1 (%)")

    handles = l1 + l2
    labels = [h.get_label() for h in handles]
    ax2.legend(handles, labels, framealpha=1)

    plt.tight_layout()
    save_path = OUTPUT_DIR / save_name
    plt.savefig(save_path, bbox_inches="tight")
    print(f"图像已保存到: {save_path.resolve()}")

    if SHOW_PLOT:
        plt.show()
    plt.close(fig)


def plot_dual_axis_bar_chart(x_data: list, map_data: list, rank1_data: list, x_label: str, save_name: str, bar_width: float = 0.2, bar_spacing: float = 0.05):
    """
    双Y轴分组柱状图：左轴mAP，右轴Rank-1
    """
    fig, ax1 = plt.subplots(FIG_ROW, FIG_COL, figsize=(FIG_W, FIG_H), dpi=DPI)
    ax2 = ax1.twinx()

    x_pos = np.arange(len(x_data))
    pos_map = x_pos + (bar_width / 2 + bar_spacing / 2)
    pos_rank1 = x_pos - (bar_width / 2 + bar_spacing / 2)

    b1 = ax1.bar(pos_map, map_data, width=bar_width, color=COLOR_MAP, label="mAP", zorder=2)
    b2 = ax2.bar(pos_rank1, rank1_data, width=bar_width, color=COLOR_RANK1, label="Rank-1", zorder=2)

    ax1.set_xlabel(x_label)
    ax1.set_ylabel("mAP (%)")
    ax2.set_ylabel("Rank-1 (%)")

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_data)
    ax2.grid(False)

    handles = [b1, b2]
    labels = [h.get_label() for h in handles]
    ax2.legend(handles, labels, framealpha=1)

    plt.tight_layout()
    save_path = OUTPUT_DIR / save_name
    plt.savefig(save_path, bbox_inches="tight")
    print(f"图像已保存到: {save_path.resolve()}")

    if SHOW_PLOT:
        plt.show()
    plt.close(fig)


# ===================== 绘图入口函数（业务层） =====================
def plot_parameter_lossWeight():
    x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    map_res = [64.339, 69.467, 69.78, 69.707, 71.221, 69.585, 70.863, 70.301, 70.968, 70.025, 70.946]
    plot_single_line_chart(x_data=x, map_data=map_res, x_label=r"Parameter $\lambda$", save_name="parameter_lossWeight_1.png", y_lim=(60, 80), y_major_tick=3)


def plot_parameter_viewNum():
    x = [2, 4, 8]
    map_res = [58.9, 59.1, 58.2]
    rank1_res = [70.3, 71.2, 69.3]
    plot_dual_axis_bar_chart(x_data=x, map_data=map_res, rank1_data=rank1_res, x_label=r"Parameter $M$", save_name="parameter_viewNum_bar.png")


def plot_parameter_partNum_Bar():
    x = [1, 2, 4, 8, 16]
    map_res = [58.6, 59.1, 58.3, 58.2, 57.4]
    rank1_res = [70.4, 71.2, 69.0, 69.3, 69.0]
    plot_dual_axis_bar_chart(x_data=x, map_data=map_res, rank1_data=rank1_res, x_label=r"Parameter $Np$", save_name="parameter_partNum_bar.png")


def plot_parameter_partNum_line():
    x = [1, 2, 4, 8, 16]
    map_res = [58.6, 59.1, 58.3, 58.2, 57.4]
    rank1_res = [70.4, 71.2, 69.0, 69.3, 69.0]
    plot_dual_axis_line_chart(x_data=x, map_data=map_res, rank1_data=rank1_res, x_label=r"Parameter $Np$", save_name="parameter_partNum_line.png")


if __name__ == "__main__":
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 按需启用绘图
    plot_parameter_lossWeight()
    plot_parameter_viewNum()
    plot_parameter_partNum_Bar()
    plot_parameter_partNum_line()
