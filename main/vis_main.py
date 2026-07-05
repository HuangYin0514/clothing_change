import argparse
import os
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import util
from core import visualization
from data import build_dataloader
from model import ReID_Net

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="main/cfg/test.yml", help="path to config file")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args


def run(config):
    ######################################################################
    # Logger
    logger = util.Logger(path_dir=Path(config.SAVE.OUTPUT_PATH) / "logs", name="vis_logger.log")

    ######################################################################
    # Logger
    logger.info(f"Config is:\t{config}")

    ######################################################################
    # Device
    device = torch.device(config.TASK.DEVICE)
    logger.info(f"Device is:\t {device}")

    ######################################################################
    # Data
    end = time.time()
    dataset, train_loader, query_loader, gallery_loader = build_dataloader(config)
    logger.info(f"Data loading time:\t{time.time() - end:.3f}")

    ######################################################################
    # Model
    reid_net = ReID_Net(config, dataset.num_train_pids).to(device)
    util.resume_model(reid_net, config.TEST.RESUME_EPOCH, path=os.path.join(config.SAVE.OUTPUT_PATH, "models/"))
    reid_net = nn.DataParallel(reid_net)  # 默认使用所有可见GPU，2卡会自动分配

    ########################################################
    # 可视化
    ########################################################
    logger.info(f"Start Visualization...")
    visualization(config, reid_net, train_loader, query_loader, gallery_loader, logger, device)
    logger.info(f"Visualization Done!")


if __name__ == "__main__":
    args = get_args()
    config = util.load_config(args.config_file, args.opts)
    util.set_seed_torch(config.TASK.SEED)
    run(config)
    util.clean_pycache()
