import argparse
import os
import time
import warnings
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import util
from accelerate import Accelerator
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


def run(config, logger, device, accelerator, *args, **kwargs):
    ######################################################################
    # Data
    end = time.time()
    dataset, train_loader, query_loader, gallery_loader = build_dataloader(config)
    logger.info(f"Data loading time:\t{time.time() - end:.3f}")

    ######################################################################
    # Model
    reid_net = ReID_Net(config, dataset.num_train_pids)
    util.resume_model(reid_net, config.TEST.RESUME_EPOCH, path=os.path.join(config.SAVE.OUTPUT_PATH, "models/"))
    total_params, train_params = util.get_model_param_info(reid_net)
    logger.info(f"Model: {type(reid_net).__name__}, " f"Total params: {total_params/1e6:.2f} M, " f"Trainable params: {train_params/1e6:.2f} M")
    if torch.cuda.device_count() > 1:
        logger.info("Accelerator is used!")
    else:
        logger.info("Accelerator is not used!")
        reid_net = nn.DataParallel(reid_net)  # 用于本地测试

    ########################################################
    # 可视化
    ########################################################
    logger.info(f"Start Visualization...")
    visualization(config, reid_net, train_loader, query_loader, gallery_loader, logger, device)
    logger.info(f"Visualization Done!")


if __name__ == "__main__":
    # 获取命令参数
    args = get_args()
    config = util.load_config(args.config_file, args.opts)
    util.set_seed_torch(config.TASK.SEED)

    # Accelerator
    accelerator = Accelerator()
    device = accelerator.device  # device = torch.device(config.TASK.DEVICE)

    # Logger
    logger = util.Logger(path_dir=Path(config.SAVE.OUTPUT_PATH) / "logs", name="train_logger.log", accelerator=accelerator)

    # 打印信息
    logger.info(f"Config is:\t{config}")
    logger.info(f"Device is:\t{device}")

    run(config, logger, device, accelerator)

    # 清理程序
    util.clean_pycache()
