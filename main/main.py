import argparse
import os
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import util
from accelerate import Accelerator
from build_criterion import Build_Criterion
from build_optimizer import Build_Optimizer
from build_scheduler import Build_Scheduler
from core import test, train
from data import build_dataloader
from model import ReID_Net

import wandb

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
    total_params, train_params = util.get_model_param_info(reid_net)
    logger.info(f"Model: {type(reid_net).__name__}, " f"Total params: {total_params/1e6:.2f} M, " f"Trainable params: {train_params/1e6:.2f} M")
    if torch.cuda.device_count() > 1:
        logger.info("Accelerator is used!")
    else:
        logger.info("Accelerator is not used!")
        reid_net = nn.DataParallel(reid_net)  # 用于本地测试

    ######################################################################
    # Criterion
    criterion = Build_Criterion(config, dataset.num_train_pids)
    logger.info(f"Criterion:\t{criterion}")

    ######################################################################
    # Optimizer
    optimizer = Build_Optimizer(config, reid_net).optimizer
    logger.info(f"Optimizer:\t{type(optimizer).__name__}")

    ######################################################################
    # Scheduler
    scheduler = Build_Scheduler(config, optimizer).scheduler
    logger.info(f"Scheduler:\t{type(scheduler).__name__}")

    ######################################################################
    # Training & Evaluation
    logger.info("Start Training...")
    best_epoch, best_mAP, best_rank1 = 0, 0, 0
    reid_net, optimizer, train_loader = accelerator.prepare(reid_net, optimizer, train_loader)
    for epoch in range(0, config.OPTIMIZER.TOTAL_TRAIN_EPOCH):
        meter = train(config, reid_net, train_loader, criterion, optimizer, scheduler, device, epoch, logger, accelerator)
        logger.wandb(
            {
                "Epoch": epoch,
                "Lr": float(f"{optimizer.param_groups[0]['lr']:.1e}"),
                **{k: float(f"{v:.1e}") for k, v in meter.get_dict().items()},
            }
        )

        if epoch % config.TEST.EVAL_EPOCH == 0 or epoch == config.OPTIMIZER.TOTAL_TRAIN_EPOCH - 1:
            mAP, CMC = test(config, reid_net, query_loader, gallery_loader, device, logger)
            logger.wandb(
                {
                    "Dataset": config.DATA.TRAIN_DATASET,
                    "Test_epoch": epoch,
                    "mAP": mAP,
                    "Rank1": CMC[0],
                    "CMC": CMC,
                }
            )

            if mAP >= best_mAP:  # CMC[0] >= best_rank1
                best_epoch = epoch
                best_rank1 = CMC[0]
                best_mAP = mAP
                logger.wandb(
                    {
                        "best_epoch": best_epoch,
                        "best_mAP": best_mAP,
                        "best_rank1": best_rank1,
                    }
                )
                if epoch > 40:
                    util.save_model(model=reid_net, epoch=epoch, path_dir=os.path.join(config.SAVE.OUTPUT_PATH, "models/"), accelerator=accelerator)

    logger.info(f"Training done. Best model is: epoch: {best_epoch}, mAP: {best_mAP}%, Rank1: {best_rank1}%.")


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

    # 设置Wandb
    WANDB_KEY = "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"
    logger.wandb_start(api_key=WANDB_KEY, entity="yinhuang-team-projects", task_config=config.TASK)

    # 打印信息
    logger.info(f"Config is:\t{config}")
    logger.info(f"Device is:\t{device}")

    # 运行
    run(config, logger, device, accelerator)

    # 清理程序
    util.clean_pycache()
    wandb.finish()
