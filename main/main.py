import argparse
import os
import time
import warnings
from pathlib import Path

import torch
import util
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


def run(config):
    ######################################################################
    # Logger
    logger = util.Logger(path_dir=Path(config.SAVE.OUTPUT_PATH) / "logs", name="train_logger.log")

    ######################################################################
    # Logger
    logger.info(f"Config is:\t{config}")

    ######################################################################
    # Device
    device = torch.device(config.TASK.DEVICE)
    logger.info(f"Device is:\t{device}")

    # ######################################################################
    # # Data
    end = time.time()
    dataset, train_loader, query_loader, gallery_loader = build_dataloader(config)
    logger.info(f"Data loading time:\t{time.time() - end:.3f}")

    # ######################################################################
    # # Model
    reid_net = ReID_Net(config, dataset.num_train_pids).to(device)
    total_params, train_params = util.get_model_param_info(reid_net)
    logger.info(f"Model: {type(reid_net).__name__}, " f"Total params: {total_params/1e6:.2f} M, " f"Trainable params: {train_params/1e6:.2f} M")

    # ######################################################################
    # # Criterion
    criterion = Build_Criterion(config, dataset.num_train_pids)
    logger.info(f"Criterion:\t{criterion}")

    # # ######################################################################
    # # Optimizer
    optimizer = Build_Optimizer(config, reid_net).optimizer
    logger.info(f"Optimizer:\t{type(optimizer).__name__}")

    # ######################################################################
    # # Scheduler
    scheduler = Build_Scheduler(config, optimizer).scheduler
    logger.info(f"Scheduler:\t{type(scheduler).__name__}")

    # ######################################################################
    # Training & Evaluation
    logger.info("Start Training...")
    best_epoch, best_mAP, best_rank1 = 0, 0, 0
    for epoch in range(0, config.OPTIMIZER.TOTAL_TRAIN_EPOCH):
        meter = train(config, reid_net, train_loader, criterion, optimizer, scheduler, device, epoch, logger)
        logger.wandb(
            {
                "Epoch": epoch,
                "Lr": optimizer.param_groups[0]["lr"],
                **meter.get_dict(),
            }
        )

        if epoch % config.TEST.EVAL_EPOCH == 0 or epoch == config.OPTIMIZER.TOTAL_TRAIN_EPOCH - 1:
            mAP, CMC = test(config, reid_net, query_loader, gallery_loader, device, logger)
            logger.wandb(
                {
                    "Dataset": config.DATA.TRAIN_DATASET,
                    "test_epoch": epoch,
                    "mAP": mAP,
                    "Rank1": CMC[0],
                    "CMC": CMC,
                }
            )

            is_best_map_flag = mAP >= best_mAP  # is_best_rank_flag = CMC[0] >= best_rank1
            if is_best_map_flag:
                best_epoch = epoch
                best_rank1 = CMC[0]
                best_mAP = mAP
                logger.wandb(
                    {
                        "Best_epoch": best_epoch,
                        "Best_mAP": best_mAP,
                        "Best_Rank1": best_rank1,
                    }
                )
                if epoch > 40:
                    util.save_model(model=reid_net, epoch=epoch, path_dir=os.path.join(config.SAVE.OUTPUT_PATH, "models/"))

    logger.info(f"Training done. Best model is: epoch: {best_epoch}, mAP: {best_mAP}%, Rank1: {best_rank1}%.")


if __name__ == "__main__":
    args = get_args()
    config = util.load_config(args.config_file, args.opts)
    util.set_seed_torch(config.TASK.SEED)

    # 填入你的完整密钥
    api_key = "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"
    # 强制重新登录，覆盖旧缓存
    wandb.login(key=api_key, relogin=True)

    # 初始化wandb
    wandb.init(
        entity="yinhuang-team-projects",
        project=config.TASK.PROJECT,
        name=config.TASK.NAME,
        notes=config.TASK.NOTES,
        tags=config.TASK.TAGS,
        config=config,
    )
    run(config)
    util.clean_pycache()
    wandb.finish()
