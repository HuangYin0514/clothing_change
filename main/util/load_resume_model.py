import os

import torch

from .make_dirs import make_dirs
from .os_walk import os_walk


def save_model(model, epoch, path_dir, accelerator, logger=None):
    # 创建保存模型的目录
    make_dirs(path_dir)

    # 保存模型
    model_file_path = os.path.join(path_dir, f"model_{epoch}.pth")
    accelerator.save(model.state_dict(), model_file_path)  # torch.save(model.state_dict(), model_file_path)

    # 控制保存模型的数量，最多保存max_file_num个模型文件
    max_file_num = 1
    root, _, files = os_walk(path_dir)
    if len(files) > max_file_num:
        for file in files:
            if ".pth" not in file:
                files.remove(file)  # 移除非模型文件
        file_iters = sorted([int(file.replace(".pth", "").split("_")[1]) for file in files], reverse=False)
        model_file_path = os.path.join(root, f"model_{file_iters[0]}.pth")
        os.remove(model_file_path)  # 移除非最新的模型文件

    # 输出结果
    logger.info(f"保存模型至 {model_file_path}")


def resume_model(model, path, resume_epoch=None, logger=None):
    # 如果没有指定resume_epoch，则自动选择最新的模型文件
    if resume_epoch is None or resume_epoch == -1:
        root, _, files = os_walk(path)
        if len(files) > 0:
            file_iters = sorted([int(file.replace(".pth", "").split("_")[1]) for file in files], reverse=True)
            resume_epoch = file_iters[0]
        else:
            raise Exception("未找到模型文件")
    else:
        resume_epoch = resume_epoch

    # 加载模型权重文件
    model_path = os.path.join(path, f"model_{resume_epoch}.pth")
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    clean_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            clean_k = k[7:]
            clean_state[clean_k] = v
        else:
            clean_state[k] = v

    # 加载权重至模型
    missing, unexpected = model.load_state_dict(clean_state, strict=False)

    # 输出结果
    logger.info(f"加载权重：缺失key {len(missing)} 个，多余key {len(unexpected)} 个")
    if missing:
        logger.info("缺失参数前10个:", missing[:10])
    if unexpected:
        logger.info("多余参数前10个:", unexpected[:10])
    logger.info(f"Successfully resume model from {model_path}")
