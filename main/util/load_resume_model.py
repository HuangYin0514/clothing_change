import os

import torch

from .make_dirs import make_dirs
from .os_walk import os_walk


def save_model(model, epoch, path_dir, accelerator):
    make_dirs(path_dir)
    model_file_path = os.path.join(path_dir, f"model_{epoch}.pth")
    # torch.save(model.state_dict(), model_file_path)
    accelerator.save(model.state_dict(), model_file_path)

    root, _, files = os_walk(path_dir)
    if len(files) > 1:
        for file in files:
            if ".pth" not in file:
                files.remove(file)  # 移除非模型文件
        file_iters = sorted([int(file.replace(".pth", "").split("_")[1]) for file in files], reverse=False)
        model_file_path = os.path.join(root, f"model_{file_iters[0]}.pth")
        os.remove(model_file_path)  # 移除非最新的模型文件


def resume_model(model, resume_epoch, path):
    # model_path = os.path.join(path, f"model_{resume_epoch}.pth")
    # # model.load_state_dict(torch.load(model_path, weights_only=False), strict=False)
    # # model.load_state_dict(torch.load(model_path), strict=False)
    # state_dict = torch.load(model_path)
    # # 打印不匹配key，方便排查
    # missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # print(f"加载权重：缺失key {len(missing)} 个，多余key {len(unexpected)} 个")
    # if missing:
    #     print("缺失参数:", missing[:10])
    # print(f"Successfully resume model from {model_path}")

    model_path = os.path.join(path, f"model_{resume_epoch}.pth")
    # 安全加载权重
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    # 清洗DDP module. 前缀
    clean_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            clean_k = k[7:]
            clean_state[clean_k] = v
        else:
            clean_state[k] = v

    # 加载权重
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    print(f"加载权重：缺失key {len(missing)} 个，多余key {len(unexpected)} 个")
    if missing:
        print("缺失参数前10个:", missing[:10])
    if unexpected:
        print("多余参数前10个:", unexpected[:10])
    print(f"Successfully resume model from {model_path}")
