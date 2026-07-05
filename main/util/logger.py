from pathlib import Path

import util

import wandb


class Logger:
    def __init__(self, path_dir, name):
        # 用pathlib优化路径拼接
        log_path = Path(path_dir)
        util.make_dirs(log_path)
        self.file_path = str(log_path / name)
        self.clear()

    def __call__(self, msg):
        """基础打印，无级别标签"""
        msg = str(msg)
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        print(msg)

    def wandb(self, msg):
        wandb.log(msg)
        msg = "; ".join([f"{k}: {v}" for k, v in msg.items()])
        self(msg)

    # 新增分级打印
    def info(self, msg):
        self(f"[INFO] {msg}")

    def warn(self, msg):
        self(f"[WARN] {msg}")

    def error(self, msg):
        self(f"[ERROR] {msg}")

    def debug(self, msg):
        self(f"[DEBUG] {msg}")

    def clear(self):
        with open(self.file_path, "w", encoding="utf-8"):
            pass

    # def __str__(self):
    #     line = "*" * 20
    #     return f"{line}\nConfig:{{'log_file': '{self.file_path}'}}\n{line}"
