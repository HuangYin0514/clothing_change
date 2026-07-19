from pathlib import Path

import util

import wandb


class Logger:
    def __init__(self, path_dir, name, accelerator=None):
        # 本地日志文件初始化
        log_path = Path(path_dir)
        util.make_dirs(log_path)
        self.file_path = str(log_path / name)

        # 分布式多进程控制
        self.is_main = True
        self.set_accelerator(accelerator)

        # wandb 相关缓存变量
        self._wandb_inited = False  # 标记是否已初始化wandb

        self.clear()

    def set_accelerator(self, accelerator):
        """动态绑定accelerator"""
        if self.accelerator is not None:
            self.accelerator = accelerator
            self.is_main = self.accelerator.is_main_process

    def wandb_start(self, api_key, entity, task_config):
        """
        执行 wandb.login + wandb.init
        :param api_key: wandb密钥字符串
        :param entity: 团队名
        :param task_config: 包含 PROJECT / NAME / NOTES / TAGS 的config对象
        """
        # 只有主进程才执行wandb初始化，避免多进程重复创建run
        if not self.is_main:
            return
        if self._wandb_inited:
            self.warn("Wandb已经初始化，无需重复启动")
            return

        # 登录wandb
        self.wandb_api_key = api_key
        self.wandb_entity = entity
        self.wandb_config = task_config

        settings = wandb.Settings(silent=True, show_info=False, show_warnings=False, show_errors=True)
        wandb.login(key=api_key, relogin=True)
        wandb.init(
            entity=entity,
            project=task_config.PROJECT,
            name=task_config.NAME,
            notes=task_config.NOTES,
            tags=task_config.TAGS,
            config=task_config,
            settings=settings,
        )
        self._wandb_inited = True
        self.info(f"Wandb run 初始化完成: {task_config.PROJECT}/{task_config.NAME}")

    def wandb_finish(self):
        """结束wandb run，上传剩余数据"""
        if self.is_main and self._wandb_inited:
            wandb.finish()
            self._wandb_inited = False
            self.info("Wandb run 已结束")

    def __call__(self, msg):
        """基础打印，无级别标签"""
        if not self.is_main:
            return
        msg = str(msg)
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        print(msg)

    def wandb(self, msg_dict):
        """上传指标到wandb，同时写入本地日志"""
        if self.is_main and self._wandb_inited:
            wandb.log(msg_dict)
        msg_str = "; ".join([f"{k}: {v}" for k, v in msg_dict.items()])
        self(msg_str)

    # 分级日志
    def info(self, msg):
        self(f"[INFO] {msg}")

    def warn(self, msg):
        self(f"[WARN] {msg}")

    def error(self, msg):
        self(f"[ERROR] {msg}")

    def debug(self, msg):
        self(f"[DEBUG] {msg}")

    def clear(self):
        if self.is_main:
            with open(self.file_path, "w", encoding="utf-8"):
                pass
