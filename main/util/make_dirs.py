import shutil
from pathlib import Path
from typing import Union


def make_dirs(target_dir: Union[str, Path], clean: bool = False, verbose: bool = True) -> Path:
    """
    创建/初始化目录，合并原有两种逻辑
    :param target_dir: 目标目录，字符串或Path对象
    :param clean: True=先删除原有目录再重建；False=仅不存在时创建，保留原有内容
    :param verbose: 是否打印操作日志
    :return: 目录绝对路径Path对象
    """
    p = Path(target_dir).resolve()

    # 清空旧目录逻辑
    if clean and p.exists():
        try:
            shutil.rmtree(p)
            if verbose:
                print(f"[CLEAN] Removed old directory: {p}")
        except OSError as e:
            raise RuntimeError(f"删除目录失败 {p}：{e}") from e

    # 创建多级目录，已存在不会报错
    p.mkdir(parents=True, exist_ok=True)

    if verbose:
        if clean:
            print(f"[INIT] Successfully initialized dir: {p}")
        else:
            # 区分新建/已存在两种提示
            if p.exists() and not clean:
                print(f"Existed dirs: {p}")
            else:
                print(f"Successfully make dirs: {p}")
    return p
