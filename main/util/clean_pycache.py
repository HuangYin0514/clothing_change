import os
import shutil


def clean_pycache(root_dir="."):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 删除 __pycache__ 文件夹
        if "__pycache__" in dirnames:
            cache_path = os.path.join(dirpath, "__pycache__")
            shutil.rmtree(cache_path)
            # print(f"已删除: {cache_path}")
        # 删除 .pyc 文件
        for f in filenames:
            if f.endswith(".pyc"):
                pyc_path = os.path.join(dirpath, f)
                os.remove(pyc_path)
                # print(f"已删除文件: {pyc_path}")
    print("已删除: __pycache__文件夹、*.pyc 文件")


if __name__ == "__main__":
    clean_pycache()
