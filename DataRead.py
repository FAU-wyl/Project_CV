# DataRead.py

import os
import numpy as np
from scipy.io import loadmat


def describe_array(name, arr):
    """
    打印 numpy 数组的基础信息。
    """

    print(f"Variable name: {name}")
    print(f"  type: {type(arr)}")
    print(f"  shape: {arr.shape}")
    print(f"  dtype: {arr.dtype}")

    # 只对数值数组打印 min/max
    if np.issubdtype(arr.dtype, np.number):
        try:
            print(f"  min: {np.nanmin(arr)}")
            print(f"  max: {np.nanmax(arr)}")
            print(f"  mean: {np.nanmean(arr)}")
        except Exception as e:
            print(f"  Could not calculate min/max/mean: {e}")

    print()


def read_one_mat_file(file_path):
    """
    读取并检查单个 .mat 文件。
    """

    print("=" * 80)
    print(f"Reading file: {file_path}")
    print("=" * 80)

    data = loadmat(file_path)

    print("All keys:")
    for key in data.keys():
        print(" ", key)

    print("\nUseful variables:")
    print("-" * 80)

    for key, value in data.items():
        # loadmat 会自动生成这些系统字段，不是我们要的数据
        if key.startswith("__"):
            continue

        if isinstance(value, np.ndarray):
            describe_array(key, value)
        else:
            print(f"Variable name: {key}")
            print(f"  type: {type(value)}")
            print()


def read_all_mat_files(data_dir="data"):
    """
    读取 data 目录下的所有 .mat 文件并输出信息。
    """

    if not os.path.exists(data_dir):
        print(f"Data directory does not exist: {data_dir}")
        return

    mat_files = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".mat"):
            mat_files.append(os.path.join(data_dir, filename))

    mat_files.sort()

    print(f"Found {len(mat_files)} mat files.")

    for file_path in mat_files:
        read_one_mat_file(file_path)


if __name__ == "__main__":
    read_all_mat_files("data")