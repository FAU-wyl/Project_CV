# config.py

# Exercise 1 中通过颜色区域统计得到的 Bayer 排列结果。
# 这个模式只用于 IMG_9939.npy 这份 numpy RAW 数据。
PATTERN_NPY = "GBRG"

# CR3 文件的 Bayer 相位。
# 这里是通过四种 Bayer 排列的可视化结果比较后确定的。
PATTERN_CR3 = "RGGB"

# 数据根目录。
DATA_DIR = "exercise_2_data"

# Exercise 1 使用的 numpy RAW 和相机 JPG 参考图。
EX1_NPY = f"{DATA_DIR}/01/IMG_9939.npy"
EX1_JPG = f"{DATA_DIR}/01/IMG_9939.JPG"

# Exercise 2/3/4/8 使用的单张 CR3 RAW。
EX2_RAW = f"{DATA_DIR}/02/IMG_4782.CR3"

# Exercise 5 是一组不同曝光时间的 RAW，用于验证传感器线性。
EX5_DIR = f"{DATA_DIR}/05"

# Exercise 6/7 是一组包围曝光 RAW，用于合成 HDR。
EX6_DIR = f"{DATA_DIR}/06"
