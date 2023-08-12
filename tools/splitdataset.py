import os
import random
import shutil

# 数据集路径
dataset_path = "/path/to/dataset"

# 定义训练集和验证集的比例
train_ratio = 0.8
val_ratio = 0.2

# 获取所有类别的文件夹
classes = os.listdir(dataset_path)

# 对每个类别的数据进行随机打乱
for class_name in classes:
    class_path = os.path.join(dataset_path, class_name)
    files = os.listdir(class_path)
    random.shuffle(files)

    # 计算切分索引
    num_files = len(files)
    train_split = int(train_ratio * num_files)

    # 切分数据集
    train_files = files[:train_split]
    val_files = files[train_split:]

    # 创建保存训练集和验证集的文件夹
    train_save_path = os.path.join(dataset_path, "train", class_name)
    val_save_path = os.path.join(dataset_path, "val", class_name)
    os.makedirs(train_save_path, exist_ok=True)
    os.makedirs(val_save_path, exist_ok=True)

    # 移动数据文件到对应的文件夹
    for file in train_files:
        src = os.path.join(class_path, file)
        dst = os.path.join(train_save_path, file)
        shutil.move(src, dst)

    for file in val_files:
        src = os.path.join(class_path, file)
        dst = os.path.join(val_save_path, file)
        shutil.move(src, dst)
