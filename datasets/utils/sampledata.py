import os
import random
import shutil

def sample_images(src_dir, dest_dir, target_class_ratio=10):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for root, _, files in os.walk(src_dir):
        for file in files:
            src_file = os.path.join(root, file)
            dest_file = os.path.join(dest_dir, file)

            # 按照target_class_ratio的比例决定是否复制图片
            if random.randint(1, target_class_ratio) == 1:
                shutil.copy(src_file, dest_file)

if __name__ == "__main__":
    src_dir = "data/frames/frames"  # 原始图片路径
    dest_dir = "data/frames"       # 复制后图片保存路径
    target_class_ratio = 10        # 采样比例，10:1

    sample_images(src_dir, dest_dir, target_class_ratio)
