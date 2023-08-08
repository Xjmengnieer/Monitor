import random
import shutil
import os

from tqdm import tqdm
from init_train_cate import init_cates

img_root_path = '/home/data/monitor/init_train/'
init_train_json = '/home/data/monitor/init_train.json'

test_img_path = '/home/data/monitor/init_val/'

test_percent = 0.1     #percent of train_pic

sub_dirs = os.listdir(img_root_path)

nums = 0
infos = {}
for sub_dir in sub_dirs:
    assert sub_dir in init_cates, f'the {sub_dir} is out the train category'
    sub_dir_path = os.path.join(img_root_path, sub_dir)
    val_path = os.path.join(test_img_path, sub_dir)

    if not os.path.exists(val_path):
        os.makedirs(val_path, exist_ok=True)
    
    img_nums = os.listdir(sub_dir_path)
    val_num = int(test_percent * len(img_nums))

    print(f'the {sub_dir} has {len(img_nums)} imgs, and we random select {val_num} to be used for validing')

    for img_name in tqdm(random.sample(img_nums, val_num)):
        img_path = os.path.join(sub_dir_path, img_name)
        val_img_path = os.path.join(val_path, img_name)
        os.system(f'mv {img_path} {val_img_path}')



'''
#    move pic to train_dir
# '''
# for img_name in random.sample(os.listdir( img_root_path ), test_num):
#     if img_name != 'Thumbs.db':
#         mask_name = img_name.replace('jpg', 'png')
#         shutil.move(os.path.join(img_root_path, img_name), test_img_path)

#         # mask_name = img_name.replace('jpg','png')

#         shutil.move(os.path.join(mask_root_path, mask_name), test_mask_path)
#         print('move succeed')
#     else:
#         print('img name is not right')


# for class_dir in os.listdir(os.path.join(img_root_path,'train')):
#     test_img_path = os.path.join(img_root_path,'val', class_dir)
#     train_img_dir = os.path.join(img_root_path,'train',class_dir)
#
#     if not os.path.exists(test_img_path):
#         os.makedirs(test_img_path)
#
#     train_percent = 0.1  # percent of train_pic
#
#     total_img = len(os.listdir(train_img_dir))
#     train_num = int(train_percent * total_img)
#
#     for img_name in random.sample(os.listdir(train_img_dir), train_num):
#         if img_name != 'Thumbs.db':
#             shutil.move(os.path.join(train_img_dir, img_name), test_img_path)
#
#             # shutil.move(os.path.join(mask_root_path, mask_name), train_mask_path)
#             print('move succeed')
#         else:
#             print('img name is not right')
#
# for sub in os.listdir(img_root_path):
#     shutil.move(os.path.join(img_root_path, sub), train_img_path)
# for sub in os.listdir(mask_root_path):
#     shutil.move(os.path.join(mask_root_path, sub), train_mask_path)