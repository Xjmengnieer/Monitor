import os
import torch
import random

import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
# from dir2id import dir2id
from .init_train_cate import init_cates

class ImageNet(Dataset):
    def __init__(self, root, num_classes, transforms = None) -> None:
        super().__init__()
        self.root = root
        self.num_classes = num_classes
        self.imgs = self._getImgs_()
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def _getImgs_(self):
        sub_dirs = os.listdir(self.root)

        # assert len(sub_dirs) == len(init_cates), 'the categories\'s nums doesn\'t match the init_cates\
        #     please make sure the category is right'

        imgs = []
        for sub_dir in sub_dirs:
            assert sub_dir in init_cates, f'the {sub_dir} is not in {init_cates}'

            sub_dir_path = os.path.join(self.root, sub_dir)
            for img in os.listdir(sub_dir_path):
                imgs.append((os.path.join(sub_dir_path, img), init_cates[sub_dir]['idx']))
        
        return imgs

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        img = Image.open(img_path).convert('RGB')

        if self.transforms:
            b_trans = []
            for trans in self.transforms.transforms:
                if 'mosaic' in str(trans.__class__):
                    p = random.random()
                    if p < 0.3:
                        b_trans.append(transforms.CenterCrop(256))
                        m_trans = transforms.Compose(b_trans)
                        imgs = [transforms.CenterCrop(256)(img)]
                        labels = [label]
                        for _ in range(3):
                            random_index = random.randint(0, len(self.imgs)-1)
                            m_img_path, m_label = self.imgs[random_index]
                            m_img = Image.open(m_img_path).convert('RGB')
                            m_img = m_trans(m_img)
                            imgs.append(m_img)
                            labels.append(m_label)
                        img = trans(imgs, labels)
                        label = np.array(labels)
                    else:
                        continue

                else:
                    img = trans(img)
                    b_trans.append(trans)

        del b_trans

        label_one_hot = np.zeros(self.num_classes)
        label_one_hot[label] = 1

        return img, label_one_hot