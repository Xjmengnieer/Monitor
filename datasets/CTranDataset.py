import os
import torch
import random

import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
# from dir2id import dir2id
from .utils import init_cates

from datasets import Datasets
from typing import Type, Any, Callable, Union, List, Optional

@Datasets.register_dataset('CTranData')
class CTranData(Dataset):
    def __init__(self, root,
                 num_classes, 
                 known_labels = 0, 
                 testing = False,
                 **kwargs: Any) -> None:
        super().__init__()
        self.root = root
        self.num_classes = num_classes
        self.imgs = self._getImgs_()
        self.known_labels = known_labels
        self.testing = testing
        self.num_labels = num_classes

    def add_transforms(self, transforms):
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
        label_one_hot = torch.from_numpy(label_one_hot)

        unk_mask_indices = get_unk_mask_indices(img, self.testing, self.num_labels,self.known_labels)
        
        mask = label_one_hot.clone()
        mask.scatter_(0,torch.Tensor(unk_mask_indices).long() , -1)

        sample = {}
        sample['image'] = img
        sample['labels'] = label_one_hot
        sample['mask'] = mask
        # sample['imageIDs'] = image_ID

        return sample

import hashlib
def get_unk_mask_indices(image,testing,num_labels,known_labels,epoch=1):
    if testing:
        # for consistency across epochs and experiments, seed using hashed image array 
        random.seed(hashlib.sha1(np.array(image)).hexdigest())
        unk_mask_indices = random.sample(range(num_labels), (num_labels-int(known_labels)))
    else:
        # sample random number of known labels during training
        if known_labels>0:
            random.seed()
            num_known = random.randint(0,int(num_labels*0.75))
        else:
            num_known = 0

        unk_mask_indices = random.sample(range(num_labels), (num_labels-num_known))

    return unk_mask_indices