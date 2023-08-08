import os
import cv2
from tqdm import tqdm
from datasets.validIds import validIds

imgnet_path = '/home/data/monitor/train'

dirs = []
for key, value in validIds.items():
    dirs.append(value['dir'])

    dir_path = os.path.join(imgnet_path, value['dir'])