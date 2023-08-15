import os
import torch

import numpy as np

from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from monitor_model import bulid_classifier
from utils import *
from datasets.Imagenet import ImageNet


def main(args):
    config = config_load(args.config)
    val_trans = transforms.Compose([
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize(mean = config.datasets.mean, std = config.datasets.std)
    ])
    val_dataset = ImageNet(config.datasets.val.root, config.model.num_classes, transforms=val_trans)
    val_dataloader = DataLoader(val_dataset, batch_size=config.datasets.batch_size)
    data_root_path = config.datasets.val.root
    print(f'datasets root path is: {data_root_path}')
    sub_dirs = os.listdir(data_root_path)

    if config.inference.load_from:
        print(f'load from {config.inference.load_from}')
        model = torch.load(config.inference.load_from, map_location='cpu').eval().cuda()
    else:
        model = bulid_classifier(config).eval().cuda()

    model.eval()
    par = tqdm(val_dataloader)
    tps = 0
    gtA = 0
    ppA = 0
    threshold=0.9
    with torch.no_grad():
        for batch in par:
                par.set_description(f'validation: ')
                inputs, label = batch[0].cuda(),batch[1].cuda()
                output = model(inputs)
                pred = torch.sigmoid(output)
                # print(f'pred: {pred}')
                pred = (pred > threshold).long()
                tp = (label * pred).sum(axis=[0,1])
                gt = label.sum(axis=[0,1])
                pp = pred.sum(axis=[0,1])
                # print(f'pred: {pred},label: {label}')              
                # print(f'tp: {tp}, gt: {gt}, pp: {pp}')
                ppA += pp
                tps += tp
                gtA += gt

    recall = (tps / gtA)  #'%.4f'%
    precession = (tps / ppA)

    recall = (tps / gtA)
    precession = (tps / ppA)
    print(f'threshold: {threshold}')
    print(f'recall: {recall}, precession: {precession}')


if __name__ == '__main__':
    args = get_args()
    main(args)

