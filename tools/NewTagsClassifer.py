import os
import torch
import argparse
import logging


import torch.nn as nn

from time import time
from termcolor import colored
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import MultiStepLR
from typing import Type, Any, Callable, Union, List, Optional
from torch.utils.tensorboard import SummaryWriter

from monitor_model import bulid_classifier
from datasets import bulid_dataset
from datasets import mosaic
from utils import *

def main(args):
    config = config_load(args.config)
    
    model = bulid_classifier(config).cuda()
    train_trans = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomAffine(degrees=30, translate=(0.2, 0.2)),
        mosaic(size = (512,512)),
        transforms.Resize(756),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize(mean = config.datasets.mean, std = config.datasets.std)
    ])
    val_trans = transforms.Compose([
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize(mean = config.datasets.mean, std = config.datasets.std)
    ])

    train_dataset = bulid_dataset(config, train = True)
    val_dataset = bulid_dataset(config, train = False)
    
    train_dataset.add_transforms(train_trans)
    val_dataset.add_transforms(val_trans)

    train_dataloader = DataLoader(train_dataset, batch_size=config.datasets.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.datasets.batch_size)

    optimizer = SGD(model.parameters(), lr = 0.01, momentum = 0, weight_decay = 0)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 40, 60, 80, 100], gamma=0.1)

    critertion = nn.BCEWithLogitsLoss()

    trainer = Trainer(model, config,
            train_dataloader,
            val_dataloader,
            optimizer,
            scheduler,
            critertion,
            )
    trainer.init_model()
    trainer.train()


if __name__ == '__main__':
    args = get_args()
    main(args)