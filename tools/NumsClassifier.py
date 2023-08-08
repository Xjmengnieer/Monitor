import os
import torch
import argparse # 命令行解析器
import logging # 日志


import torch.nn as nn

from time import time
from termcolor import colored # 给输出添加颜色
from tqdm import tqdm
from torchvision import transforms 
from torch.utils.data import DataLoader 
from torch.optim import SGD, Adam, AdamW 
from torch.optim.lr_scheduler import MultiStepLR # 学习率衰减
from typing import Type, Any, Callable, Union, List, Optional
from torch.utils.tensorboard import SummaryWriter

from monitor_model import bulid_classifier
from datasets.Imagenet import ImageNet
from datasets.mosaic import mosaic
from utils import *
import numpy as np
#
class Trainer():
    def __init__(self, model, config,
                 train_dataloader,
                 val_dataloader,
                 optimizer,
                 scheduler, # 学习率衰减
                 criterion, # 损失函数
                 **kwarg: Any):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.epoch = config.train_param.epoch
        self.save_interval = config.train_param.save_interval
        self.log_interval_step = config.train_param.log_interval_step

        self.init_logger()
        self.init_writer()
    
    def init_writer(self):
        self.loggerInfo('Initializing writer')
        self.writer = SummaryWriter(self.config.train_param.writer_dir)  #将信息记录到TensorBoard中
    
    def init_logger(self):
        logger = logging.getLogger(self.config.train_param.logger_name)
        logger.setLevel(logging.DEBUG) # 设置日志级别

        ch_hander = logging.StreamHandler() # 输出到控制台
        dir_path = os.path.dirname(self.config.train_param.logger_path) 
        mkdirs(dir_path)
        fh_hander = logging.FileHandler(self.config.train_param.logger_path.format(name = time()))

        ch_hander.setLevel(logging.DEBUG)
        fh_hander.setLevel(logging.DEBUG)

        fmt = logging.Formatter("%(asctime)s - %(name)s  - line:%(lineno)d - %(levelname)s - %(message)s")
        color_fmt = colored('%(asctime)s %(name)s', 'green') + colored(' %(filename)s %(lineno)d', 'yellow') + ':%(levelname)s %(message)s'  #%filename)s
        ch_hander.setFormatter((logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S')))

        fh_hander.setFormatter(fmt)

        logger.addHandler(ch_hander)
        logger.addHandler(fh_hander)

        self.logger = logger
    
    def loggerInfo(self, info:str):
        self.logger.info(info)
    
    def train(self):
        self.loggerInfo('start training')
        self.step = 0
        mkdirs(self.config.train_param.save_path)

        # criterion1 = nn.L1Loss()
        # criterion2 = nn.MSELoss()

        for epoch in range(self.epoch):
            self.val()                              # 每个epoch都进行一次验证
            par = tqdm(self.train_dataloader)
            for batch in par:
                par.set_description(f'training epoch {epoch}: ')
                inputs, label = batch[0].cuda(), batch[1].cuda().to(torch.float32)
                output = self.model(inputs)

                loss0 = self.criterion(output, label)
                # output_logit = torch.sigmoid(output)
                # loss1 = criterion1(output_logit, label)
                # loss2 = criterion2(output_logit, label)

                loss = loss0 #  + loss1 + loss2
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                

                self.step += 1

                if (self.step+1) % self.config.train_param.log_interval_step == 0:
                    # self.loggerInfo(loss)
                    self.writer.add_scalar(f'train_loss', loss, self.step)
                    # self.writer.add_scalar(f'L1_loss', loss0, self.step)
                    # self.writer.add_scalar(f'MSE_loss', loss0, self.step)

            self.scheduler.step()

            if (epoch+1) % self.config.train_param.save_interval == 0:
                save_path = os.path.join(self.config.train_param.save_path, f'%010d'%(epoch + 1)+'.pth')
                torch.save(self.model, save_path)
            
    def val(self):
        self.loggerInfo('start validation')
        self.model.eval()
        par = tqdm(self.val_dataloader)
        tps = 0
        gtA = 0
        ppA = 0
        # 单独看看火的正确率
        tp_label_31 = 0  # 记录label=31的真正例数
        p_label_31 = 0   # 记录label=31的总正例数

        with torch.no_grad():
            for batch in par:
                par.set_description(f'validation: ')
                inputs, label = batch[0].cuda(), batch[1].cuda()
                output = self.model(inputs)

                pred = torch.argmax(output, dim=-1)
                label = torch.argmax(label, dim=-1)

                tps += (pred == label).sum().item()
                gtA += label.size(0)
                ppA += pred.size(0)

                # 统计label=31的真正例和总正例数
                tp_label_31 += ((pred == label) & (label == 31)).sum().item()
                p_label_31 += (label == 31).sum().item()

        precession = tps / ppA
        recall_label_31 = tp_label_31 / p_label_31  # label=31的正确率

        self.loggerInfo(f'precession : {precession}')
        self.loggerInfo(f'recall label 31 : {recall_label_31}')
        self.writer.add_scalar(f'validation precession', precession, self.step)
        self.writer.add_scalar(f'recall label 31', recall_label_31, self.step)

        self.model.train()






def main(args):
    config = config_load(args.config)
    # model = bulid_classifier(config).cuda()
    # 从预训练模型加载
    model = torch.load(config.inference.load_from, map_location='cpu').cuda() 

    # 增加模型输出层的类别数
    num_classes_old = model.fc.out_features
    num_classes_new = num_classes_old + 1  # 添加一个新的类别

    # 创建一个新的全连接层用于新类别
    new_fc_layer = nn.Linear(in_features=model.fc.in_features, out_features=num_classes_new)
    # nn.init.xavier_uniform_(new_fc_layer.weight)  # 初始化新层的权重
    # 将之前的权重复制到新的全连接层
    with torch.no_grad():
        new_fc_layer.weight[:num_classes_old, :] = model.fc.weight
        new_fc_layer.bias[:num_classes_old] = model.fc.bias

    # 初始化新增类别的权重
    nn.init.xavier_uniform_(new_fc_layer.weight[num_classes_old:, :])
    # 替换模型中的旧全连接层为新的全连接层
    model.fc = new_fc_layer
    model = model.cuda()

    
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
        transforms.CenterCrop(225),
        transforms.ToTensor(),
        transforms.Normalize(mean = config.datasets.mean, std = config.datasets.std)
    ])

    train_dataset = ImageNet(config.datasets.train.root, config.model.num_classes, transforms=train_trans)
    val_dataset = ImageNet(config.datasets.val.root, config.model.num_classes, transforms=val_trans)
    
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
    
    trainer.train()


if __name__ == '__main__':
    args = get_args()
    main(args)