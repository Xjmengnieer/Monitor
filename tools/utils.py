import os
import yaml
import argparse

from addict import Dict

def config_load(path):
    with open(path, mode='r') as f:
        return Dict(yaml.load(f, Loader=yaml.FullLoader))

def mkdirs(path): 
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str, default='./configs/CTran.yaml')
    parser.add_argument('--data_path', type = str, default='data/monitor/')
    parser.add_argument('--output', type = str, default='data/monitor_inference/')
    
    return parser.parse_args()

import torch
import logging

from time import time
from tqdm import tqdm
from termcolor import colored

from typing import Type, Any, Callable, Union, List, Optional
from torch.utils.tensorboard import SummaryWriter

class Trainer():
    def __init__(self, model, config,
                 train_dataloader,
                 val_dataloader,
                 optimizer,
                 scheduler,
                 criterion,
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
        
    def init_model(self):
        self.loggerInfo('Initializing model')
        
        if self.config.train_param.load_from:
            # 从预训练模型中加载权重
            old_model = torch.load(self.config.train_param.load_from, map_location='cpu').eval().cuda()
            old_state_dicts = old_model.state_dict()

            now_state_dicts = self.model.state_dict()

            # 将预训练模型中与新模型权重形状相同的权重复制到新模型中
            for k, v in now_state_dicts.items():
                if k in old_state_dicts and v.shape == old_state_dicts[k].shape:
                    v.copy_(old_state_dicts[k])

            # 将初始化后的权重加载到新模型中
            self.model.load_state_dict(now_state_dicts)
    
    def init_writer(self):
        self.loggerInfo('Initializing writer')
        self.writer = SummaryWriter(self.config.train_param.writer_dir)
    
    def init_logger(self):
        logger = logging.getLogger(self.config.train_param.logger_name)
        logger.setLevel(logging.DEBUG)

        ch_hander = logging.StreamHandler()
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

        mkdirs(self.config.train_param.save_path)

        for epoch in range(self.epoch):
            self.val()
            self.model._train(self.train_dataloader, 
                             self.config, 
                             optimizer = self.optimizer,
                             writer = self.writer,
                             scheduler = self.scheduler, 
                             )
            self.loggerInfo(f'finish {epoch}th epoch')
            if (epoch+1) % self.config.train_param.save_interval == 0:
                save_path = os.path.join(self.config.train_param.save_path, f'%010d'%(epoch + 1)+'.pth')
                torch.save(self.model, save_path)
            
    def val(self):
        self.loggerInfo('start validation')
        self.model._eval(self.val_dataloader, self.config)
        # par = tqdm(self.val_dataloader)
        # tps = 0
        # gtA = 0
        # ppA = 0
        # with torch.no_grad():
        #     for batch in par:
        #         par.set_description(f'validation: ')
        #         inputs, label = batch[0].cuda(),batch[1].cuda()
        #         output = self.model(inputs)
        #         pred = torch.sigmoid(output)
        #         pred = (pred > 0.8).long()
        #         tp = (label * pred).sum(axis=[0,1])
        #         gt = label.sum(axis=[0,1])
        #         pp = pred.sum(axis=[0,1])              
        #         ppA += pp
        #         tps += tp
        #         gtA += gt

        # recall = (tps / gtA)  #'%.4f'%
        # precession = (tps / ppA)
        
        # self.loggerInfo(f'recall : {recall}')
        # self.loggerInfo(f'precession : {precession}')
        # self.writer.add_scalar(f'validation recall', recall, self.step)
        # self.writer.add_scalar(f'validation precession', precession, self.step)

        # self.model.train()


