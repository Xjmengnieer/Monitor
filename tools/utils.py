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
    parser.add_argument('--config', type = str, default='./configs/default.yaml')
    parser.add_argument('--data_path', type = str, default='data/monitor/')
    parser.add_argument('--output', type = str, default='data/monitor_inference/')
    
    return parser.parse_args()

