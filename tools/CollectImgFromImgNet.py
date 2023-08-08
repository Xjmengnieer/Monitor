import os
import torch

from monitor_model.network import resnet50

from datasets import validIds

class rmUnusedImg():
    def __init__(self, root) -> None:
        self.root = root
        self.dirs = os.listdir(root)
        self.validDirs = self.get_validDirs()
        self.rmunsuseddirs()
    
    def get_validDirs(self):
        validdirs = []

        for key, value in validIds.items():
            validdirs.append(value['dir'])

        return validdirs

    def rmunsuseddirs(self):
        for dir in self.dirs:
            if dir not in self.validDirs:
                dir_path = os.path.join(self.root, dir)
                os.system(f'rm -r {dir_path}')
        
        print('done')

a = rmUnusedImg('/home/data/monitor/val/')