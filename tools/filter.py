import os
import torch

import numpy as np

from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from monitor_model import bulid_classifier
from utils import *

def main(args):
    config = config_load(args.config)
    data_root_path = config.inference.data_path
    print(f'datasets root path is: {data_root_path}')
    sub_dirs = os.listdir(data_root_path)

    out_path = config.inference.output

    if config.inference.load_from:
        print(f'load from {config.inference.load_from}')
        model = torch.load(config.inference.load_from, map_location='cpu').eval().cuda()
    else:
        model = bulid_classifier(config).eval().cuda()

    trans = transforms.Compose([
        transforms.CenterCrop(225),
        transforms.ToTensor(),
        transforms.Normalize(mean = config.datasets.mean, std = config.datasets.std)
    ]
    )
    preds = []
    print(f'test dirs are {sub_dirs}')
    
    for dir_name in sub_dirs:
        dir_path = os.path.join(data_root_path, dir_name)
        imgs = os.listdir(dir_path)
        par = tqdm(imgs)
        for img in par:
            par.set_description(f'{dir_name} dir has been processed: ')
            img_path = os.path.join(dir_path, img)

            image = Image.open(img_path).convert('RGB')
            image = trans(image).unsqueeze(0).cuda()

            with torch.no_grad():
                output = model(image)
                
            pred = torch.argmax(output,dim=-1)
            if pred not in preds:
                preds.append(pred)
            out_ = os.path.join(out_path, str(pred.cpu().numpy()[0]))
            mkdirs(out_)
            out_ = os.path.join(out_, img)
            os.system(f'mv {img_path} {out_}')
    
    print(preds)
    
    
    


if __name__ == '__main__':
    args = get_args()
    main(args)

