import os

from tqdm import tqdm

def unzipImg(path):
    tars = os.listdir(path)

    for tar in tqdm(tars):
        tar_name = tar.split('.')[0]
        tarPath = os.path.join(path, tar)
        dirPath = os.path.join(path, tar_name)
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
        
        os.system(f'tar -xvf {tarPath} -C {dirPath}')
        os.system(f'rm -r {tarPath}')

root = '/home/data/monitor/train/'
unzipImg(root)