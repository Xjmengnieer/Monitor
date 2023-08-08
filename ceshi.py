import os
import cv2
from tqdm import tqdm

videos_path = '/home/data/monitor/fire_video/'
save_path = '/home/data/monitor/fire_video/frames'

if not os.path.exists(save_path):
    os.makedirs(save_path)

videos = os.listdir(videos_path)

i = 0
for videoName in tqdm(videos):
    videoPath = os.path.join(videos_path, videoName)
    videoCap = cv2.VideoCapture(videoPath)

    ret, frame = videoCap.read()
    
    while ret:
        imgPath = os.path.join(save_path, '04%d'%i + '.jpg')
        i += 1
        
        cv2.imwrite(imgPath, frame)
        ret, frame = videoCap.read()