import os
import random
from PIL import Image

class mosaic():
    def __init__(self, size, center_or_random='center', bg_path='data/BG', distance_threshold=50):
        self.size = size
        self.center_or_random = center_or_random
        self.bg_path = bg_path
        self.distance_threshold = distance_threshold
    
    def __call__(self, imgs, labels):

        # num = random.randint(4, 12)
        # if len(imgs) < num:
        #     raise ValueError(f"The 'imgs' list must contain at least {num} images.")
        num = len(imgs)
        h, w = self.size[0], self.size[1]
        mosaic = Image.new('RGB', self.size)

        # 选择随机背景图片
        bg_files = os.listdir(self.bg_path)
        bg_image_path = os.path.join(self.bg_path, random.choice(bg_files))
        bg_image = Image.open(bg_image_path)
        bg_image = bg_image.resize(self.size)
        mosaic.paste(bg_image, (0, 0))

        center_idy, center_idx = (h // 2, w // 2)
        scale = 2+num
        coordinates = []
        for i in range(num):
            img = imgs[i].resize((w // scale, h // scale))
            if self.center_or_random == 'center':
                x = center_idx + random.randint(-center_idx // 2, center_idx // 2)
                y = center_idy + random.randint(-center_idy // 2, center_idy // 2)
            else:
                x = random.randint(0, w - img.size[0])
                y = random.randint(0, h - img.size[1])
            
            # 检查新的图片左上角坐标与已有图片的距离
            while any(abs(x - coord[0]) < self.distance_threshold and abs(y - coord[1]) < self.distance_threshold for coord in coordinates):
                if self.center_or_random == 'center':
                    x = center_idx + random.randint(-center_idx // 2, center_idx // 2)
                    y = center_idy + random.randint(-center_idy // 2, center_idy // 2)
                else:
                    x = random.randint(0, w - img.size[0])
                    y = random.randint(0, h - img.size[1])
            
            mosaic.paste(img, (x, y))
            coordinates.append((x, y))
        return mosaic


# dataroot='data/monitor/init/train/banana'
# imgs = []
# labels = []
# for img in os.listdir(dataroot):
#     imgs.append(Image.open(os.path.join(dataroot, img)))
#     labels.append(45) #

# mosaic_generator = mosaic(size=(512, 512), center_or_random='center')
# mosaic_image = mosaic_generator(imgs, labels)
# mosaic_image.save('mosaic_image.jpg')

