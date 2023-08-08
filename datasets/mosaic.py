import numpy as np
import random
from PIL import Image

class mosaic():
    def __init__(self, size, center_or_random = 'center') -> None:
        self.size = size
        self.center_or_random = center_or_random
    
    def __call__(self, imgs, labels):
        h, w = self.size[0], self.size[1]
        mosaic = Image.new(imgs[0].mode, self.size)

        center_idy, center_idx = (h // 2, w // 2)

        x00 = random.randint(0, center_idx // 4)
        y00 = random.randint(0, center_idy // 4)

        mosaic.paste(imgs[0], (x00, y00))
        h0, w0 = imgs[0].size

        x01, y01 = x00 + random.randint(w0 * 3 // 4, w0 * 5 // 4),\
                   y00 + random.randint(0, y00 * 5 // 4)
        
        mosaic.paste(imgs[1], (x01, y01))

        x10, y10 = random.randint(0, x00 * 5 // 4), \
                   y00 + random.randint(h0 * 3 // 4, h0 * 5 // 4)
        mosaic.paste(imgs[2], (x10, y10))

        x11, y11 = x00 + random.randint(w0 * 3 // 4, w0 * 5 // 4),\
                   y00 + random.randint(h0 * 3 // 4, h0 * 5 // 4)
        
        mosaic.paste(imgs[3], (x11, y11))
        
        return mosaic