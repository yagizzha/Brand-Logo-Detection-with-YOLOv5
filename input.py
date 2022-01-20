import time
from threading import Thread
import cv2
import numpy as np
import re

class LoadStreams:
    def __init__(self, sources='streams.txt', img_size=640, stride=32, auto=True, height=720, width=1280):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride
        self.height = height
        self.width = width


        sources = [sources]
        n = len(sources)
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later

        self.auto = auto
        for i, s in enumerate(sources):  # index, source
            st = f'{i + 1}/{n}: {s}... '
            s = eval(s) if s.isnumeric() else s  # webcamnumber
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f'{st}Failed to open {s}'
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT,self.height)
            self.fps[i] = max(cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 30.0  # 30 FPS fallback
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback
            _, self.imgs[i] = cap.read()  # first img for checks
            self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=True)
            print(f"{st} Success ({self.frames[i]} frames {self.width}x{self.height} at {self.fps[i]:.2f} FPS)")
            self.threads[i].start()
        print() 

        s = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0].shape for x in self.imgs])
        self.rect = np.unique(s, axis=0).shape[0] == 1  

    def update(self, i, cap, stream):
        n, f, read = 0, self.frames[i], 1 
        while cap.isOpened() and n < f:
            n += 1
            cap.grab()
            if n % read == 0:
                success, im = cap.retrieve()
                if success:
                    self.imgs[i] = im
                else:
                    print('No input !')
                    self.imgs[i] = np.zeros_like(self.imgs[i])
                    cap.open(stream) 
            #time.sleep(1 / self.fps[i])  

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'): 
            cv2.destroyAllWindows()
            raise StopIteration

        img0 = self.imgs.copy()
        img = [letterbox(x, self.img_size, stride=self.stride, auto=self.rect and self.auto)[0] for x in img0]
        img = np.stack(img, 0)
        img = img[..., ::-1].transpose((0, 3, 1, 2))  
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None, ''

    def __len__(self):
        return len(self.sources)

def clean_str(s):
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup: 
        r = min(r, 1.0)

    ratio = r, r  
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  
    if auto: 
        dw, dh = np.mod(dw, stride), np.mod(dh, stride) 
    dw /= 2  
    dh /= 2

    if shape[::-1] != new_unpad: 
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color) 
    return im, ratio, (dw, dh)