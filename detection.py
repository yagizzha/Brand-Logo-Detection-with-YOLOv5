import cv2
import numpy as np


class Annotator:

    def __init__(self,im,line_width=None,font_size=None,font='Arial.ttf',pil=False,example='abc'):
        self.im = im
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003),2) 

    def box_label(self,box,label='',color=(128,128,128),txt_color=(255,255,255)):
        p1,p2 = (int(box[0]),int(box[1])),(int(box[2]),int(box[3]))
        cv2.rectangle(self.im,p1,p2,color,thickness=self.lw,lineType=cv2.LINE_AA)
        if label:
            tf = max(self.lw - 1,1)
            w,h = cv2.getTextSize(label,0,fontScale=self.lw / 3,thickness=tf)[0]  
            outside = p1[1] - h - 3 >= 0 
            p2 = p1[0] + w,p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(self.im,p1,p2,color,-1,cv2.LINE_AA) 
            cv2.putText(self.im,label,(p1[0],p1[1] - 2 if outside else p1[1] + h + 2),0,self.lw / 3,txt_color,
                        thickness=tf,lineType=cv2.LINE_AA)


    def result(self):
        return np.asarray(self.im)


class Colors:
    def __init__(self):
        hex = ('FF3838','FF9D97','FF701F','FFB21D','CFD231','48F90A','92CC17','3DDB86','1A9334','00D4BB',
               '2C99A8','00C2FF','344593','6473FF','0018EC','8438FF','520085','CB38FF','FF95C8','FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self,i,bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2],c[1],c[0]) 

    @staticmethod
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2],16) for i in (0,2,4))
