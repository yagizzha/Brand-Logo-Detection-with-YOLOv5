import argparse
import os
import sys
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from modeltest import Model
from input import LoadStreams
from helper import make_divisible, scale_coords, non_max_suppression, time_sync
from detection import Annotator, Colors


@torch.no_grad()
def run(weights='yolov5s.pt',  # model.pt path(s)
        source='0', # 0 for webcam
        imgsz=(720, 1280),  # inference size (height, width)
        conf_thres=0.2,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        line_thickness=3,  # bounding box thickness (pixels)
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):

    max_det=1000
    

    # Load model
    device="cuda:0"
    device = torch.device(device)
    model = Model(weights, device=device, dnn=dnn, data=None)
    stride, names = model.stride, model.names
    """
    file1 = open('classes.txt', 'r')
    Lines = file1.readlines()
    print(names)
    names=[]
    for line in Lines:
        names.append(line)
    names[26]="Google"
    print(names)"""
    imgsz = [max(make_divisible(x, 32), 0) for x in imgsz] # check image size

    # Half
    half &= device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    model.model.half() if half else model.model.float()
    capture = LoadStreams(source, img_size=imgsz, stride=stride, auto=True)

    cudnn.benchmark = True 

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    colors=Colors()
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in capture:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path[i], im0s[i].copy(), capture.count
            s += f'{i}: '

            p = Path(p)  # to Path
            #s += '%gx%g ' % im.shape[2:]  # print string
            #cv2.imwrite("test.jpg",im0)
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, " 

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = f'{names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))

            print(f'{s}Done. ({t3 - t2:.3f}s)')
            im0 = annotator.result()
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)  



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='curr.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default="0", help='index of webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[736,1280], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.2, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device,0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', default=True, help='class-agnostic NMS')
    parser.add_argument('--augment', default=True, help='augmented inference')
    parser.add_argument('--visualize', default=True, help='visualize features')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness')
    parser.add_argument('--half', default=True)
    parser.add_argument('--dnn', default=True)
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print(', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
