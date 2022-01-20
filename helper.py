import torch
import math
import torchvision
import time


def time_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def make_divisible(x, divisor):
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max()) 
    return math.ceil(x / divisor) * divisor

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None: 
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2 
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    coords[:, [0, 2]] -= pad[0]  #x pad
    coords[:, [1, 3]] -= pad[1]  #y pad
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, shape):
    if isinstance(boxes, torch.Tensor): 
        boxes[:, 0].clamp_(0, shape[1])  
        boxes[:, 1].clamp_(0, shape[0]) 
        boxes[:, 2].clamp_(0, shape[1])  
        boxes[:, 3].clamp_(0, shape[0]) 
    else:  
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0]) 

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,labels=(), max_det=1000):
    #list of detections

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  

    assert 0 <= conf_thres <= 1, 'Must be between 0-1.0'
    assert 0 <= iou_thres <= 1, 'Must be between 0-1.0'


    min_wh, max_wh = 2, 4096  
    max_nms = 30000  
    multi_label &= nc > 1  

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]  # confidence

        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Yolo to box CV
        box = xywh2xyxy(x[:, :4])

        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
        n = x.shape[0]  #num of det
        if not n:  
            continue
        elif n > max_nms:  
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes , scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  
            i = i[:max_det]

        output[xi] = x[i]

    return output

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y