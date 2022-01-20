import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, weights='curr.pt', device=None, dnn=False, data=None):

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        model = attempt_load(w, map_location=device)
        self.stride = int(model.stride.max())  
        names = model.names 
        print(names)
        self.model = model  
        self.__dict__.update(locals())  

    def forward(self, im, augment=False, visualize=False, val=False):
        y = self.model(im, augment=augment, visualize=visualize)
        return y if val else y[0]

    def warmup(self, imgsz=(1, 3, 640, 640), half=False):
        if isinstance(self.device, torch.device) and self.device.type != 'cpu':  
            im = torch.zeros(*imgsz).to(self.device).type(torch.half if half else torch.float)  
            self.forward(im)  

class Ensemble(nn.ModuleList):
    def __init__(self):
        super().__init__()

def attempt_load(weights, map_location=None):

    model = Ensemble()
    for w in [weights]:
        print("YO WE HERE",w,map_location)
        ckpt = torch.load(w, map_location=map_location)  # load
        model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval()) 
    if len(model) == 1:
        return model[-1]  # return model
    else:
        print(f'Ensemble created with {weights}\n')
        for k in ['names']:
            setattr(model, k, getattr(model[-1], k))
        model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  
        return model  

    