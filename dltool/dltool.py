import numpy as np
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import torch as t
import cv2
from math import log10

from dltool.util import bgr2ycbcr, imresize

def pil2cv(image):
    im = np.asarray(image)
    if im.ndim == 2:
        return im
    elif im.ndim == 3:
        return cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

def cv2pil(im):
    if im.ndim == 3:
        return Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
    elif im.ndim == 2:
        return Image.fromarray(im)

def _ToYCbCr(image, only_y=True):
    im = pil2cv(image.convert('RGB'))
    y = bgr2ycbcr(im, only_y)
    return cv2pil(y)

def ToYCbCr(only_y=True):
    return lambda x: _ToYCbCr(x, only_y)
    
def Resize(scale):
    return lambda im: imresize(im, scale)

def cv2tensor(im):
    image = cv2pil(im)
    return ToTensor()(image).unsqueeze(0).cpu()

def tensor2cv(tensor):
    image = tensor2pil(tensor)
    return pil2cv(image)
    
def tensor2pil(tensor):
    tensor = tensor.cpu()
    if tensor.ndim == 4:
        tensor = tensor[0]
    image = ToPILImage()(tensor)
    return image

def dir2tensor(dir, transforms = lambda x: x, device = 'cuda'):
    '''
    Input:
        dir of a image
    Output.shape = (1, channel, wei, hei)
    '''
    image = Image.open(dir)
    return ToTensor()(transforms(image)).unsqueeze(0).to(device)

def psnr(t1, t2):
    mse = t.nn.MSELoss()(t1, t2)
    return 10*log10(1. / mse.item())

def normalization(tensor):
    with t.no_grad():
        tensor = tensor.clone().detach()
        
        tensor = t.max(tensor, t.zeros(tensor.shape) )
        tensor = t.min(tensor, t.ones(tensor.shape)  )
    return tensor