import numpy as np
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import torch as t
import cv2
from math import log10

import warnings

from dltool.util import bgr2ycbcr, imresize

def pil2cv(image) -> np.ndarray:
    """
    Args:
        image (PIL): gray or RGB image in PIL

    Returns:
        numpy.ndarray: BGR image in opencv2
    """
    im = np.asarray(image)
    if im.ndim == 2:
        return im
    elif im.ndim == 3:
        return cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    else:
        # TODO: warnning
        print('image.dim should be 2 or 3 but not ' + str(im.ndim))
        print('we try to convert it to RGB')
        im = np.asarray(image.convert('RGB'))
        return cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

def cv2pil(im: np.ndarray):
    """
    Args:
        im (ndarray): gray or BGR image in opencv2

    Returns:
        gray or RGB image in PIL
    """
    assert (im.ndim == 3 or im.ndim == 2)
    if im.ndim == 3:
        return Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
    elif im.ndim == 2:
        return Image.fromarray(im)

def _ToYCbCr(image, only_y=True) -> t.Tensor:
    '''used in ToYCbCr'''
    im = pil2cv(image.convert('RGB'))
    y = bgr2ycbcr(im, only_y)
    return cv2pil(y)

def ToYCbCr(only_y=True):
    """
    Same interface as torchvision
    ToYCbCr is the realization of MATLAB
    """
    return lambda x: _ToYCbCr(x, only_y)
    
def Resize(scale):
    """
    Same interface as torchvision
    imresize is the realization of MATLAB
    """
    return lambda im: imresize(im, scale)

def cv2tensor(im: np.ndarray) -> t.Tensor:
    """
    Args:
        im (ndarray): gray or BGR image in opencv2

    Returns:
        torch.Tensor: device == 'cpu'
    """
    image = cv2pil(im)
    return ToTensor()(image).unsqueeze(0).cpu()

def pil2tensor(im) -> t.Tensor:
    """
    Args:
        im (PIL): gray or RGB image in PIL

    Returns:
        t.Tensor: device == 'cpu'
    """
    # TODO: im.mode == gray or RGB?
    return ToTensor()(im)

def tensor2cv(tensor: t.Tensor) -> np.ndarray:
    """
    Args:
        tensor (torch.Tensor): tensor.ndim == 3 or
            (tensor.ndim == 4 and tensor.shape[0] == 1)

    Returns:
        np.ndarray: BGR or gray image in opencv2
    """
    image = tensor2pil(tensor)
    return pil2cv(image)
    
def tensor2pil(tensor: t.Tensor):
    """
    Args:
        tensor (torch.Tensor): tensor.ndim == 3 or
            (tensor.ndim == 4 and tensor.shape[0] == 1) or
            tensor.ndim == 2

    Returns:
        gray or BGR image in PIL
    """
    # TODO: check ndim
    tensor = tensor.cpu()
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            # TODO: warnning
            print('tensor.ndim == 4, we only get the first image')
        tensor = tensor[0]
    assert (tensor.ndim == 3 or tensor.ndim == 2)
    image = ToPILImage()(tensor)
    return image

def dir2tensor(dir: str, transforms = lambda x: x, device = 'cuda') -> t.Tensor:
    """
    Args:
        dir (str): dir of a image

    Returns:
        torch.Tensor: Output.shape = (1, channel, wei, hei)
    """
    image = Image.open(dir)
    return ToTensor()(transforms(image)).unsqueeze(0).to(device)

def psnr(t1, t2):
    mse = t.nn.MSELoss()(t1, t2)
    return 10*log10(1. / mse.item())

def normalization(tensor: t.Tensor) -> t.Tensor:
    """
    Keep the number between 0 and 1
    """
    with t.no_grad():
        tensor = tensor.clone().detach()
        device = tensor.device
        tensor = t.max(tensor, t.zeros(tensor.shape, device=device) )
        tensor = t.min(tensor, t.ones(tensor.shape, device=device)  )
    return tensor