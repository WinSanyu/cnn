from torchvision.transforms import Compose, ToTensor
import torch.utils.data as data
import os
from PIL import Image
import torch
from dataset import SetuDataset


def gaussian_additive_noise(im_torch, sigma):
    noise = torch.randn(im_torch.size()).mul_(sigma / 255.0)
    return im_torch + noise


def input_transform_fun(im, sigma):
    im_torch = Compose([
        ToTensor(),
        ])(im)
    return gaussian_additive_noise(im_torch, sigma)
    
    
class input_transform(object):

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, img):
        return input_transform_fun(img, self.sigma)
        
        
def target_transform(): 
    return Compose([
        ToTensor(),
    ])


def get_training_set(train_dir, sigma):
    return SetuDataset(train_dir, input_transform(sigma), target_transform())
