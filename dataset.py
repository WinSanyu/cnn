# coding:utf8
import os
from PIL import Image
from torch.utils import data
from torchvision import transforms as T



class SetuDataset(data.Dataset):

    def __init__(self, image_dir, input_transform = None, target_transform = None):
        
        super(SetuDataset, self).__init__()
        self.image_filenames = [ os.path.join(image_dir, x) for x in os.listdir(image_dir) ]
        
        self.input_transform = input_transform
        self.target_transform = target_transform 
        
        
    def __getitem__(self, index):
        
        target_image = Image.open(self.image_filenames[index])      
        input_image = target_image.copy()
       
        target_image = self.target_transform(target_image)
        input_image = self.input_transform(input_image)
        
        return input_image, target_image
        
        
    def __len__(self):
        
        return len(self.image_filenames)