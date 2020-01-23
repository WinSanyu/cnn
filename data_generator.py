'''
block the images in dataset_dir
save to save_dir

Example:
python create_patch.py
 ├── create_patch.py
 ├── train
 │   ├── 0.jpg
 │   ├── 1.jpg
 │   ├── 2.jpg
 │   └── ...
 └── patch
     ├── 0_0.jpg
     ├── 0_1.jpg
     ├── 0_2.jpg
     ├── ...
     ├── 1_0.jpg
     ├── 1_1.jpg
     ├── 1_2.jpg
     └── ...
'''

import os
from torchvision.transforms import Compose, RandomChoice, RandomHorizontalFlip, RandomRotation, RandomVerticalFlip
from PIL import Image
import argparse
from itertools import product


parser = argparse.ArgumentParser()
parser.description='create patch'
parser.add_argument("--dataset_dir", help="this is dataset_dir", type=str, default="train")
parser.add_argument("--save_dir", help="this is save_dir",  type=str, default="patch")
parser.add_argument("--PATCH_SIZE", help="PATCH SIZE",  type=int, default=40)
parser.add_argument("--STRIDE", help="STRIDE",  type=int, default=10)
parser.add_argument("--AUG_TIMES", help="AUG_TIMES",  type=int, default=1)
# parser.add_argument("--SCALES", help="SCALES",  type=str, default="[1, 0.9, 0.8, 0.7]")
args = parser.parse_args()


def transform():
    return RandomChoice([
            Compose([]),
            RandomVerticalFlip(1),
            RandomRotation((90, 90)),
            Compose([RandomVerticalFlip(1), RandomRotation((90,90)), ]),
            RandomRotation((180, 180)),
            RandomHorizontalFlip(1),
            RandomRotation((270, 270)),
            Compose([RandomHorizontalFlip(1), RandomRotation((90,90)), ]),
    ])


def gen_patches(file_name, save_dir, 
                PATCH_SIZE, STRIDE, AUG_TIMES, SCALES):
    
    data_aug = transform()
    
    # read image
    image = Image.open(file_name)
    w, h = image.width, image.height
    
    _, tmpname = os.path.split(file_name)
    savename = tmpname.split('.')[0]
    savename_cnt = 0
    for s in SCALES:
        w_scaled, h_scaled = int(w*s), int(h*s)
        img_scaled = image.resize((w_scaled, h_scaled), Image.BICUBIC)
        # extract patches
        for h_index, w_index in product(range(0, h_scaled-PATCH_SIZE+1, STRIDE), range(0, w_scaled-PATCH_SIZE+1, STRIDE)):
            x = img_scaled.crop((w_index, h_index, w_index + PATCH_SIZE, h_index + PATCH_SIZE))
                
            # data aug
            for k in range(0, AUG_TIMES):
                x_aug = data_aug(x)
                x_aug.save(os.path.join(save_dir, str(savename) + '_' + str(savename_cnt)+'.jpg'))
                savename_cnt += 1     
                    
              
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def image_name_list(dataset_dir):
    return [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir) if is_image_file(x)]


def create_patch(dataset_dir = 'train', save_dir = 'patch', 
                 PATCH_SIZE = 40, STRIDE = 10, AUG_TIMES = 1, SCALES = [1, 0.9, 0.8, 0.7]):
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    filenames = image_name_list(dataset_dir)

    for filename in filenames:
        try:
            gen_patches(filename, save_dir, PATCH_SIZE, STRIDE, AUG_TIMES, SCALES)
            print(filename + ' finish\n')
        except:
            print(filename + ' fail\n')
            continue
    
    
if __name__ == '__main__':   
    data = create_patch(dataset_dir = args.dataset_dir,  # os.path.join('BSDS300', 'images', 'train')
                        PATCH_SIZE = args.PATCH_SIZE,
                        STRIDE = args.STRIDE,
                        AUG_TIMES = args.AUG_TIMES,
                        )
