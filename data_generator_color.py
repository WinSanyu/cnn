import glob
import os
import cv2
import numpy as np
from PIL import Image

'''
$ python create_patch.py 
将 train 中的图片分成 patch
并保存在 patch 中
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

# TODO: 写成参数
patch_size, stride = 40, 10
aug_times = 1
scales = [1, 0.9, 0.8, 0.7]
batch_size = 128


def data_aug(img, mode=0):

    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

def gen_patches(file_name, index, save_dir = 'patch'):

    # read image
    image = Image.open(file_name)
    # 注意PIL的size和opencv的shape是反的
    w, h = image.size[0], image.size[1] 
    img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR) # PIL2cv
    
    cnt = 0
    for s in scales:
        h_scaled, w_scaled = int(h*s), int(w*s)
        img_scaled = cv2.resize(img, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
        # extract patches
        for j in range(0, h_scaled-patch_size+1, stride):
            for i in range(0, w_scaled-patch_size+1, stride):
                try:
                    x = img_scaled[i:i+patch_size, j:j+patch_size]
                except:
                    print(file_name + "超出范围\n")
                    print('x: ' + str(i) + ' ,y: ' + str(j) + '\n')
                    print('边界: ' + str(h_scaled) + ', ' + str(w_scaled))
                    continue
                
                # data aug
                for k in range(0, aug_times):
                    x_aug = data_aug(x, mode=np.random.randint(0,8))
                    try:
                        cv2.imwrite(os.path.join(save_dir, str(index) + '_' + str(cnt)+'.jpg'), x_aug)
                    except:
                        print(file_name + "保存错误\n")
                    cnt += 1
                    

def create_patch(data_dir = 'train', save_dir = 'patch'):
    
    # get name list of all .jpg files
    filenames = glob.glob(os.path.join(data_dir, '*.jpg'))  

    for i, filename in enumerate(filenames):
        try:
            gen_patches(filename, i, save_dir)
            # print(filename + ' finish\n')
        except:
            print(filename + ' fail\n')
            continue
    
    
if __name__ == '__main__':   
    # data = create_patch(data_dir=os.path.join('BSDS300', 'images', 'train'))
    data = create_patch()
