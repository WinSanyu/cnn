# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 23:37:48 2019

@author: WinSa
."""

import cv2 as cv
from PIL import Image
import os
import numpy as np

def read_directory( directory_name ):
    '''
    用于清洗质量不佳的图片
    输入0意味着质量不佳
    质量合格输入任意值即可
    '''
    pre = ''
    cnt = 0
    for filename in os.listdir( directory_name ):
        filepath = os.path.join(directory_name, filename)
        try:
            img = Image.open( filepath )
            im = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
        except:
            continue
        cv.imshow(filename,im)
        a = cv.waitKey(0)
        cv.destroyAllWindows()
        if a == 48:
            os.remove( filepath )
            continue
        elif a == 8:
            break
        newname = pre + str(cnt) + '.jpg'
        os.rename( filepath, os.path.join(directory_name, newname) )
        cnt += 1

#==================================================================================
root = "D:/临时/pic/"
read_directory( root )
