# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 21:06:55 2022

@author: Amean
"""

from pathlib import Path
from PIL import Image
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

imagePath = Path("train-jpg")

saveDir = Path("trainJpeg")



imageFilePaths = list(imagePath.glob("*"))

print(len(imageFilePaths))


for filePath in imageFilePaths[1:7]:
    
    image = cv.imread(str(filePath),0)
    # image = Image.open(filePath)
    fileName = filePath.stem
    fileName = fileName + ".jpeg"
    savePath = saveDir / fileName
    plt.imshow(image)
    # image.save(savePath, quality="web_high")
    imf = np.float32(image)/255.0  # float conversion/scale
    dct = cv.dct(imf)              # the dct
    imgcv1 = np.uint8(dct*255.0)    # convert back to int
    # plt.imshow(imgcv1)
# jpegFilePaths = list(saveDir.glob("*"))

# for sample in jpegFilePaths[:2]:
#     print(sample)
#     im = Image.open(sample)
#     print(im.quantization)
