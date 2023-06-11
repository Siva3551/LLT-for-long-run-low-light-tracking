

import numpy as np
import cv2

from PIL import Image

def singleScaleRetinex(img,variance):
    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), variance))
    return retinex


# Do the single scale retinex. Here the full frame, sigma value for retinex and the bounding box coordinates are given as the inputs
def SSR(img, variance, box):
   
    # Select bit higher area than the actual input area (bounding box)
    crop = img[int((box[1]-box[3]/4)):int((box[1]+1.25*box[3])),int((box[0]-0.25*box[2])):int((box[0]+1.25*box[2]))]

    # SSR is done to the selected area of the image
    imge = np.float64(crop) + 1.0
    img_retinex = singleScaleRetinex(imge, variance)

    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(np.int32(img_retinex[:, :, i]*100), return_counts=True)
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break            
        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.01:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.01:
                high_val = u / 100.0
                break            
        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)
        
        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                               * 255
    img_retinex = np.uint8(img_retinex)  

    # The SSR done area is combined with the original full frame
    original_img = np.array(img) 
    original_img[int((box[1]-box[3]/4)):int((box[1]+1.25*box[3])),int((box[0]-0.25*box[2])):int((box[0]+1.25*box[2]))] = img_retinex
 
    return original_img






def template(img, variance, box):

    crop = img[int(box[1]):int((box[1]+box[3])),int(box[0]):int((box[0]+box[2]))]

    imge = np.float64(crop) + 1.0
    img_retinex = singleScaleRetinex(imge, variance)

    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(np.int32(img_retinex[:, :, i]*100), return_counts=True)
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break            
        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.01:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.01:
                high_val = u / 100.0
                break            
        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)
        
        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                               * 255
    img_retinex = np.uint8(img_retinex)  

    return img_retinex