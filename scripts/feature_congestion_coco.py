# -*- coding: utf-8 -*-
"""
feature_congestion_coco.py

A script to calculate feature congestion for MS-COCO images.

Created on Thursday Jun 9 11:33 2022

@author: linel
"""
import numpy as np
import pickle
import time
from visual_clutter import Vlc

def batch_fc(imgs : np.ndarray) -> list:
    """
    Given a list of images, return a list of tuples 
    (img_name, feature congestion clutter map) via Vlc.getClutter_FC
    
    Parameters:
    imgs : np.ndarray - The array of (file, img) tuples where img are
    RGB images (as np arrays) to compute feature congestion.
    
    Returns:
    A list of (img_name, feature congestion, clutter map) tuples generated via
    Vlc.getClutter_FC.
    """
    start = time.perf_counter()
    print("Computing feature congestion . . . ", end="")
    
    result = []
    
    for i, img in enumerate(imgs):
        # make visual clutter object
        clt = Vlc(img[1])
        
        # get Feature Congestion clutter 
        fc, clutter_map = clt.getClutter_FC(p=1)
        
        result.append((img[0], fc, clutter_map))
        
        if i == (len(imgs)//2):
            print("Halfway there . . . ", end="")
        
    stop = time.perf_counter()
    print("Done. Total time to compute: {0:.4f} s".format(stop - start))
    
    return result

if __name__ == '__main__':
    # get image paths and image arrays
    cleaned_coco_imgs = pickle.load(open('cleaned_coco_imgs.p', 'rb'))
    
    # get clutter and cluttermaps per image
    coco_fc = batch_fc(cleaned_coco_imgs)
    
    # save with pickle
    pickle.dump(coco_fc, open('coco_fc.p', 'wb'))
    
    
    
    