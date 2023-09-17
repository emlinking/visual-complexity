# -*- coding: utf-8 -*-
"""
feature_congestion.py

A script to calculate feature congestion for SAVOIAS images.

Created on Wed Jun  8 14:55:44 2022

@author: linel
"""
# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np
import os
import pickle
# import pymeanshift as pms
# import scipy.stats
import time
# import urllib
from visual_clutter import Vlc

def batch_fc(img_paths : list) -> list:
    """
    Given a list of images, return a list of tuples 
    (img_name, feature congestion clutter map) via Vlc.getClutter_FC
    
    Parameters:
    img_paths : list - The list of filepaths for images to compute feature
    congestion for.
    
    Returns:
    A list of (img_name, feature congestion clutter map) tuples generated via
    Vlc.getClutter_FC.
    """
    start = time.perf_counter()
    print("Computing feature congestion . . . ", end="")
    
    result = []
    
    for i, img_path in enumerate(img_paths):
        # make visual clutter object
        clt = Vlc(img_path)
        
        # get Feature Congestion clutter 
        fc, clutter_map = clt.getClutter_FC(p=1)
        
        result.append((img_path, fc, clutter_map))
        
        if i == (len(img_paths)//2):
            print("Halfway there . . . ", end="")
        
    stop = time.perf_counter()
    print("Done. Total time to compute: {0:.4f} s".format(stop - start))
    
    return result

if __name__ == '__main__':
    # get image paths
    savoias_objs = [os.path.join('/scratch/el55/Savoias-Dataset/Images/Objects',
                            path)
                    for path in os.listdir('/scratch/el55/Savoias-Dataset/Images/Objects')
                    if path[0] != '.']
    savoias_scenes = [os.path.join('/scratch/el55/Savoias-Dataset/Images/Scenes',
                              path)
                      for path in os.listdir('/scratch/el55/Savoias-Dataset/Images/Scenes')
                      if path[0] != '.']
    savoias_interiors = [os.path.join('/scratch/el55/Savoias-Dataset/Images/Interior Design', 
                                 path)
                         for path in os.listdir('/scratch/el55/Savoias-Dataset/Images/Interior Design')
                         if path[0] != '.']
    
    # get clutter and cluttermaps per image
    savoias_objs_fc = batch_fc(savoias_objs)
    savoias_scenes_fc = batch_fc(savoias_scenes)
    savoias_interiors_fc = batch_fc(savoias_interiors)
    
    # save with pickle
    pickle.dump(savoias_objs_fc, open('savoias_objs_fc.p', 'wb'))
    pickle.dump(savoias_scenes_fc, open('savoias_scenes_fc.p', 'wb'))
    pickle.dump(savoias_interiors_fc, open('savoias_interiors_fc.p', 'wb'))
    
    
    
    