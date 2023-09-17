# -*- coding: utf-8 -*-
"""
normalize_regression_data.py

Created on Mon Jul 11 13:44:05 2022

@author: linel
"""

import json
import numpy as np
import sys

def load_data(file : str) -> list:
    '''
    Load data from json file and return as list.

    Parameters
    ----------
    file : str
        json file containing list of complexity scores with each entry in
        format [coco image id, regions, distinct regions, grayscale]

    Returns
    -------
    list
        list read from file

    '''
    return json.load(open(file, 'r'))

def normalize_data(mode : str, in_file : str, out_file : str, min_r : int=None, 
                   max_r : int=None, reduction : int=None) -> None:
    '''
    Normalize complexity data to be in range [0, 1].

    Parameters
    ----------
    mode : str, possible values: ['min-max', 'tanh']
        Type of normalization to use
    in_file : str
        path to json file containing raw complexity data
    out_file : str
        path to json file to save normalized complexity data
        
    The following parameters are used only if mode == 'min-max':
        
    min_r : int, optional
        value to take as minimum complexity score; any values less than min_r
        will be rounded up to min_r
    max_r : int, optional
        value to take as maximum complexity score; any values greater than max_r
        will be rounded down to max_r
    reduction : int, optional
        divide number of regions / reduction before taking tanh

    Returns
    -------
    None

    '''
    raw_data = load_data(in_file)
    normed_data = np.array([r for _, _, r, _ in raw_data])
    
    if mode == 'min-max':
        normed_data[normed_data < min_r] = min_r
        normed_data[normed_data > max_r] = max_r
        normed_data = (normed_data - min_r)/(max_r - min_r)
    elif mode == 'tanh':
        normed_data = np.tanh(normed_data/reduction)
    else:
        raise ValueError('Valid normalization modes are "min-max" and "tanh"')
    
    # copy over normalized data to new list
    new_data = raw_data
    for i in range(len(new_data)):
        new_data[i][2] = normed_data[i]
        
    json.dump(new_data, open(out_file, 'w'))
    
def main(args : list) -> None:
    '''
    Normalize distinct # of regions as preprocessing step for training
    regression model.

    Parameters
    ----------
    args : list
        args[0] : script name
        
        args[1] : file with raw complexity data
        
        args[2] : file to save normalized data
        
        args[3] : normalization mode - 'min-max' or 'tanh'
        
        The following args are only used if the normalization mode is 'min-max':
            
        args[4] : floor for data
        
        args[5] : ceiling for data
        
        The following args are only used if the normalization mode is 'tanh':
        
            args[4] : reduction - divide number of regions / reduction before 
            taking tanh

    Returns
    -------
    None

    '''
    if len(args) == 6:
        if args[3] == 'min-max':
            normalize_data(mode=args[3], in_file=args[1], out_file=args[2], 
                           min_r=int(args[4]), max_r=int(args[5]))
        elif args[3] == 'tanh':
            raise ValueError('Too many args for normalization mode "tanh"')
    elif len(args) == 5:
        if args[3] == 'tanh':
            normalize_data(mode=args[3], in_file=args[1], out_file=args[2],
                           reduction=int(args[4]))
        else:
            raise ValueError('Invalid args[3] "{}", must be "min-max" or "tanh"'.format(args[3]))
    
if __name__ == '__main__':
    main(sys.argv)