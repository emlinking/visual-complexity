# -*- coding: utf-8 -*-
"""
get_regression_data.py

Created on Mon Jul 11 11:06:53 2022

Create dataset for training regression model.

@author: linel
"""

import dataset_builder as db
import json
import sys

def main(args : list):
    '''
    Read in complexity data and write as list of data dictionaries to json.

    Parameters
    ----------
    args : list
        args[0] : script name
        
        args[1] : path to data
        
        args[2] : whether to filter out grayscale images
        
        args[3] : output file

    Returns
    -------
    None.

    '''
    
    data, filter_gray, output = args[1], bool(int(args[2])), args[3]
    all_scores = db.get_complexity_scores(path=data, 
                                          filter_grayscale=filter_gray)
    all_scores_sorted = db.sort_scores(all_scores)
    
    json.dump(all_scores_sorted, open(output, 'w'))
    
if __name__ == '__main__':
    main(sys.argv)