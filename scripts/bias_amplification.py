# -*- coding: utf-8 -*-
"""
bias_amplification.py

Created on Fri Jul 15 15:53:35 2022

@author: linel
"""
import csv
import pickle
import sys
import visualize_coco_datasets as vcd
import visualize_model_predictions as vmp

def csv_from_category_counts(category_counts : dict, out_file : str) -> None:
    '''
    Convert category_counts dictionary generated by 
    visualize_model_predictions.count_prediction_categories to csv that can be
    graphed by visualize_coco_datasets.

    Parameters
    ----------
    category_counts : dict
        Output of visualize_model_predictions.count_prediction_categories
    out_file : str
        Path to .csv file to save formatted data

    Returns
    -------
    None

    '''
    with open(out_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        
        writer.writerow(['category', 'complex_captions', 'noncomplex_captions'])
        for cat in category_counts:
            writer.writerow([cat, category_counts[cat]['complex'],
                             category_counts[cat]['noncomplex']])
            
def main(args : list) -> None:
    '''
    Plot category counts of complex/noncomplex samples from predictions as 
    well as versus original val set counts.

    Parameters
    ----------
    args : list
        args[0] : script name
        
        args[1] : split (train/val/test)
        
        args[2] : predictions file
        
        args[3] : path to save csv file with counts by category

    Returns
    -------
    None

    '''
    predictions = pickle.load(open(args[2], 'rb'))
    counts_dict = vmp.count_prediction_categories(split=args[1], 
                                                  predictions=predictions)
    csv_from_category_counts(category_counts=counts_dict, out_file=args[3])
        
if __name__ == '__main__':
    main(sys.argv)