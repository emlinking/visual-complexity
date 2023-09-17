# -*- coding: utf-8 -*-
"""
get_coco_dataset_stats.py

Created on Fri Jul 15 16:44:46 2022

@author: linel
"""
import csv
import os
import sys
import visualize_coco_datasets

def csv_from_dataset_logs1(out_file : str, log_directory : str) -> None:
    with open(out_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        
        writer.writerow(['category', 'complex_captions', 'noncomplex_captions',
                         'complex_train_captions', 'noncomplex_train_captions',
                         'complex_val_captions', 'noncomplex_val_captions',
                         'complex_test_captions', 'noncomplex_test_captions'])
        
        files = os.listdir(log_directory)
        for file in files:
            if file.endswith('_classification_set.txt'):
                out_line = [file.replace('_classification_set.txt', '')]
                with open(os.path.join(log_directory, file), 'r') as f:
                    lines = f.readlines()
                    
                    for line in lines:
                        if line.startswith('Complex image captions'):
                            out_line.append(line.replace('Complex image captions: ', 
                                                        '').strip())
                        if line.startswith('Noncomplex image captions'):
                            out_line.append(line.replace('Noncomplex image captions: ', 
                                                        '').strip())
                                
                                
                writer.writerow(out_line)

def csv_from_dataset_logs2(out_file : str, log_directory : str) -> None:
    with open(out_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        
        writer.writerow(['category', 'train_captions', 'val_captions',
                         'test_captions'])
        
        files = os.listdir(log_directory)
        for file in files:
            if file.endswith('tanh_norm.txt'):
                out_line = [file.replace('regression_', '').replace('_tanh_norm.txt',
                                                                    '')]
                with open(os.path.join(log_directory, file), 'r') as f:
                    lines = f.readlines()
                    
                    for line in lines:
                        if line.startswith('Size of dataset'):
                            out_line.append(line.replace('Size of dataset: ', 
                                                        '').strip())
                                
                writer.writerow(out_line)
                
def main(args : list) -> None:
    '''
    Write dataset stats to csv in COCO category order.

    Parameters
    ----------
    args : list
        args[0] : script name
        args[1] : log_directory
        args[2] : out_file
        args[3] : 'classification' or 'regression'

    Returns
    -------
    None

    '''
    if args[3] == 'classification':
        csv_from_dataset_logs1(args[2], args[1])
    elif args[3] == 'regression':
        csv_from_dataset_logs2(args[2], log_directory=args[1])
    
if __name__ == '__main__':
    main(sys.argv)