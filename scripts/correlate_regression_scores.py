# -*- coding: utf-8 -*-
"""
correlate_regression_scores.py

Created on Fri Jul 15 11:07:32 2022

@author: linel
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.stats import pearsonr
import sys

def load_predictions(file : str) -> dict:
    '''
    Load predictions dict from file.
    
    Parameters
    ----------
    file : str
        path to pickled predictions dict
    
    Returns
    -------
    dict
        has keys 'scores', 'captions', 'label', 'url' for each image in the
        validation set.
    
    '''
    return pickle.load(open(file, 'rb'))

def correlate_predictions(in_file : str, out_file : str, title : str,
                          xlabel : str, ylabel : str, reduction : str) -> tuple:
    '''
    Correlate predicted complexity scores with true values.
    In addition to returning the r and p-value for the Pearson correlation,
    also saves scatterplot in out_file.

    Parameters
    ----------
    in_file : str
        path to predictions dict pickled file which has keys 'scores', 
        'captions', 'label', 'url' for each image in the validation set.

    out_file : str
        path to save scatterplot
        
    title : str
        title for scatterplot
        
    xlabel : str
        xlabel for scatterplot
        
    ylabel : str
        ylabel for scatterplot
        
    reduction : str
        'none' or 'average' - whether to average predicted values
        
    Returns
    -------
    tuple
        (Pearson r, p-value) from correlating predictions and true values.

    '''
    predicted_vals, true_vals = np.array([]), np.array([])
    predictions_dict = load_predictions(in_file)
    
    for prediction in predictions_dict.values():
        if reduction == 'none':
            predicted_vals = np.append(predicted_vals, prediction['scores'])
            true_vals = np.append(true_vals,
                                  [prediction['label']]*len(prediction['scores']))
        elif reduction == 'average':
            predicted_vals = np.append(predicted_vals, 
                                       sum(prediction['scores'])/len(prediction['scores']))
            true_vals = np.append(true_vals, prediction['label'])
        
    plt.scatter(true_vals, predicted_vals)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(out_file)
    
    r, p = pearsonr(predicted_vals, true_vals)
    r, p = np.format_float_scientific(r), np.format_float_scientific(p)
    
    return r, p

def main(args : list) -> None:
    '''
    Correlate regression model predictions with true values (Pearson r).

    Parameters
    ----------
    args : list
        
        args[0] : script name
        
        args[1] : path to predictions file (pickled)
        
        args[2] : path to save scatterplot
        
        args[3] : title for scatterplot
        
        args[4], args[5] : x and y labels for scatterplot
        
        args[6] : path to file to save r, p-value
        
        args[7] : reduction - 'average' or 'none' - whether to average scores
        per image

    Returns
    -------
    None

    '''
    with open(args[6], 'a') as f:
        print(args[1] + '\t' +
              str(correlate_predictions(in_file=args[1], out_file=args[2],
                                    title=args[3], xlabel=args[4], 
                                    ylabel=args[5], reduction=args[7])), file=f)
        
if __name__ == '__main__':
    main(sys.argv)
    