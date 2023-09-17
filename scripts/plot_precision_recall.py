# -*- coding: utf-8 -*-
"""
regression_as_classification.py

Created on Wed Jul 13 11:26:16 2022

Plot precision-recall for classification model v. regression model using
regression scores for classification.

@author: linel
"""
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import average_precision_score, precision_recall_curve, \
                                PrecisionRecallDisplay
import sys
import visualize_model_predictions as vmp

def main(args : list):
    '''
    Plot ROC curves.

    Parameters
    ----------
    args : list
        command line args
        
        args[0] : name of script
        
        args[1] : classification model predictions
        
        args[2] : regression model predictions
        
        args[3] : precision-recall plot filepath to save
        
        args[4] : text filepath to save AP

    Returns
    -------
    None.

    '''
        
    classifier_predictions = pickle.load(open(args[1], 'rb'))
    regressor_predictions = pickle.load(open(args[2], 'rb'))
    
    # we can use labels from classification data for measuring regression data
    # accuracy
    clabels, cpreds = vmp.get_precision_recall_data(classifier_predictions)
    _, rpreds = vmp.get_precision_recall_data(regressor_predictions)
    
    cprec, crecall, cthresh = precision_recall_curve(clabels, cpreds,
                                                     pos_label=1)
    cAP = average_precision_score(clabels, cpreds, pos_label=1)
    rprec, rrecall, rthresh = precision_recall_curve(clabels, rpreds,
                                                     pos_label=1)
    rAP = average_precision_score(clabels, rpreds, pos_label=1)

    plt.figure(0).clf()
    plt.plot(crecall, cprec, label='classification model, AP='+str(cAP))
    plt.plot(rrecall, rprec, label='regression model, AP='+str(rAP))
    title = ('Precision-recall: classification v. regression model (positive label: 1)')
    plt.title(title)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    
    plt.savefig(args[3])
    
    with open(args[-1], 'a') as file:
        print(args[3], ',', cAP, ',', rAP, file=file)
    
if __name__ == '__main__':
    main(sys.argv)