# -*- coding: utf-8 -*-
"""
train_progress.py

Functions to analyze training progress from logs.

Created on Thu Jun 23 16:11:17 2022

@author: linel
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle

def load_logs(logs : str) -> dict:
    """
    Parameters
    ----------
    logs : str
        Path to .p file storing
        dictionary with keys ['train_loss', 'train_accuracy', 'val_loss', 
                              'val_accuracy'] mapping to lists tracking these
        values over the course of training the model.

    Returns
    -------
    dict

    """
    return pickle.load(open(logs, 'rb'))

def min_val_acc(logs : dict) -> float:
    """
    Get minimum accuracy on valset.

    Parameters
    ----------
    logs : str
        dictionary with keys ['train_loss', 'train_accuracy', 'val_loss', 
                              'val_accuracy'] mapping to lists tracking these
        values over the course of training the model.

    Returns
    -------
    float
        Minimum val_accuracy recorded.

    """
    return min(logs['val_accuracy_by_epoch'])

def max_val_acc(logs : dict) -> float:
    """
    Get maximum accuracy on valset.

    Parameters
    ----------
    logs : str
        dictionary with keys ['train_loss', 'train_accuracy', 'val_loss', 
                              'val_accuracy'] mapping to lists tracking these
        values over the course of training the model.

    Returns
    -------
    float
        Maximum val_accuracy recorded.

    """
    return max(logs['val_accuracy_by_epoch'])

def min_train_acc(logs : dict) -> float:
    """
    Get minimum accuracy on trainset.

    Parameters
    ----------
    logs : str
        dictionary with keys ['train_loss', 'train_accuracy', 'val_loss', 
                              'val_accuracy'] mapping to lists tracking these
        values over the course of training the model.

    Returns
    -------
    float
        Minimum train_accuracy recorded.

    """
    return min(logs['train_accuracy_by_epoch'])

def max_train_acc(logs : dict) -> float:
    """
    Get maximum accuracy on trainset.

    Parameters
    ----------
    logs : str
        dictionary with keys ['train_loss', 'train_accuracy', 'val_loss', 
                              'val_accuracy'] mapping to lists tracking these
        values over the course of training the model.

    Returns
    -------
    float
        Maximum train_accuracy recorded.

    """
    return max(logs['train_accuracy_by_epoch'])

def min_train_loss(logs : dict) -> float:
    """
    Get minimum loss on trainset.

    Parameters
    ----------
    logs : str
        dictionary with keys ['train_loss', 'train_accuracy', 'val_loss', 
                              'val_accuracy'] mapping to lists tracking these
        values over the course of training the model.

    Returns
    -------
    float
        Minimum train loss recorded.

    """
    return min(logs['train_loss_by_epoch'])

def max_train_loss(logs : dict) -> float:
    """
    Get maximum loss on trainset.

    Parameters
    ----------
    logs : str
        dictionary with keys ['train_loss', 'train_accuracy', 'val_loss', 
                              'val_accuracy'] mapping to lists tracking these
        values over the course of training the model.

    Returns
    -------
    float
        Maximum train loss recorded.

    """
    return max(logs['train_loss_by_epoch'])

def min_val_loss(logs : dict) -> float:
    """
    Get minimum loss on valset.

    Parameters
    ----------
    logs : str
        dictionary with keys ['train_loss', 'train_accuracy', 'val_loss', 
                              'val_accuracy'] mapping to lists tracking these
        values over the course of training the model.

    Returns
    -------
    float
        Minimum valset loss recorded.

    """
    return min(logs['val_loss_by_epoch'])

def max_val_loss(logs : dict) -> float:
    """
    Get maximum loss on trainset.

    Parameters
    ----------
    logs : str
        dictionary with keys ['train_loss', 'train_accuracy', 'val_loss', 
                              'val_accuracy'] mapping to lists tracking these
        values over the course of training the model.

    Returns
    -------
    float
        Maximum val loss recorded.

    """
    return max(logs['val_loss_by_epoch'])

def plot_training(mode : str, logs : dict, title : str, filename : str) -> None:
    """
    Plot training progress and save to png.
    Uses accuracies/losses from first batch in first epoch as estimate for 
    initial accuracy/loss of model.

    Parameters
    ----------
    mode : str
        'classification' or 'regression'
    logs : dict
        dictionary with keys ['train_loss', 'train_accuracy', 'val_loss', 
                              'val_accuracy'] mapping to lists tracking these
        values over the course of training the model.
    title : str
        title for figure
    filename : str
        name of path to save training plot

    Returns
    -------
    None
    """
    
    if mode == 'classification': 
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 9))
        fig.suptitle(title)
        ax1.plot(logs['val_accuracy_by_epoch'])
        ax1.set_title('Accuracy on val set')
        ax1.set_xlabel('Epochs')
        ax1.set_xticks(np.arange(len(logs['val_accuracy_by_epoch'])), 
                       np.arange(len(logs['val_accuracy_by_epoch'])) + 1)
        ax1.set_ylim(0, 1)
        
        ax2.plot(logs['train_accuracy_by_epoch'])
        ax2.set_title('Accuracy on train set')
        ax2.set_xlabel('Epochs')
        ax2.set_xticks(np.arange(len(logs['train_accuracy_by_epoch'])), 
                       np.arange(len(logs['train_accuracy_by_epoch'])) + 1)
        ax2.set_ylim(0, 1)
        
        ax3.plot(logs['val_loss_by_epoch'])
        ax3.set_title('Loss on val set')
        ax3.set_xlabel('Epochs')
        ax3.set_xticks(np.arange(len(logs['val_loss_by_epoch'])), 
                       np.arange(len(logs['val_loss_by_epoch'])) + 1)
        ax3.set_ylim(0, max(logs['val_loss_by_epoch'] + logs['train_loss_by_epoch']))
        
        ax4.plot(logs['train_loss_by_epoch'])
        ax4.set_title('Loss on train set')
        ax4.set_xlabel('Epochs')
        ax4.set_xticks(np.arange(len(logs['train_loss_by_epoch'])), 
                       np.arange(len(logs['train_loss_by_epoch'])) + 1)
        ax4.set_ylim(0, max(logs['val_loss_by_epoch'] + logs['train_loss_by_epoch']))
    elif mode == 'regression':
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4))
        fig.suptitle(title)
        
        ax1.plot(logs['val_loss_by_epoch'])
        ax1.set_title('Loss on val set')
        ax1.set_xlabel('Epochs')
        ax1.set_xticks(np.arange(len(logs['val_loss_by_epoch'])), 
                       np.arange(len(logs['val_loss_by_epoch'])) + 1)
        ax1.set_ylim(0, max(logs['val_loss_by_epoch'] + logs['train_loss_by_epoch']))
        
        ax2.plot(logs['train_loss_by_epoch'])
        ax2.set_title('Loss on train set')
        ax2.set_xlabel('Epochs')
        ax2.set_xticks(np.arange(len(logs['train_loss_by_epoch'])), 
                       np.arange(len(logs['train_loss_by_epoch'])) + 1)
        ax2.set_ylim(0, max(logs['val_loss_by_epoch'] + logs['train_loss_by_epoch']))
        
    plt.savefig(filename)