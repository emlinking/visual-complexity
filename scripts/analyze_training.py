# -*- coding: utf-8 -*-
"""
analyze_training.py

Created on Thu Jun 23 16:39:49 2022

Usage: pythn analyze_training.py training-logs-file plot-title plot-file
@author: linel
"""
import sys
import train_progress

def main(args : list) -> None:
    """
    Plot training progress and get max val/train accuracy.

    Parameters
    ----------
    args : list
        command line args. Expect args[0] = script name, args[1] = file with
        training logs, args[2] = title of plot, args[3] = file to save plots to.
        args[4] = mode ('classification' or 'regression')

    Returns
    -------
    None

    """
    logs = train_progress.load_logs(args[1])
    if args[4] == 'classification':
        print('Maximum training accuracy:', train_progress.max_train_acc(logs))
        print('Maximum val accuracy:', train_progress.max_val_acc(logs))
    print('Minimum train loss:', train_progress.min_train_loss(logs))
    print('Minimum val loss:', train_progress.min_val_loss(logs))
    train_progress.plot_training(args[4], logs, args[2], args[3])
    
if __name__ == '__main__':
    main(sys.argv)