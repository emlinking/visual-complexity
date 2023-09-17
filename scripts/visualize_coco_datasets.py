# -*- coding: utf-8 -*-
"""
visualize_coco_datasets.py

Script to plot complex/noncomplex images per COCO category.

Usage:

    python visualize_coco_datasets.py [input file] [output file] [plot title]
        [split: 'train' or 'val']
    
Created on Thu Jun 30 10:08:50 2022

@author: linel
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import visualize_model_predictions as vmp

def get_dataset_stats(file : str) -> pd.DataFrame:
    '''
    Read csv file and return Pandas dataframe.

    Parameters
    ----------
    file : str
        csv file to read as pandas dataframe

    Returns
    -------
    Dataframe read from file.

    '''
    df = pd.read_csv(file, index_col=0)
    df = df.reindex(vmp.COCO_CATEGORIES)
    
    return df

def plot_datasets(df : pd.DataFrame, file : str, title: str,
                  complex_idx : str, noncomplex_idx : str) -> None:
    '''
    Plot complex and noncomplex samples per COCO object category as grouped bar 
    chart. Save as PNG file.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas dataframe with # of complex/noncomplex samples per COCO category
    file : str
        Name of file to save plot.
    title : str
        Title for plot
    complex_idx : str
        Index of column with data for complex samples per category
    noncomplex_idx : str
        Index of column with data for noncomplex samples per category

    Returns
    -------
    None

    '''
    # data
    xlabels = df.index.to_numpy()
    complex_images = df[complex_idx].to_numpy(dtype=int)
    noncomplex_images = df[noncomplex_idx].to_numpy(dtype=int)
    
    x = np.arange(len(xlabels)) # bar locations on x-axis
    width = 0.35
        
    y = np.arange(0, max(np.max(complex_images), np.max(noncomplex_images)))
    
    fig, ax = plt.subplots()
    fig.set_figheight(5)
    fig.set_figwidth(12)
    rects1 = ax.bar(x + width/2, noncomplex_images, width, label='Number of noncomplex image captions')
    rects2 = ax.bar(x - width/2, complex_images, width, label='Number of complex image captions')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Number of image captions')
    ax.set_title(title)
    ax.set_xticks(x, xlabels, rotation=90)
    ax.set_yticks(y, rotation=45)
    ax.legend()
    
    plt.yscale('log')
    plt.grid(axis='y')
    ax.set_axisbelow(True)
    
    fig.tight_layout()
    plt.savefig(file)
    plt.show()
    
def main(args : list) -> None:
    """
    Plot COCO dataset statistics.

    Parameters
    ----------
    args : list
        command line args. 
        args[0] = name of script
        args[1] = name of input csv file with dataset stats
        args[2] = name of output png file to save plotted data
        args[3] = title of plot
        args[4], args[5] = complex_idx, noncomplex_idx of dataframe

    Returns
    -------
    None

    """
    df = get_dataset_stats(args[1])
    plot_datasets(df=df, file=args[2], title=args[3], complex_idx=args[4], 
                  noncomplex_idx=args[5])
    
if __name__ == '__main__':
    main(sys.argv)