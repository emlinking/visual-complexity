# -*- coding: utf-8 -*-
"""
correlate_scores.py
Created on Sun Jul 10 19:06:40 2022

Script to correlate complexity scores with SAVOIAS groundtruth.
@author: linel
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.stats
import sys

# get groundtruths
def get_groundtruth(filename : str, category : str) -> np.ndarray:
    """
    Return ndarray of (image filename, complexity score) tuples sorted by filenames
    (0.jpg - [# of images in category].jpg).

    Parameters
    ----------
    filename : str
        Name of a .npy file from which to read groundtruth visual complexity 
        scores.
        Dictionary read from .npy comes in format 
        {image_index : [image_filename_as_png, image score]}
        Note that latest GitHub release of dataset stores files as JPEG, not
        PNG, so it is necessary to substitute the correct file extension in the
        image filename.
    
    category : str
        obj, scn, or int (objects, scenes, or interior design)

    Returns
    -------
    list
        ndarray of [image filename, complexity score] rows sorted by filenames.
        Image filename is reformatted as [image category]_[image #].

    """
    scores = np.load(filename, allow_pickle=True).item()
    return np.array([['{}_'.format(category) + str(item[0]), int(item[1][1])]
                    for item in sorted(scores.items(), key=lambda item: item[0])])

def build_groundtruth_df(file : str, category : str) -> pd.DataFrame:
    '''
    Build Pandas DataFrame with groundtruth SAVOIAS complexity scores.

    Parameters
    ----------
    file : str
        Name of a .npy file from which to read groundtruth visual complexity 
        scores.
        Dictionary read from .npy comes in format 
        {image_index : [image_filename_as_png, image score]}
        Note that latest GitHub release of dataset stores files as JPEG, not
        PNG, so it is necessary to substitute the correct file extension in the
        image filename.
    
    category : str
        obj, scn, or int (objects, scenes, or interior design)

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame of groundtruth visual complexity scores read from
        file, indexed by filenames in format [image category]_[image #].

    '''
    
    gt = get_groundtruth(file, category)
    return pd.DataFrame(data={'gt': gt[:, 1].astype(int)}, index=gt[:, 0])

def build_regions_df(data : str, category : str) -> pd.DataFrame:
    '''
    Build a dataframe of distinct regions data indexed by filenames.

    Parameters
    ----------
    data : str
        path to directory containing complexity data.
    category : str
        obj, scn, or int (objects, scenes, or interior design)

    Returns
    -------
    pd.DataFrame
        dataframe of distinct regions data indexed by filenames in format 
        [image category]_[image #].

    '''
    files = os.listdir(data)
    regions_list = []
    for i in range(len(files)):
        with open(os.path.join(data, files[i]), 'r') as f:
            try:
                lines = f.readlines()
                regions_filtered = lines[1]
                gray = (len(lines) == 3)
                
                if not gray:
                    regions_list.append(('{0}_{1}'.format(category, 
                                                         files[i][:-len('.jpg.txt')]), 
                                         str(regions_filtered)))
            except Exception as e:
                print('{0} on file {1}'.format(e, files[i]))
    
    regions_np = np.array(regions_list)
    return pd.DataFrame(data={'regions': regions_np[:, 1].astype(int)}, 
                        index=regions_np[:, 0])

def build_category_df(category : str, data : str, gt_file : str) -> pd.DataFrame:
    '''
    Build Pandas DataFrame with groundtruth complexity scores and distinct
    regions for images from given SAVOIAS category.

    Parameters
    ----------
    category : str
        obj, scn, or int (objects, scenes, or interior design)
    data : str
        path to directory containing complexity data.
    gt_file : str
        Name of a .npy file from which to read groundtruth visual complexity 
        scores. Provided by SAVOIAS authors in their GitHub repo.

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame with groundtruth complexity scores and distinct
        regions for images from given SAVOIAS category, indexed by filenames in 
        format [image category]_[image #].

    '''
    gt_df = build_groundtruth_df(file=gt_file, category=category)
    regions_df = build_regions_df(data=data, category=category)
    
    return regions_df.join(gt_df)
    
def plot_scores(df : pd.DataFrame, title : str, out_file : str) -> None:
    '''
    Plot # of distinct regions v. groundtruth complexity scores using data from
    df and save to out_file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns for groundtruth visual complexity and distinct
        number of regions, indexed by image file.
    title : str
        title for scatterplot.
    out_file : str
        file to save plot

    Returns
    -------
    None

    '''
    plt.scatter(df['regions'], df['gt'])
    plt.title(title)
    plt.xlabel('Distinct mean-shift-segmented regions')
    plt.ylabel('SAVOIAS groundtruth visual complexity')
    plt.savefig(out_file)
    plt.close()
    
def compute_correlation(df : pd.DataFrame):
    '''
    Compute Pearson's correlation between SAVOIAS groundtruth complexity scores
    and mean-shift-segmented distinct regions from data in df.

    Parameters
    ----------
    df : pd.DataFrame
        has columns 'regions' and 'gt'

    Returns
    -------
    s : str
        string containing r and p

    '''
    r, p = scipy.stats.pearsonr(df['regions'], df['gt'])
    s = 'r=' + str(r) + '\n'
    s += 'p=' + str(p)
    
    return s

def main(args : list) -> None:
    '''
    Generate scatterplots of groundtruth visual complexity scores v. distinct
    number of regions, and calculate Pearson's correlation for SAVOIAS objects,
    scenes, and interiors categories.

    Parameters
    ----------
    args : list
        args[0] : script name
        
        args[1] : directory with objects data
        
        args[2] : directory with scenes data
        
        args[3] : directory with interior design data
        
        args[4] : output file for objects plot
        
        args[5] : output file for scenes plot
        
        args[6] : output file for interior design plot
        
        args[7] : output file for combined categories plot
            
        args[8] : output file for correlations
 
    Returns
    -------
    None

    '''
    obj_df = build_category_df(category='obj', data=args[1], 
                               gt_file='/scratch/el55/Savoias-Dataset/Ground truth/npy/global_ranking_objects.npy')
    scn_df = build_category_df(category='scn', data=args[2], 
                               gt_file='/scratch/el55/Savoias-Dataset/Ground truth/npy/global_ranking_scenes.npy')
    int_df = build_category_df(category='int', data=args[3],
                               gt_file='/scratch/el55/Savoias-Dataset/Ground truth/npy/global_ranking_interior_design.npy')
    combined_df = pd.concat([obj_df, scn_df, int_df])
    
    plot_scores(df=obj_df, title='Ground truth complexity v. distinct regions\n for SAVOIAS Objects images', 
                out_file=args[4])
    plot_scores(df=scn_df, title='Ground truth complexity v. distinct regions\nfor SAVOIAS Scenes images', 
                out_file=args[5])
    plot_scores(df=int_df, title='Ground truth complexity v. distinct regions\nfor SAVOIAS Interior design images', 
                out_file=args[6])
    plot_scores(df=combined_df, title='Ground truth complexity v. distinct regions\nfor SAVOIAS Objects, Scenes, and Interior design images', 
                out_file=args[7])
    
    with open(args[8], 'a') as f:
        print('Correlation for objects:', file=f)
        print(compute_correlation(df=obj_df), file=f)
        print('Correlation for scenes:', file=f)
        print(compute_correlation(df=scn_df), file=f)
        print('Correlation for interiors:', file=f)
        print(compute_correlation(df=int_df), file=f)
        print('Correlation for all categories combined:', file=f)
        print(compute_correlation(df=combined_df), file=f)
if __name__ == '__main__':
    main(sys.argv)
    