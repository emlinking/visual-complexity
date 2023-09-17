# -*- coding: utf-8 -*-
"""
A script for building HTML pages to display images by their complexity scores.

Created on 6/23/2022.

@author: linel
"""
import numpy as np

def write_header(filename : str, title : str):
    """
    Write HTML header to file, including CSS styles.

    Parameters
    ----------
    filename : str
        Name of file to write header
    title : str
        Title of file

    Returns
    -------
    None.

    """
    f = open(filename, 'w')
    
    f.write('<!DOCTYPE html>\n')
    f.write(('<html>\n<head>\n<title>{0}</title>\n<style>'
             '\ntable {{ \n\t width: 100%; \n }}\n'
             'img {{'
             '\n\twidth: 100%;'
             '\n}}'
             '\ntd {{'
             '\n\ttext-align: center;\n\tvertical-align:bottom;'
             '\n}}\n</style>\n</head>'
            '\n\t\t<body><h1>{0}</h1>\n').format(title))
    
    f.close()
    
def write_footer(filename : str):
    """
    Write HTML footer to file.

    Parameters
    ----------
    filename : str
        Name of file to write footer

    Returns
    -------
    None.

    """
    f = open(filename, 'a')
    f.write('''\t</body>\n</html>''')
    f.close()
    
def get_image_ranking(filename : str) -> list:
    """
    Return list of (image filename, complexity score) tuples sorted by visual 
    complexity scores, high to low.

    Parameters
    ----------
    filename : str
        Name of a .npy file from which to read groundtruth visual complexity 
        scores.
        Dictionary read from .npy comes in format 
        {image_index : [image_filename_as_png, image score]}
        Note that latest GitHub release of dataset stores files as JPEG, not
        PNG, so it is necessary to substitute the correct file extension.

    Returns
    -------
    list
        List of (image filename, complexity score) tuples sorted from highest 
        visual complexity to lowest.

    """
    scores = np.load(filename, allow_pickle=True).item()
    return [(str(item[0]) + '.jpg', item[1][1]) 
             for item in sorted(scores.items(), key=lambda item: item[1][1], 
                                reverse=True)]

def write_images(filename : str, imgs : list, caption : str=''):
    """
    Write table of images to HTML file, 5 images per row, high to low 
    complexity left to right and top to bottom. Score is printed beneath each
    photo.

    Parameters
    ----------
    filename : str
        Name of html file to write table to.
    imgs : str
        List of (image paths, scores) to display.
    caption : str
        Table caption.
        
    Returns
    -------
    None.

    """
    f = open(filename, 'a')
    f.write('\t\t<table border="0">\n\t\t\t<caption>{}</caption>\n\t\t\t\t<tr>'.format(caption))
            
    for i, img in enumerate(imgs):
        f.write('<td><img src="{0}" alt="{0}" loading="lazy"/>{1}</td>'.format(img[0], 
                                                                               img[1]))
        
        # new row
        if (i + 1) % 5 == 0:
            f.write('</tr>\n\t\t\t\t<tr>')
    
    f.write('\n\t\t</table>\n')
    f.close()
    