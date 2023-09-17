# -*- coding: utf-8 -*-
"""
get_coco_category_list.py

Created on Fri Jun 24 10:11:46 2022

@author: linel
"""

import json

coco_2017_categories = json.load(open('C:/Users/linel/Desktop/DREU/coco_annotations_2017/instances_train2017.json', 
                                      'r'))
categories_list = coco_2017_categories['categories']

with open('coco_2017_categories.txt', 'w') as f:
    for category in categories_list:
        print(category, file=f)