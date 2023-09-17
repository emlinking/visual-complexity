# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 21:49:35 2022

@author: linel
"""

import dataset_builder as db
import html_builder
import torchvision.transforms as transforms

# verify grayscale filtering was successful by building html pages
# load PyTorch Datasets
img_transf = transforms.Compose([transforms.Resize(256),
                                  transforms.CenterCrop((224, 224)),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                       std=[0.229, 0.224, 0.225])])

rgb_train2017 = db.CocoComplexityDataset(split='train', 
                                         data='/scratch/el55/coco-out/coco_train2017_from_script.json', 
                                         caption_ann='/scratch/data/COCO/annotations/captions_train2017.json', 
                                         images='/scratch/data/COCO/train2017',
                                         transform=img_transf,
                                         filter_grayscale=True)
rgb_val2017 = db.CocoComplexityDataset(split='val', 
                                         data='/scratch/el55/coco-out/coco_val2017_from_script.json', 
                                         caption_ann='/scratch/data/COCO/annotations/captions_val2017.json', 
                                         images='/scratch/data/COCO/val2017',
                                         transform=img_transf,
                                         filter_grayscale=True)

img_paths = [image['coco_url'] for image in rgb_val2017.metadata]
print(len(img_paths))
scores = [image['regions_filtered'] for image in rgb_val2017.metadata]
images = [(img_path, score) for (img_path, score) in zip(img_paths, scores)]
images = sorted(images, key=lambda image: image[1], reverse=True)

html_builder.write_header('color-coco-val2017.html', 
                          'COCO val2017 Color Images')
html_builder.write_images('color-coco-val2017.html', images)
html_builder.write_footer('color-coco-val2017.html')