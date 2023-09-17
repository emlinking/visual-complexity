# -*- coding: utf-8 -*-
"""
filter_coco_class.py

A script to filter COCO images by class.

Usage: python filter_coco_class.py [mode] [class1] [class2] . . . 
[train-data] [val-data] [test-data] [name-of-log-file.txt] [noun_mask]
[verb_mask] [adjective_mask] [adverb_mask]

Created on Wed Jun 22 15:27:39 2022

@author: linel
"""
import dataset_builder as db
import html_builder
import pickle
import sys
import time
import torchvision.transforms as transforms

def main(args : list):
    """
    Parameters
    ----------
    args : list
        list of command line arguments to script - see file header for 
        description

    Returns
    -------
    None.

    """
    start = time.perf_counter()
    
    img_transf = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                           std=[0.229, 0.224, 0.225])])
    
    mode, classes, train_data, val_data, test_data, log_file = args[1], \
                                                                args[2:-8], \
                                                                args[-8], \
                                                                args[-7], \
                                                                args[-6], \
                                                                args[-5]
    mask_nouns, mask_verbs, mask_adjs, mask_advs = bool(int(args[-4])), \
        bool(int(args[-3])), bool(int(args[-2])), bool(int(args[-1]))    
                                                            
    if len(classes) == 0: # no filtering for class
        category = None
        train_ann, val_ann, test_ann = None, None, None
    else: 
        # note that train/val sets are created from COCO train set, while
        # test set comes from COCO val set
        category = classes
        train_ann = '/scratch/data/COCO/annotations/instances_train2017.json'
        val_ann = '/scratch/data/COCO/annotations/instances_train2017.json'
        test_ann = '/scratch/data/COCO/annotations/instances_val2017.json'
        
    train2017 = db.CocoComplexityDataset(mode, split='train', 
                                            data=train_data,
                                            caption_ann='/scratch/data/COCO/annotations/captions_train2017.json',
                                            instance_ann=train_ann,
                                            images='/scratch/data/COCO/train2017',
                                            transform=img_transf,
                                            obj_class=category,
                                            noun_mask=mask_nouns,
                                            adj_mask=mask_adjs,
                                            verb_mask=mask_verbs,
                                            adv_mask=mask_advs)
    val2017 = db.CocoComplexityDataset(mode, split='val', 
                                            data=val_data,
                                            caption_ann='/scratch/data/COCO/annotations/captions_train2017.json',
                                            instance_ann=val_ann,
                                            images='/scratch/data/COCO/train2017',
                                            transform=img_transf,
                                            obj_class=category,
                                            noun_mask=mask_nouns,
                                            adj_mask=mask_adjs,
                                            verb_mask=mask_verbs,
                                            adv_mask=mask_advs)
    test2017 = db.CocoComplexityDataset(mode, split='test', 
                                            data=test_data,
                                            caption_ann='/scratch/data/COCO/annotations/captions_val2017.json',
                                            instance_ann=test_ann,
                                            images='/scratch/data/COCO/val2017',
                                            transform=img_transf,
                                            obj_class=category,
                                            noun_mask=mask_nouns,
                                            adj_mask=mask_adjs,
                                            verb_mask=mask_verbs,
                                            adv_mask=mask_advs)
        
    # record info about datasets
    with open(log_file, 'w') as f:
        print(train2017, file=f)
        print(val2017, file=f)
        print(test2017, file=f)
        
    filename = ''
    if category is not None:
        filename += '_'.join(category)
    else:
        filename += 'full_set'
        
    if mask_nouns:
        filename += '_maskn'
    if mask_verbs:
        filename += '_maskv'
    if mask_adjs:
        filename += '_maskadj'
    if mask_advs:
        filename += '_maskadv'
        
    if mode == 'classification':
            
        pickle.dump(train2017, open('{}_train2017_classification.p'.format(filename), 
                                    'wb'))
        pickle.dump(val2017, open('{}_val2017_classification.p'.format(filename), 
                                  'wb'))
        pickle.dump(test2017, open('{}_test2017_classification.p'.format(filename), 
                                   'wb'))
        
        # verify filtering was successful by building html pages
        img_paths = [image['coco_url'] for image in val2017.metadata]
        scores = [str(image['regions_filtered']) + "\n" + caption['caption'] 
                  for image, caption in zip(val2017.metadata, val2017.captions)]
        images = [(img_path, score) for (img_path, score) in zip(img_paths, scores)]
        images = sorted(images, key=lambda image: int(image[1].split()[0]), 
                        reverse=True)
        
        html_builder.write_header('coco-{}-val2017-new.html'.format(filename), 
                                  'COCO val2017 {} Images'.format(filename))
        html_builder.write_images('coco-{}-val2017-new.html'.format(filename), images)
        html_builder.write_footer('coco-{}-val2017-new.html'.format(filename))
    
    elif mode == 'regression':
        pickle.dump(train2017, open('{}_train2017_regression.p'.format(filename), 'wb'))
        pickle.dump(val2017, open('{}_val2017_regression.p'.format(filename), 'wb'))
        pickle.dump(test2017, open('{}_test2017_regression.p'.format(filename), 'wb'))
        
        # verify filtering was successful by building html pages
        img_paths = [image['coco_url'] for image in val2017.metadata]
        scores = [str(image['score']) + "\n" + caption['caption'] 
                  for image, caption in zip(val2017.metadata, val2017.captions)]
        images = [(img_path, score) for (img_path, score) in zip(img_paths, scores)]
        images = sorted(images, key=lambda image: float(image[1].split()[0]), 
                        reverse=True)
        
        html_builder.write_header('coco-{}-val2017-regression.html'.format(filename), 
                                  'COCO val2017 {} Images'.format(filename))
        html_builder.write_images('coco-{}-val2017-regression.html'.format(filename), images)
        html_builder.write_footer('coco-{}-val2017-regression.html'.format(filename))
    stop = time.perf_counter()
    
    print('Time to run script: {0:.4f} s'.format(stop - start))
    
if __name__ == '__main__':
    main(sys.argv)