# -*- coding: utf-8 -*-
"""
visualize_model_predictions.py

Build html pages of model's most confident predicted complex/noncomplex
images, along with their captions and confidence scores.

Created on Mon Jul  4 14:24:41 2022

@author: linel
"""
from classify import count_correct
import dataset_builder as db
import html_builder
import pickle
from pycocotools.coco import COCO
import numpy as np
import statistics as stats
import sys
import torch
import torch.nn
import torch.utils.data
from transformers import BertForSequenceClassification

COCO_CATEGORIES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
                   'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 
                   'stop sign', 'parking meter', 'bench', 
                   'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 
                   'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 
                   'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 
                   'sports ball', 'kite', 'baseball bat', 'baseball glove', 
                   'skateboard', 'surfboard', 'tennis racket', 
                   'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 
                   'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 
                   'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
                   'couch', 'potted plant', 'bed', 'dining table', 
                   'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
                   'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
                   'refrigerator', 'book', 'clock', 'vase', 'scissors', 
                   'teddy bear', 'hair drier', 'toothbrush']

def get_val_dataset(valset : str) -> db.CocoComplexityDataset:
    '''
    Read dataset from pickle file.

    Parameters
    ----------
    valset : str
        path to .p file with dataset

    Returns
    -------
    CocoComplexityDataset

    '''
    return pickle.load(open(valset, 'rb'))

def get_val_dataloader(valset : str) -> torch.utils.data.DataLoader:
    '''
    Get DataLoader for valset to use in evalution.

    Parameters
    ----------
    valset : str
        Path to .p file with CocoComplexityDataset.

    Returns
    -------
    torch.utils.data.DataLoader

    '''
    valset = get_val_dataset(valset)
    batch_size = 10
    
    val_loader = torch.utils.data.DataLoader(valset, 
                                             batch_size = batch_size, 
                                             shuffle = False)
    
    return val_loader
    
def get_image_list(dataset : db.CocoComplexityDataset, ids : list, 
                   n_images : int) -> list:
    '''
    Get list of image urls from dataset for img ids.

    Parameters
    ----------
    dataset : CocoComplexityDataset

    ids : list
        indices of images to get from dataset.
        
    n_images : int
        number of images to pull from dataset

    Returns
    -------
    list
        urls for ids

    '''
    ids = ids[:n_images]
    urls = [dataset.get_metadata(idx)['coco_url'] for idx in ids]
    
    return urls
    
def get_caption_list(dataset : db.CocoComplexityDataset, ids : list, 
                   n_images : int) -> list:
    '''
    Get list of image captions from dataset for img ids.

    Parameters
    ----------
    dataset : CocoComplexityDataset

    ids : list
        indices of images to get captions for from dataset.
        
    n_images : int
        number of image captions to pull from dataset

    Returns
    -------
    list
        captions for ids

    '''
    ids = ids[:n_images]
    captions = [dataset.get_captions(idx)[0]['caption'] for idx in ids]
    
    return captions
  
def get_precision_recall_data(predictions : dict) -> tuple:
    '''
    Extract sample-wise arrays of true labels and probability estimates of
    positive class (complex) for plotting precision-recall curve. Helper
    function for evaluate_model().

    Parameters
    ----------
    predictions : dict
        See evaluate_model()

    Returns
    -------
    tuple
        (y_true, probas_pred)

    '''
    y_true, probas_pred = np.array([]), np.array([])
    
    for image_id in predictions:
        num_captions = len(predictions[image_id]['captions'])
        y_true = np.append(y_true, [predictions[image_id]['label']]*num_captions)
        probas_pred = np.append(probas_pred, predictions[image_id]['scores'])
        
    return y_true, probas_pred


def evaluate_model(model_weights : str, dataset : str, normalize : bool,
                   complex_predictions : str, noncomplex_predictions : str,
                   n_images : int, category: str,  mode : str, out_file : str):
    '''
    Load pretrained PyTorch model with weights from .pth file, and evaluate on
    val set.

    Parameters
    ----------
    model_weights : str
        Path to .pth file with weights.
        
    dataset : str
        Path to dataset (.p file)    
    
    normalize : bool
        Whether to normalize model outputs via sigmoid() function.
        Should be True for classification models or regression models that
        compute normalized complexity score.
        
    complex_predictions : str
        Path to html file to write visualization of complex predictions.
        
    noncomplex_predictions : str
        Path to html file to write visualization of noncomplex predictions.

    n_images : int
        Number of images to show per page
        
    category: str
        COCO category images drawn from (will go in title of html pages)
        
    mode : str
        'regression' or 'classification' - used to determine loss function to
        use when calculating loss on entire val set
    
    out_file : str
        path to text file to print model, dataset, accuracy, and 
        loss
    
    Returns
    -------
    None
    '''
    valset = get_val_dataset(dataset)
    dataloader = get_val_dataloader(dataset)
    
    # load in model weights
    num_categories = 1
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', 
                                                            num_labels = num_categories,  
                                                            output_attentions = False, 
                                                            output_hidden_states = False)
    model.load_state_dict(torch.load(model_weights))
    model.eval();
    device = torch.device('cuda')
    model.to(device);
    
    # get predictions on entire val set
    if mode == 'classification':
        print('mode set to "classification" . . .', flush=True)
        cost_function = torch.nn.BCEWithLogitsLoss(reduction='none')
        cumulative_correct = 0
    elif mode == 'regression':
        print('mode set to "regression" . . .', flush=True)
        cost_function = torch.nn.MSELoss(reduction='none')
    else:
        raise ValueError('mode for evaluate_model() should be "classification" or "regression"')
        
    cumulative_loss = 0
    num_samples = 0
    
    predictions = {}
    idx = 0 # for retrieving image urls
    
    for (batch_id, (img_ids, imgs, texts, text_masks, labels)) in enumerate(dataloader):
        # Move to GPU.
        texts = texts.cuda()
        text_masks = text_masks.cuda()
        labels = labels.unsqueeze(-1).cuda()
    
        # Compute predictions.
        outputs = model(texts, text_masks)
        
        if normalize: # for storing predictions
            probabilities = outputs.logits.data.sigmoid() # normalize
            
        # for computing loss
        if mode == 'classification': # BCEWithLogitsLoss takes care of normalization
            loss = cost_function(outputs.logits, labels)
            cumulative_correct += count_correct(outputs.logits, labels)
        elif mode == 'regression': # MSE Loss expects normalized outputs
            loss = cost_function(outputs.logits.sigmoid(), labels)
            
        cumulative_loss += loss.sum().item()
        num_samples += texts.size(0)
        
        probabilities = probabilities.flatten().tolist()
        img_ids = img_ids.flatten().tolist()
        labels = labels.flatten().tolist()
        
        for p, i, t, l in zip(probabilities, img_ids, texts, labels):
            if i not in predictions:
                predictions[i] = {}
                predictions[i]['scores'] = []
                predictions[i]['captions'] = []
                predictions[i]['label'] = l
                predictions[i]['url'] = valset.get_metadata(idx)['coco_url']
                
            predictions[i]['scores'].append(p)
            tokens = valset.tokenizer.convert_ids_to_tokens(t, 
                                                            skip_special_tokens = True)
            caption = valset.tokenizer.convert_tokens_to_string(tokens)
            predictions[i]['captions'].append(caption)
            idx += 1
        
    # save accuracy and loss
    if mode == 'classification':
        accuracy, loss = cumulative_correct/num_samples, cumulative_loss/num_samples
        with open(out_file, 'a') as f:
            print(model_weights, dataset, str(accuracy), str(loss), sep='\t', 
                  file=f)
    elif mode == 'regression':
        loss = cumulative_loss/num_samples
        with open(out_file, 'a') as f:
            print(model_weights, dataset, str(loss), sep='\t', 
                  file=f)
        
    # get average prediction confidence per image and sort in descending order
    avg_scores = []
    for i in predictions:
        avg_scores.append((i, stats.mean(predictions[i]['scores'])))
    avg_scores_sorted = sorted(avg_scores, reverse=True, 
                               key=lambda x: x[1])
    
    # build html pages with captions, label, average score, and per caption
    # scores displayed
    image_info = []
    for img_id, score in avg_scores_sorted:
        caption = ''
        caption += 'Label: ' + str(predictions[img_id]['label']) + '\n'
        caption += 'Average score: ' + str(score) + '\n'
        caption += '\n'.join([str(s) + '\t' + c
                                for s, c in zip(predictions[img_id]['scores'],
                                  predictions[img_id]['captions'])])
        image_info.append((predictions[img_id]['url'], caption))
    
    complex_list = image_info[:n_images]
    noncomplex_list = image_info[:-n_images:-1]
    
    html_builder.write_header(complex_predictions,
                              'Top {0} predicted "complex" images from {1} category'.format(n_images, 
                                                                                            category))
    html_builder.write_images(complex_predictions, complex_list)
    html_builder.write_footer(complex_predictions)
    
    html_builder.write_header(noncomplex_predictions,
                              'Top {0} predicted "noncomplex" images from {1} category'.format(n_images, 
                                                                                            category))
    html_builder.write_images(noncomplex_predictions, noncomplex_list)
    html_builder.write_footer(noncomplex_predictions)

    return predictions

def count_prediction_categories(split : str, predictions : dict) -> dict:
    '''
    Count the number of complex/noncomplex predictions per COCO category.

    Parameters
    ----------
    split : str
        'train', 'val', 'test'. 'train' and 'val' splits constructed from
        COCO train set, 'test' split constructed from COCO val set.
        
    predictions : dict
        See evaluate_model

    Returns
    -------
    dict
        {coco_category_name : {'complex' : [num_complex_predictions],
                               'noncomplex' : [num_noncomplex_predictions]}}

    '''
    # build dict of img ids by category
    ids_by_cat = {}
    
    if (split == 'train') or (split == 'val'):
        coco=COCO('/scratch/data/COCO/annotations/instances_train2017.json')
    elif split == 'test':
        coco=COCO('/scratch/data/COCO/annotations/instances_val2017.json')
    else:
        raise ValueError('split must be train, val, or test')
        
    for category in COCO_CATEGORIES:
        catIds = coco.getCatIds(catNms=[category])
        ids_by_cat[category] = coco.getImgIds(catIds=catIds)
        
    # count samples per category
    complexity_counts_by_cat = {}
    for image_id in predictions:
        scores = np.array(predictions[image_id]['scores'])
        scores = np.where(scores > 0.5, 1, 0)
        
        for cat in ids_by_cat:
            if image_id in ids_by_cat[cat]:
                if cat not in complexity_counts_by_cat:
                    complexity_counts_by_cat[cat] = {'complex': 0, 
                                                     'noncomplex': 0}
                    
                complexity_counts_by_cat[cat]['noncomplex'] += scores.size - scores.sum()
                complexity_counts_by_cat[cat]['complex'] += scores.sum()
        
    return complexity_counts_by_cat
        

def main(args : list):
    '''
    Build html pages for qualitative analysis of image complexity classifier.

    Parameters
    ----------
    args : list
        command line args
        
        args[0] : name of script
        
        args[1] : dataset file
        
        args[2] : model weights file
        
        args[3], args[4] : names of files to write pages displaying top predicted
        complex/noncomplex images
        
        args[5] : number of images to display per page
        
        args[6] : COCO category of images
        
        args[7] : file to save predictions dict
        
        args[8] : mode ('regression' or 'classification')
        
        args[9] : out file for printing accuracy and loss
        
        args[10] : 'normalize': optional, whether to normalize predictions

    Returns
    -------
    None.

    '''
    if len(args) == 11 and args[-1] == 'normalize':
        print("Normalize model predictions . . .", flush=True)
        normalize = True
    else:
        normalize = False
        
    predictions = evaluate_model(model_weights=args[2], dataset=args[1],
                                 complex_predictions=args[3], 
                                 noncomplex_predictions=args[4],
                                 n_images=int(args[5]), category=args[6],
                                 normalize=normalize, mode=args[8],
                                 out_file=args[9])
    pickle.dump(predictions, open(args[7], 'wb'))
    
if __name__ == '__main__':
    main(sys.argv)