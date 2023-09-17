# -*- coding: utf-8 -*-
"""
train_coco_class.py

A script to train BERT to classify complex v. non-complex COCO images.

Created on Wed Jun 22 15:27:39 2022

Usage: python train_coco_class.py {mode} {trainset-file} {valset-file} 
{model-weights-file} {log-file} {balanced} {train-time-log}

Optional:
'balanced' means use weighted random sampler, such that class
imbalances are corrected

@author: linel
"""
import classify
import dataset_builder as db
import pickle
import sys
import time
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup

def get_class_weights(dataset : db.CocoComplexityDataset):
    '''
    Get class weights for samples in dataset. 
    class weight = 1/(number of samples in class)

    Parameters
    ----------
    dataset : db.CocoComplexityDataset
        See dataset_builder module for details

    Returns
    -------
    class_weights_all : torch.tensor
        tensor of length len(dataset) containing class weights matching
        sample labels in dataset.

    '''
    target_list = torch.tensor([dataset.get_metadata(i)['label'] == 'complex' 
                                for i in range(len(dataset))], 
                               dtype=int)
    class_counts = [dataset.num_noncomplex, dataset.num_complex]
    class_weights = 1./torch.tensor(class_counts, dtype=torch.float) 
    class_weights_all = torch.gather(class_weights, 0, target_list) # noncomplex=0, complex=1
    
    return class_weights_all

def main(args):
    """
    args[0] is name of script
    args[1] is mode ('classification' or 'regression')
    args[2] should be name of file with trainset
    args[3] should be name of file with valset
    args[4] should be name of file to save model checkpoint
    args[5] should be name of file to save training logs
    args[6] - file to save training times to
    args[7] - 'normalize' - for regression training only. whether to normalize 
    outputs before computing loss.

    """
    start = time.perf_counter()
    
    # load dataset
    train2017 = pickle.load(open(args[2], 'rb'))
    val2017 = pickle.load(open(args[3], 'rb'))
    
    # set up DataLoaders
    batch_size = 10

    if args[1] == 'classification':
        
        print('Training with weighted random sampling ...')
        
        train_weights = get_class_weights(train2017)
        train_sampler = torch.utils.data.WeightedRandomSampler(weights=train_weights, 
                                                               num_samples=len(train2017),
                                                               replacement=True)
        train2017_loader = torch.utils.data.DataLoader(dataset=train2017, 
                                                   batch_size=batch_size, 
                                                   sampler=train_sampler,
                                                   pin_memory=True,
                                                   num_workers=2)
    
            
        val2017_loader = torch.utils.data.DataLoader(val2017, 
                                                 batch_size = batch_size, 
                                                 shuffle = False)
        
             
        # train and evaluate on rgb images
        num_categories = 1
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', 
                                                                num_labels = num_categories,  
                                                                output_attentions = False, 
                                                                output_hidden_states = False)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)
        num_epochs = 4
        total_steps = len(train2017_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=0, 
                                                     num_training_steps=total_steps)
        cost_function = nn.BCEWithLogitsLoss(reduction='none')
        
        classify.train_model1(model, num_epochs, opt, scheduler, cost_function,
                             train_loader=train2017_loader, 
                             val_loader=val2017_loader,
                             model_file=args[4],
                             log_file=args[5])
    
    elif args[1] == 'regression':
        train2017_loader = torch.utils.data.DataLoader(train2017, batch_size,
                                                       shuffle=True, 
                                                       pin_memory=True, 
                                                       num_workers=2)
        
        val2017_loader = torch.utils.data.DataLoader(val2017, 
                                                 batch_size = batch_size, 
                                                 shuffle = False)
        
        # train and evaluate on rgb images
        num_categories = 1
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', 
                                                                num_labels = num_categories,  
                                                                output_attentions = False, 
                                                                output_hidden_states = False)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)
        num_epochs = 4
        total_steps = len(train2017_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=0, 
                                                     num_training_steps=total_steps)
        cost_function = nn.MSELoss(reduction='none')
        
        if len(args) == 8:
            if args[-1] == 'normalize':
                print('Training regression model with normalized outputs . . .',
                      flush=True)
                classify.train_model2(model=model, num_epochs=num_epochs, 
                                      optimizer=opt, scheduler=scheduler, 
                                      cost_function=cost_function, 
                                      train_loader=train2017_loader, 
                                      val_loader=val2017_loader, 
                                      model_file=args[4], log_file=args[5], 
                                      normalization=True)
            else:
                raise ValueError('unrecognized argument "{}"'.format(args[-1]))
        elif len(args) == 7:
            print('Training regression model with raw outputs . . .', 
                  flush=True)
            classify.train_model2(model=model, num_epochs=num_epochs, 
                                  optimizer=opt, scheduler=scheduler, 
                                  cost_function=cost_function, 
                                  train_loader=train2017_loader, 
                                  val_loader=val2017_loader, 
                                  model_file=args[4], log_file=args[5], 
                                  normalization=False)
        else:
            raise ValueError('Too many or too few args -- check documentation.')
                
    stop = time.perf_counter()
    with open(args[6], 'a') as f:
        print(args[4], 'Time to run script: {0:.4f} s'.format(stop - start),
              file=f)
    
if __name__ == '__main__':
    main(sys.argv)