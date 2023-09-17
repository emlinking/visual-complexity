# -*- coding: utf-8 -*-
"""
classify.py

A script with functions to train and evaluate a text-based classifier that 
identifies complex images from their captions.

Created on Thu Jun 23 12:26:58 2022

@author: linel
"""
import pickle
import torch
        
def count_correct(predicted, labels):
    """
    Parameters:
    predicted and labels are torch tensors of shape (batch_size, num_labels)
    predicted = logits of output object from model
    rows are samples

    Returns: number of correct predictions in the batch
    """
    probabilities = predicted.data.sigmoid()
    pred_vals = probabilities > 0.5 # Anything with sigmoid > 0.5 is 1.

    return (pred_vals == labels)[:, 0].sum().item()

def train_model1(model, num_epochs : int, optimizer, scheduler, cost_function, 
                train_loader, val_loader, model_file : str, log_file : str):
    """
    Train classification model and save best checkpoint.

    Parameters
    ----------
    model : Pretrained model, e.g. Hugging Face transformers.BertForSequenceClassification
        Model to finetune.
    num_epochs : int
        Number of epochs to finetune model over.
    optimizer : PyTorch optimizer, e.g., AdamW, to update model parameters
        during finetuning.
    scheduler : Learning rate scheduler, instantiated with total_steps.
    cost_function : torch.nn loss function, e.g., CrossEntropyLoss
    train_loader, val_loader : instances of torch.utils.data.DataLoader for
        their respective splits of dataset
    model_file : str
        name of file to save model.
    log_file : str
        name of file to save training progress.
        
    Returns
    -------
    None

    """
    # for tracking best model so far
    best_accuracy = 0
    logs = {'train_loss_by_batch': [], 'train_accuracy_by_batch': [], 
            'val_loss_by_batch': [], 'val_accuracy_by_batch': [], 
            'train_loss_by_epoch': [], 'val_loss_by_epoch': [], 
            'train_accuracy_by_epoch': [], 'val_accuracy_by_epoch': []} 
    
    # move model to GPU
    model.cuda()
    
    # loop through epochs
    for epoch in range(0, num_epochs):
        # variables to store training progress
        correct_predictions = 0
        cumulative_loss = 0
        num_samples = 0
    
        # make sure the model is in training mode (set to eval by default)
        model.train()
    
        # training: loop through batches
        for img_ids, imgs, texts, text_masks, labels in train_loader:
            # Move to GPU.
            texts = texts.cuda()
            text_masks = text_masks.cuda()
            labels = labels.unsqueeze(-1).cuda()
    
            # Compute predictions.
            predicted = model(texts, text_masks)
    
            # Compute loss.
            loss = cost_function(predicted.logits, labels)
    
            # Compute cumulative loss and accuracy.
            cumulative_loss += loss.data.sum().item() # .item() converts torch > number
            correct_predictions += count_correct(predicted.logits, labels)
            num_samples += texts.size(0)
            
            # update logs
            logs['train_loss_by_batch'].append(cumulative_loss / num_samples)
            logs['train_accuracy_by_batch'].append(correct_predictions / num_samples)
    
            # Backpropagation and SGD update step.
            model.zero_grad() # clear out gradients from previous batches
            loss.mean().backward() # calculate new gradients
            optimizer.step() # update parameters
            
        logs['train_loss_by_epoch'].append(cumulative_loss / num_samples)
        logs['train_accuracy_by_epoch'].append(correct_predictions / num_samples)
        
        # reset variables to track training progress
        correct_predictions = 0
        cumulative_loss = 0
        num_samples = 0
    
        # switch to evaluation mode
        model.eval()
    
        # eval loop
        for (img_ids, imgs, texts, text_masks, labels) in val_loader:
            # Move to GPU.
            texts = texts.cuda()
            text_masks = text_masks.cuda()
            labels = labels.unsqueeze(-1).cuda() # (10) -> (10, 1)
    
            # Compute predictions.
            # see output prediction attributes here:
            # https://huggingface.co/docs/transformers/main/en/main_classes/
            # output#transformers.modeling_outputs.SequenceClassifierOutput
            predicted = model(texts, text_masks)
    
            # Compute loss.
            loss = cost_function(predicted.logits, labels)
    
            # Compute cumulative loss and correct predictions
            cumulative_loss += loss.data.sum().item()
            correct_predictions += count_correct(predicted.logits, labels)
            num_samples += texts.size(0)
            
            # update logs
            logs['val_loss_by_batch'].append(cumulative_loss / num_samples)
            logs['val_accuracy_by_batch'].append(correct_predictions / num_samples)
    
        # report loss and accuracy on val set
        logs['val_loss_by_epoch'].append(cumulative_loss / num_samples)
        logs['val_accuracy_by_epoch'].append(correct_predictions / num_samples)   
    
        # Advance scheduler. - updates learning rate
        if scheduler != -1:
            scheduler.step()
    
        # Save the parameters for the best accuracy on the validation set so far.
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict
        if logs['val_accuracy_by_epoch'][-1] > best_accuracy:
            best_accuracy = logs['val_accuracy_by_epoch'][-1]
            torch.save(model.state_dict(), model_file) 
    
    # save training log
    pickle.dump(logs, open(log_file, 'wb'))
    
def train_model2(model, num_epochs : int, optimizer, scheduler, cost_function,
                 normalization : bool,
                train_loader, val_loader, model_file : str, log_file : str):
    """
    Train regression model and save best checkpoint.

    Parameters
    ----------
    model : Pretrained model, e.g. Hugging Face transformers.BertForSequenceClassification
        Model to finetune.
        
    num_epochs : int
        Number of epochs to finetune model over.
        
    optimizer : PyTorch optimizer, e.g., AdamW, to update model parameters
        during finetuning.
        
    scheduler : Learning rate scheduler, instantiated with total_steps.
    
    cost_function : torch.nn loss function, e.g., CrossEntropyLoss
    
    normalization : bool
        whether to apply sigmoid() normalization function to outputs before
        computing loss
    
    train_loader, val_loader : instances of torch.utils.data.DataLoader for
        their respective splits of dataset
        
    model_file : str
        name of file to save model.
        
    log_file : str
        name of file to save training progress.
        
    Returns
    -------
    None

    """
    # for tracking best model so far
    logs = {'train_loss_by_batch': [], 'val_loss_by_batch': [], 
            'train_loss_by_epoch': [], 'val_loss_by_epoch': []} 
    
    # move model to GPU
    model.cuda()
    
    # loop through epochs
    for epoch in range(0, num_epochs):
        # variables to store training progress
        cumulative_loss = 0
        num_samples = 0
    
        # make sure the model is in training mode (set to eval by default)
        model.train()
    
        # training: loop through batches
        for img_ids, imgs, texts, text_masks, labels in train_loader:
            # Move to GPU.
            texts = texts.cuda()
            text_masks = text_masks.cuda()
            labels = labels.unsqueeze(-1).cuda()
    
            # Compute predictions.
            predicted = model(texts, text_masks)
    
            # Compute loss.
            if normalization:
                loss = cost_function(predicted.logits.sigmoid(), labels)
            else:
                loss = cost_function(predicted.logits, labels)
    
            # Compute cumulative loss and accuracy.
            cumulative_loss += loss.data.sum().item() # .item() converts torch > number
            num_samples += texts.size(0)
            
            # update logs
            logs['train_loss_by_batch'].append(cumulative_loss / num_samples)
    
            # Backpropagation and SGD update step.
            model.zero_grad() # clear out gradients from previous batches
            loss.mean().backward() # calculate new gradients
            optimizer.step() # update parameters
            
        logs['train_loss_by_epoch'].append(cumulative_loss / num_samples)
        
        # reset variables to track training progress
        cumulative_loss = 0
        num_samples = 0
    
        # switch to evaluation mode
        model.eval()
    
        # eval loop
        for (img_ids, imgs, texts, text_masks, labels) in val_loader:
            # Move to GPU.
            texts = texts.cuda()
            text_masks = text_masks.cuda()
            labels = labels.unsqueeze(-1).cuda() # (10) -> (10, 1)
    
            # Compute predictions.
            # see output prediction attributes here:
            # https://huggingface.co/docs/transformers/main/en/main_classes/
            # output#transformers.modeling_outputs.SequenceClassifierOutput
            predicted = model(texts, text_masks)
    
            # Compute loss.
            if normalization:
                loss = cost_function(predicted.logits.sigmoid(), labels)
            else:
                loss = cost_function(predicted.logits, labels)
    
            # Compute cumulative loss and correct predictions
            cumulative_loss += loss.data.sum().item()
            num_samples += texts.size(0)
            
            # update logs
            logs['val_loss_by_batch'].append(cumulative_loss / num_samples)
    
        # report loss and accuracy on val set
        logs['val_loss_by_epoch'].append(cumulative_loss / num_samples)
    
        # Advance scheduler. - updates learning rate
        if scheduler != -1:
            scheduler.step()
    
        # Save the parameters for the best loss on the validation set so far.
        if logs['val_loss_by_epoch'][-1] <= min(logs['val_loss_by_epoch']):
            torch.save(model.state_dict(), model_file) 
    
    # save training log
    pickle.dump(logs, open(log_file, 'wb'))
    
def train_model(mode : str, model, num_epochs : int, optimizer, scheduler, 
                cost_function, train_loader, val_loader, model_file : str, 
                log_file : str, normalization : bool=None):
    if mode == 'classification':
        train_model1(model, num_epochs, optimizer, scheduler, cost_function, 
                     train_loader, val_loader, model_file, log_file)
    elif mode == 'regression':
        if normalization is not None:
            train_model2(model, num_epochs, optimizer, scheduler, cost_function, 
                         train_loader, val_loader, model_file, log_file,
                         normalization)
        else:
            raise ValueError('mode "regression" requires arg "normalization"')
    else:
        raise ValueError('mode must be "classification" or "regression"')