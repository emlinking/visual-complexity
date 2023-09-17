# -*- coding: utf-8 -*-
"""
dataset_builder.py

Functions to build dataset of complex/non-complex images based on # of
mean-shift segmented distinct regions.

Created on Wed Jun 22 15:46:32 2022

@author: linel
"""
import copy
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
import os, json
from PIL import Image
from pycocotools.coco import COCO
import random
import sys
import torch
import torchvision
import torchvision.transforms as transforms
from transformers import BertTokenizer

def get_complexity_scores(path : str, filter_grayscale : bool) -> list:
    """
    Read in complexity data from directory path.

    Parameters
    ----------
    path : str
        Path to directory containing text files recording COCO image id, 
        # of image regions using pymeanshift, # of distinct regions, and 
        (optionally) a flag for whether the image was converted from a 2D 
        grayscale image. Each field is on a separate line.
    filter_grayscale : bool
        Whether to exclude grayscale images from dataset.
        
    Returns
    -------
    List of tuples (image_id, image_regions, distinct_image_regions, 
                    is_converted_from_grayscale_2d)

    """
    output_files = os.listdir(path)
    outputs = list()
    
    for file in output_files:
        image_id = int(file[:-len('.jpg.txt')])
        fp = os.path.join(path, file)
        
        with open(fp, 'r') as f:
            lines = f.readlines()
            if lines[-1] == 'grayscale 2d':
                if not filter_grayscale:
                    outputs.append((image_id, int(lines[0]), int(lines[1]), 
                                     True))
            else:
                outputs.append((image_id, int(lines[0]), int(lines[1]), 
                                     False))
                
    return outputs

def sort_scores(scores : list) -> list:
    """
    Sort list of (image_id, image_regions, distinct_image_regions, 
                    is_converted_from_grayscale_2d) tuples by
    distinct_image_regions.

    Parameters
    ----------
    scores : list
        list of (image_id, image_regions, distinct_image_regions, 
                        is_converted_from_grayscale_2d) tuples

    Returns
    -------
    list
        list of (image_id, image_regions, distinct_image_regions, 
                        is_converted_from_grayscale_2d) tuples sorted by
        distinct_image_regions.

    """
    return sorted(scores, key=lambda img: img[2], reverse=True)

def split_by_complexity(sorted_scores : list, p : float) -> dict:
    """
    Split top and bottom p% of images with most/fewest distinct regions and 
    label as complex/noncomplex to create dataset dictionary.
    Parameters
    ----------
    sorted_scores : list
        list of (image_id, image_regions, distinct_image_regions, 
                        is_converted_from_grayscale_2d) tuples sorted by
        distinct_image_regions.
    p : float
        Percentile cutoff to determine which images will be counted as most/
        least complex.

    Returns
    -------
    dict
        {img_id : {'regions' : regions, 'regions_filtered' : regions_filtered, 
                   'was_grayscale_2d' : True or False, 'label' : 'noncomplex' or 
                   'complex'}}
    """
    # split by complexity
    num_imgs = int(len(sorted_scores)*p)
    
    complex_imgs = sorted_scores[:num_imgs]
    noncomplex_imgs = sorted_scores[-num_imgs:]
    
    # convert lists to dictionaries indexed by image ids
    complex_img_dict = {img_id : {'regions' : regions, 
                                  'regions_filtered' : regions_filtered, 
                                  'was_grayscale_2d' : gray, 'label' : 'complex'}
                        for img_id, regions, regions_filtered, gray in 
                        complex_imgs}
    noncomplex_img_dict = {img_id : {'regions' : regions, 
                                  'regions_filtered' : regions_filtered, 
                                  'was_grayscale_2d' : gray, 
                                  'label' : 'noncomplex'}
                        for img_id, regions, regions_filtered, gray in 
                        noncomplex_imgs}
    
    # merge dictionaries
    complexity_data = {}
    complexity_data.update(noncomplex_img_dict)
    complexity_data.update(complex_img_dict)
    
    return complexity_data
    
def split_val_from_train1(train_complexity_data : dict, n_images : int) -> tuple:
    """
    Split train set into train and val by randomly sampling n_images from
    complex and noncomplex images in train set.

    Parameters
    ----------
    train_complexity_data : dict
        Output from split_by_complexity, full dict of 23.2k
    n_images : int
        Number of images to put in val set. Will be split 50-50 between complex/
        noncomplex images.

    Returns
    -------
    tuple
        (new_train_set, new_val_set)

    """
    
    # separate dataset into complex and noncomplex
    train_set_noncomplex = {}
    train_set_complex = {}
    for img_id, sample in train_complexity_data.items():
        if sample['label'] == 'noncomplex':
            train_set_noncomplex[img_id] = sample
        else:
            train_set_complex[img_id] = sample
            
    # randomly sample subset of n_images for val set from train set
    complex_val_img_ids = random.sample(list(train_set_complex.keys()), 
                                        int(n_images/2))
    noncomplex_val_img_ids = random.sample(list(train_set_noncomplex.keys()), 
                                           int(n_images/2))
    
    # remaining images not sampled stay in train set
    val_set_complex, val_set_noncomplex = {}, {}
    for k in complex_val_img_ids:
        val_set_complex[k] = train_set_complex[k]
        del train_set_complex[k]
    for k in noncomplex_val_img_ids:
        val_set_noncomplex[k] = train_set_noncomplex[k]
        del train_set_noncomplex[k]
        
    # merge and return new sets
    train_complexity_data, val_complexity_data = {}, {}
    train_complexity_data.update(train_set_complex)
    train_complexity_data.update(train_set_noncomplex)
    val_complexity_data.update(val_set_complex)
    val_complexity_data.update(val_set_noncomplex)
    
    return train_complexity_data, val_complexity_data

def split_val_from_train2(train_complexity_data : list, n_images : int) -> tuple:
    """
    Split train set into train and val by randomly sampling n_images from
    train set.

    Parameters
    ----------
    train_complexity_data : list
        Output from normalize_regression_data.py
    n_images : int
        Number of images to put in val set.
        
    Returns
    -------
    tuple
        (new_train_set, new_val_set)

    """
    train_complexity_data = json.load(open(train_complexity_data, 'r'))
    val_idx = random.sample(range(len(train_complexity_data)), n_images)
    
    val_set, train_set = [], copy.deepcopy(train_complexity_data)
    for idx in val_idx:
        val_set.append(train_complexity_data[idx])
        train_set.remove(train_complexity_data[idx])
    
    return (train_set, val_set)
    
def split_val_from_train(mode : str, train_complexity_data, 
                         n_images : int) -> tuple:
    if mode == 'classification':
        return split_val_from_train1(train_complexity_data, n_images)
    elif mode == 'regression':
        return split_val_from_train2(train_complexity_data, n_images)
    else:
        raise ValueError('Invalid value for arg "mode": "{}". Must be "classification" or "regression"'.format(mode))
        
def build_dataset(data : str, p : float, filter_grayscale : bool) -> dict:
    """
    Build dataset of complex/noncomplex images in dict format.

    Parameters
    ----------
    data : str
        Path to directory containing text files recording COCO image id, 
        # of image regions using pymeanshift, # of distinct regions, and 
        (optionally) a flag for whether the image was converted from a 2D 
        grayscale image. Each field is on a separate line.
    p : float
        Percentile cutoff to determine which images will be counted as most/
        least complex.
    filter_grayscale : bool
        Whether to filter grayscale images so as to exclude from dataset.

    Returns
    -------
    dict
        {img_id : {'regions' : regions, 'regions_filtered' : regions_filtered, 
                   'was_grayscale_2d' : True or False, 'label' : 'noncomplex' or 
                   'complex'}}
    """
    complexity_scores = get_complexity_scores(data, filter_grayscale)
    sorted_scores = sort_scores(complexity_scores)
    complexity_dataset = split_by_complexity(sorted_scores, p)
    
    return complexity_dataset

def mask_words(sentence : str, mask_nouns : bool=False, 
               mask_verbs : bool=False, mask_adjs : bool=False, 
               mask_advs : bool=False) -> str:
    '''
    Mask nouns with 'object(s)', verbs with 'act(ed/ing/s)', adjectives and 
    adverbs with 'plain(er/est/ly).'
    '''
    tokenized_text_list = word_tokenize(sentence)
    tagged_words_list = nltk.pos_tag(tokenized_text_list)
    masked_words_list = []
    for word, tag in tagged_words_list:
        if mask_nouns:
            if tag in ['NN', 'NNP']: # singular nouns
                masked_words_list.append('object')
                continue
            elif tag in ['NNS', 'NNPS']: # plural nouns
                masked_words_list.append('objects')
                continue
        if mask_verbs:
            if tag in ['VB', 'VBP']: # base or present tense (not 3rd person sing) verb
                masked_words_list.append('act')
                continue
            elif tag in ['VBD', 'VBN']: # past tense or past participle
                masked_words_list.append('acted')
                continue
            elif tag in ['VBG']: # present participle or gerund
                masked_words_list.append('acting')
                continue
            elif tag in ['VBZ']: # present 3rd person singular
                masked_words_list.append('acts')
                continue
        if mask_adjs:
            if tag in ['JJ']: # adjective or ordinal numeral
                masked_words_list.append('plain')
                continue
            elif tag in ['JJR']: # comparative adjective
                masked_words_list.append('plainer')
                continue
            elif tag in ['JJS']: # superlative adjective
                masked_words_list.append('plainest')
                continue
        if mask_advs:
            if tag in ['RB']: # adverb
                masked_words_list.append('plainly')
                continue 
            elif tag in ['RBR']: # comparative adverb
                masked_words_list.append('plainer')
                continue
            elif tag in ['RBS']: # superlative adverb
                masked_words_list.append('plainest')
                continue
                       
        # not a mask target
        masked_words_list.append(word)
    
    return ' '.join(masked_words_list)

class CocoComplexityDataset(torch.utils.data.Dataset):
    def __init__(self, mode : str, split : str, data : str, caption_ann : str, 
                 images : str, transform=transforms.ToTensor(), 
                 instance_ann : str=None, obj_class : list=None, 
                 noun_mask : str=False, adj_mask : bool=False, 
                 verb_mask : bool=False, adv_mask : bool=False):
        """
        Initialize a COCO complexity dataset where each dataset item is an image including COCO
        annotations, filename, mean-shift segmented number of regions, distinct number of regions,
        and whether or not image was converted from grayscale.

        Parameters:
            
        mode : 'classification' or 'regression'
        
        split : train or val
        
        data : path to file with complexity data for images
        
        caption_ann : path to file with COCO captioning annotations for images
        
        instance_ann : path to file with COCO instances annotations for images
        
        images : path to directory with images
        
        obj_class : if not None, filter for images containing instances of these
        object class labels. Must be 1 of the 80 included in COCO.
        
        transform : torchvision transformation to apply to images, for purpose
        of easier viewing

        CocoComplexityDataset attributes:
        split : 'train' or 'val'
        
        info : dataset description, url, version, year, contributor, and date of creation
        
        licenses : types of copyright licenses for images in dataset
        
        image_transform : transformation to apply to images before returning in __get_item__
        
        images : str
            path to directory containing images
            
        categories : 'complex' and 'noncomplex'
        
        categories2ids : dictionary mapping categories to labels
        
        metadata : list of COCO annotations including the following fields for each image: license, filename,
        image url on COCO website, height, width, date captured, image url on Flickr, and image id.
        Augmented with fields for # of image regions (filtered and unfiltered), whether image was
        converted from grayscale, and groundtruth complexity label.
        
        captions : list of COCO captions and caption metadata for each image
        
        tokenizer : bert-base-uncased pretrained tokenizer
        
        tokenized_captions : using bert-base-uncased pretrained tokenizer
        
        obj_class : list of COCO object categories to filter for. If not None,
        dataset will include all images including any of the categories listed
        in obj_class.
        
        num_complex, num_noncomplex : int, int
            number of complex/noncomplex samples in dataset
        
        noun_mask : bool, optional (default: False)
            should nouns be replaced by the word "object(s)"?
            
        verb_mask : bool, optional (default: False)
            should verbs be replaced by the word "act(ing/ed/s)"?
            
        adj_mask : bool, optional (default: False)
            should adjectives be replaced by the word "plain(er/est)"?
            
        adv_mask : bool, optional (default: False)
            should adverbs be replaced by the word "plain(ly/er/est)"?
        """
        cap_ann = json.load(open(caption_ann, 'r'))
        
        # initialize COCO api to filter by instance and get list of image ids
        # for images containing desired instance
        # note that a given image may have more than one instance of an object
        # type and/or multiple object types matching obj_class, resulting in
        # multiple copies of the image id in instance_match_ids
        # however indexing step ensures each image occurs at most once in 
        # final dataset
        self.obj_class = obj_class
        instance_match_ids = []
        if (instance_ann is not None) and (obj_class is not None):
            print("Filtering for {} images . . .".format(obj_class))
            coco = COCO(instance_ann)
            
            for c in obj_class:
                cat_id = coco.getCatIds(catNms=[c])
                instance_match_ids += coco.getImgIds(catIds=cat_id)
        
        # api for captions
        coco_caps = COCO(caption_ann)
        
        self.mode = mode
        self.split = split
        self.info = cap_ann['info']
        self.licenses = cap_ann['licenses']
        self.image_transform = transform
        self.categories = ['noncomplex', 'complex']
        self.categories2ids = {category: id for (id, category) 
                               in enumerate(self.categories)}
        self.images = images
        
        complexity_data = json.load(open(data, 'r'))
        self.num_complex = 0
        self.num_noncomplex = 0
        
        self.noun_mask, self.adj_mask, self.verb_mask, self.adv_mask = noun_mask, \
            adj_mask, verb_mask, adv_mask
            
        # index coco captioning metadata
        print('Indexing metadata . . . ', end='', flush=True)
        self.metadata = []
        self.captions = []
        
        if self.mode == 'classification':
            curr_len = 0
            for image in cap_ann['images']:
                image_id = image['id']
                
                # filter for images in complexity dataset
                if str(image_id) in complexity_data:
                        
                    # filter for instance
                    if instance_ann != None and obj_class != None:
                        if image_id not in instance_match_ids:
                            continue # skip images without target instance
                            
                    # add image to dataset
                    self.metadata.append(image)
                    
                    # add labels, # of regions and grayscale data to metadata
                    self.metadata[curr_len]['label'] = complexity_data[str(image_id)]['label']
                    self.metadata[curr_len]['regions'] = complexity_data[str(image_id)]['regions']
                    self.metadata[curr_len]['regions_filtered'] = complexity_data[str(image_id)]['regions_filtered']
                    self.metadata[curr_len]['was_grayscale_2d'] = complexity_data[str(image_id)]['was_grayscale_2d']
                    
                    if complexity_data[str(image_id)]['label'] == 'complex':
                        self.num_complex += 1
                    else:
                        self.num_noncomplex += 1
                        
                    curr_len = self.num_complex + self.num_noncomplex
                    
                    # add captions for this image
                    caption_ids = coco_caps.getAnnIds(imgIds=image_id)
                    captions = coco_caps.loadAnns(caption_ids)
                    for caption in captions:
                        self.captions.append(caption)
                        
                    # add duplicate metadata entries for each caption added
                    for i in range(len(captions) - 1):
                        self.metadata.append(self.metadata[curr_len-1])
                        
                        if self.metadata[curr_len]['label'] == 'complex':
                            self.num_complex += 1
                        else:
                            self.num_noncomplex += 1
                        curr_len = self.num_complex + self.num_noncomplex
        elif self.mode == 'regression':
            complexity_data_dicts = {sample[0] : {'regions' : sample[1],
                                                'score' : sample[2],
                                                'gray' : sample[3]} 
                                    for sample in complexity_data}
            
            for image in cap_ann['images']:
                image_id = image['id']
                
                # filter for images in complexity dataset
                if image_id in complexity_data_dicts:
                
                    # filter for instance
                    if instance_ann != None and obj_class != None:
                        if image_id not in instance_match_ids:
                            continue # skip images without target instance
                        
                    self.metadata.append(image)
                    self.metadata[-1]['score'] = complexity_data_dicts[image_id]['score']
                    self.metadata[-1]['regions'] = complexity_data_dicts[image_id]['regions']
                    self.metadata[-1]['was_grayscale_2d'] = complexity_data_dicts[image_id]['gray']
                    
                    # add captions for this image
                    caption_ids = coco_caps.getAnnIds(imgIds=image_id)
                    captions = coco_caps.loadAnns(caption_ids)
                    for caption in captions:
                        self.captions.append(caption)
                        
                    # add duplicate metadata entries for each caption added
                    for i in range(len(captions) - 1):
                        self.metadata.append(self.metadata[-1])
        print('finished', flush=True)
        
        # tokenize captions
        print('Tokenizing captions . . . ', end = '', flush=True)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenized_captions = []
        for caption in self.captions:
            text = caption['caption']
            
            # mask nouns, verbs, adjectives, adverbs
            text = mask_words(sentence=text, mask_nouns=self.noun_mask,
                              mask_verbs=self.verb_mask, 
                              mask_adjs=self.adj_mask, mask_advs=self.adv_mask)
            
            # tokenize (optionally masked) caption
            encoded_text = self.tokenizer.encode_plus(
                text, 
                add_special_tokens = True, # [CLS], [SEP] 
                truncation = True, # if len(text) > 128, truncate
                max_length = 128, 
                padding = 'max_length', # if len(text) < 128, pad
                return_attention_mask = True, # tells which positions are padding
                return_tensors = 'pt') # as PyTorch tensors
            
            self.tokenized_captions.append(encoded_text)
        print('finished', flush=True)
    
    def __getitem__(self, index: int):
        """
        Defines behavior when indexing CocoComplexityDataset object.

        Parameters:
            
        index : int
            index of sample in dataset (NOT the same as image id)

        Returns: A tuple of (img_id, img, text, text_mask, label) for image at
        index
        """
        img_id = self.metadata[index]['id']
        img_path = os.path.join(self.images, '{0:012d}.jpg'.format(img_id))
        img = self.image_transform(Image.open(img_path).convert('RGB'))
        text = self.tokenized_captions[index]['input_ids'][0]
        text_mask = self.tokenized_captions[index]['attention_mask'][0]
        if self.mode == 'classification':
            label = torch.tensor(self.categories2ids[self.metadata[index]['label']],
                                 dtype=torch.float32)
        elif self.mode == 'regression':
            label = torch.tensor(self.metadata[index]['score'], 
                                 dtype=torch.float32)
            
        return img_id, img, text, text_mask, label

    def load_image_only(self, index: int):
        """
        Load image at index in CocoComplexityDataset.

        Parameters: 
        index: int - the index of the image in the dataset.

        Returns:
        image: an RGB PIL Image
        """
        img_id = self.metadata[index]['id']
        img_path = os.path.join(self.images, '{0:012d}.jpg'.format(img_id))
        img = Image.open(img_path).convert('RGB')

        return img

    def get_metadata(self, index: int):
        """
        Return metadata for image at index in dataset.
        """
        return self.metadata[index]

    def get_caption(self, index: int):
        """
        Return caption at index.
        Note: index =/= image id
        """
        return self.captions[index]
        
    def __len__(self):
        return len(self.metadata)
    
    def __str__(self):
        '''
        Generate string description of dataset

        Returns
        -------
        s : str
            Dataset description including split (train/val/test), licensing
            info, image transform, image directory, categories (complex/noncomplex),
            dataset size, COCO object category/categories, and first
            metadata entry.

        '''
        s = 'Split: {}\n'.format(self.split)
        s += 'Mode: {}\n'.format(self.mode)
        s += 'COCO dataset info: {}\n'.format(self.info)
        s += 'Image licenses: {}\n'.format(self.licenses)
        s += 'Image transform: {}\n'.format(self.image_transform)
        s += 'Images: {}\n'.format(self.images)
        s += 'Size of dataset: {}\n'.format(self.__len__())
        s += 'Mask nouns: {}\n'.format(self.noun_mask)
        s += 'Mask verbs: {}\n'.format(self.verb_mask)
        s += 'Mask adjectives: {}\n'.format(self.adj_mask)
        s += 'Mask adverbs: {}\n'.format(self.adv_mask)
        if self.mode == 'classification':
            s += 'Categories: {}\n'.format(self.categories)
            s += 'Complex image captions: {}\n'.format(self.num_complex)
            s += 'Noncomplex image captions: {}\n'.format(self.num_noncomplex)
        if self.obj_class != None:
            s += 'Filter for instance type: {}\n'.format(self.obj_class)
        if self.__len__() != 0:
            s += 'First metadata entry: {}\n'.format(self.get_metadata(0))
        
        return s
    
    def show_image_group(self, image_ids : list, n_images : int) -> None:
        """
        function to visualize group of sample images
        
        Parameters:
            
        image_ids : list
            index of image in dataset, NOT COCO image id
        
        n_images : int
            number of images to display
        """
        image_ids = image_ids[:n_images]
        Transf = transforms.Compose([transforms.CenterCrop((256, 224)), transforms.ToTensor()])
        imgs = [Transf(self.load_image_only(id)) for id in image_ids]
        grid = torchvision.utils.make_grid(imgs, nrow = 10)
        plt.figure(figsize=(20,10)); plt.axis(False)
        plt.imshow(transforms.functional.to_pil_image(grid));
        
    def show_caption_group(self, image_ids : list, n_images : int) -> None:
        """
        image_ids : index of image in dataset, NOT COCO image id
        """
        image_ids = image_ids[:n_images]
        for image_id in image_ids:
            print(self.get_caption(image_id))
            
def main(args : list) -> None:
    """
    Create dataset of complex/noncomplex-labeled images from precomputed data.

    Parameters
    ----------
    args : list
        args[0] : script name
        
        args[1] : mode - 'classification' or 'regression'
        
        args[2] : path to COCO trainset complexity data
        
        args[3] : path to COCO valset complexity data
        
        args[4] : path to save train set dataset
        
        args[5] : path to save val set dataset
        
        args[6] : path to save test set dataset
        
        args[7] : whether to filter grayscale images (0 = False, 1 = True)
        
    Returns
    -------
    None

    """
    mode = args[1]
    if mode == 'classification':
        train2017 = build_dataset(data=args[2], p=0.1, 
                                  filter_grayscale=bool(int(args[7])))
        test2017 = build_dataset(data=args[3], p=0.1, 
                                 filter_grayscale=bool(int(args[7])))
        train2017, val2017 = split_val_from_train(mode='classification',
                                                  train_complexity_data=train2017, 
                                                  n_images=1000)
        
        with open(args[4], 'w') as f:
            json.dump(obj=train2017, fp=f)
        with open(args[5], 'w') as f:
            json.dump(obj=val2017, fp=f)
        with open(args[6], 'w') as f:
            json.dump(obj=test2017, fp=f)
            
    elif mode == 'regression':
        train2017, val2017 = split_val_from_train(mode, train_complexity_data=args[2], 
                                         n_images=5000)
        
        with open(args[4], 'w') as f:
            json.dump(obj=train2017, fp=f)
        with open(args[5], 'w') as f:
            json.dump(obj=val2017, fp=f)
    else:
        raise ValueError('Invalid mode "{}" selected. Mode must be "classification" or "regression"'.format(mode))
        
if __name__ == '__main__':
    main(sys.argv)