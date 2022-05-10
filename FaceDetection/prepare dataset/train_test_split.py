#!/usr/bin/env python
# coding: utf-8



import shutil
import os



def train_test_split(split_ratio , IMAGES_PATH , train_path , test_path):
    """
    moves all files from IMAGES_PATH to train and test folders
    based on the given split ratio
    
    
    params:-
    
    split_ratio : number of train samples / total number of images
    
    IMAGES_PATH : directory of the folder that contains all images
    
    train_path : directory of the folder which will contain the training examples
    
    test_path : directory of the folder which will contain test examples
    
    returns : NONE
    the folder IMAGES_PATH will be empty after execution 
    
    
    """
    
    
    
    
    num_samples = len(os.listdir(IMAGES_PATH))
    
    # to make sure it is between 0 and 1 
    split_ratio = split_ratio % 1
    
    
    num_train_samples = split_ratio * num_samples
    
    ctr = 0
    for file in os.listdir(IMAGES_PATH):
        
        original = os.path.join(IMAGES_PATH,file)
        
        shutil.move(original, train_path)
        
        ctr += 1
        
        if ctr > num_train_samples:
            break
        
        


    for file in os.listdir(IMAGES_PATH):
        original = os.path.join(IMAGES_PATH,file)
        
        shutil.move(original, test_path)

