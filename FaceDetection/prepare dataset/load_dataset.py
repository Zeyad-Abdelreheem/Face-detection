#!/usr/bin/env python
# coding: utf-8


from tensorflow.data import Dataset
import json
import numpy as np
from matplotlib import pyplot as plt



def load_dataset(IMAGES_PATH , batch_size):
    """
    creates a tensorflow dataset generator from the files 
    in the given path
    
    
    params:-
    
    IMAGES_PATH : directory of the folder that contains the dataset (images)
    
    batch_size : the number of samples to be collected together to be processed
    
    returns  numpy iterator dataset with all samples in the given path
    
    """
    
    
    # collects all files in IMAGES_PATH with pattern ***.*** 
    img_dataset = Dataset.list_files(IMAGES_PATH+'/*.*',
                                     shuffle = False)
    
    image_generator = img_dataset.batch(batch_size).as_numpy_iterator()
    
    return image_generator






