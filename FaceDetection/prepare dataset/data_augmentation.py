#!/usr/bin/env python
# coding: utf-8

import os
import cv2.cv2 as cv2 
import json
import numpy as np
import albumentations as alb


# In[ ]:


def augment_data(images_path , labels_path ,aug_data_path , aug_labels_path , aug_factor, frame_dims = [640,480,640,480]):
    
    """
    instantiate data augmentation pipeline to create augmented images 
    from images_path preserving the bounding boxes found in labels_path
    
    
    params:-
    
    
    images_path : directory of the folder containing images to be augmented
    
    labels_path : directory of the folder containing labels
                  (bounding box coordinates) to be augmented
    
    aug_data_path : directory of the folder that will contain augmented images 
    
    aug_labels_path : directory of the folder that will contain labels of augmented images
    
    
    returns NONE
    augmented data and labels will be created in the given paths
    
    """
    
    
    augmentor = alb.Compose([alb.RandomCrop(width=450, height=450), 
                         alb.HorizontalFlip(p=0.5), 
                         alb.RandomBrightnessContrast(p=0.2),
                         alb.RandomGamma(p=0.2), 
                         alb.RGBShift(p=0.2), 
                         alb.VerticalFlip(p=0.5)], 
                       bbox_params=alb.BboxParams(format='albumentations', 
                                                  label_fields=['class_labels']))
    
    
    for image in os.listdir(os.path.join(images_path)):
        img = cv2.imread(os.path.join(images_path, image))

        coords = [0,0,0.00001,0.00001]

        label_path = os.path.join(labels_path, f'{image.split(".")[0]}.json')


        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label = json.load(f)

            coords[0] = label['shapes'][0]['points'][0][0]
            coords[1] = label['shapes'][0]['points'][0][1]
            coords[2] = label['shapes'][0]['points'][1][0]
            coords[3] = label['shapes'][0]['points'][1][1]
            coords = list(np.divide(coords, frame_dims))

        try: 
            for x in range(aug_factor):

                augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])

                cv2.imwrite(os.path.join(aug_data_path, f'{image.split(".")[0]}.{x}.jpg'), augmented['image'])

                annotation = {}
                annotation['image'] = image

                if os.path.exists(label_path):
                    if len(augmented['bboxes']) == 0: 
                        annotation['bbox'] = [0,0,0,0]
                        annotation['class'] = 0 
                    else: 
                        annotation['bbox'] = augmented['bboxes'][0]
                        annotation['class'] = 1
                else: 
                    annotation['bbox'] = [0,0,0,0]
                    annotation['class'] = 0 


                with open(os.path.join(aug_labels_path, f'{image.split(".")[0]}.{x}.json'), 'w') as f:
                    json.dump(annotation, f)

        except Exception as e:
            print(e)



