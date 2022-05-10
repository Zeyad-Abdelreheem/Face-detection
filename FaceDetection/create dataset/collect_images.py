#!/usr/bin/env python
# coding: utf-8


import os
import time
import uuid
import cv2.cv2 as cv2




def collect_pictures(IMAGES_PATH , number_images):
    """
    This function uses the web camera to collect images 
    
    Params :-
    
    IMAGES_PATH : the path of the folder that will store the images
    
    number_images : the number of samples to be created
    
    return : NONE
    
    
    """
    cap = cv2.VideoCapture(0)
    for img_num in range(number_images):
        print('Collecting image {}'.format(img_num))
        ret, frame = cap.read()
        imgname = os.path.join(IMAGES_PATH,f'{str(uuid.uuid1())}.jpg')
        cv2.imwrite(imgname, frame)
        cv2.imshow('frame', frame)
        time.sleep(0.8)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()





