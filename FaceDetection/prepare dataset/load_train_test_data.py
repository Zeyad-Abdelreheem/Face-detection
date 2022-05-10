#!/usr/bin/env python
# coding: utf-8


import json
import tensorflow as tf
from tensorflow.data import Dataset



def load_labels(label_path):
    """
    load labels (json format) then spread it
    
    
    returns the class , bounding box coordinates
    """
    
    with open(label_path.numpy(), 'r', encoding = "utf-8") as f:
        label = json.load(f)

    
    return [label['class']], label['bbox']



def load_image(x): 
    """
    load the image then decode it
    
    params:-
    
    x : image path
    
    
    returns decoded image
    """
    
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img


def load_train_test_data(train_img_path,
                         test_img_path,
                         train_labels_path,
                         test_labels_path,
                         img_size,
                         batch_size,
                         prefetch_batch_sz,
                         shuffle_buffer_size,rescale = 1/255 ):
    
    
    """
    load the training and testing labels and examples
    
    params :-
    
    train_img_path : path of the folder that contains training images

     test_img_path : path of the folder that contains testing images
     
     train_labels_path : path of the folder that contains training labels
     
     test_labels_path : path of the folder that contains testing labels
     
     img_size : size of the input image of the model
     
     batch_size : how many samples are to be processed together
     
     prefetch_batch_sz : how many batches are to be prefetched while the model is training
     
     shuffle_buffer_size : how many samples to be buffered when shuffling
     
     rescale : by how much images should be rescaled

    returns (train,test) tensorflow datasets
    
    """
    
    
    

    train_images = Dataset.list_files(train_img_path + '/*.*', shuffle=False)
    train_images = train_images.map(load_image)
    train_images = train_images.map(lambda x: tf.image.resize(x, img_size))
    train_images = train_images.map(lambda x: x*rescale)


    test_images = Dataset.list_files(test_img_path + '/*.*', shuffle=False)
    test_images = test_images.map(load_image)
    test_images = test_images.map(lambda x: tf.image.resize(x, img_size))
    test_images = test_images.map(lambda x: x*rescale)


    train_labels = Dataset.list_files(train_labels_path + '/*.json', shuffle=False)
    train_labels = train_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

    test_labels = tf.data.Dataset.list_files(test_labels_path + '/*.json', shuffle=False)
    test_labels = test_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))



    train = tf.data.Dataset.zip((train_images, train_labels))
    train = train.shuffle(shuffle_buffer_size)
    train = train.batch(batch_size)
    train = train.prefetch(prefetch_batch_sz)


    test = tf.data.Dataset.zip((test_images, test_labels))
    test = test.shuffle(shuffle_buffer_size)
    test = test.batch(batch_size)
    test = test.prefetch(prefetch_batch_sz)
    
    
    return train , test




