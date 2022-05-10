# Face-detection
This repo goes through the object detection pipeline and takes the face as an example

# The expected output
<img src="/face_test.png" alt="face" title="test image">


# this readme file will work as an outline
#### feel free to skip to your section of interest


# Outline:-
## 1 - Collect the dataset
## 2 - Train test split
## 3 - Augment the data
## 4 - Build the model
## 5 - Train the model
## 5.1 - Evaluate the performance
## 6 - Real-time test



## 1 - Collect the dataset

web cam was used to capture images of faces and the library labelme (https://github.com/wkentaro/labelme) was used to annotate the images and draw bounding boxes

in FaceDetection/create dataset/collect_images.py you will find the code that access the webcam and capture images
in FaceDetection/create dataset/prepare_dataset.ipynb you will find the code that calls collect_images and launches labelme to start labeling

## 2 - Train test split

in FaceDetection/prepare dataset/train_test_split.py you will find the code that split the data into train and test and copy the images to the given locations

## 3 - Augment the data

in FaceDetection/prepare dataset/data_augmentation.py data is being augmented using albumentations library(https://albumentations.ai/)
NOTE : you might face some difficulties with albumentations as it depends on older versions of opencv so feel free to use any other data augmentation library

