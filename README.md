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

#### -------------------------------------------------

## 1 - Collect the dataset

web cam was used to capture images of faces and the library labelme [2] was used to annotate the images and draw bounding boxes

in FaceDetection/create dataset/collect_images.py you will find the code that access the webcam and capture images

in FaceDetection/create dataset/prepare_dataset.ipynb you will find the code that calls collect_images and launches labelme to start labeling

## 2 - Train test split

in FaceDetection/prepare dataset/train_test_split.py you will find the code that split the data into train and test 
then copy the images to the given locations

## 3 - Augment the data

in FaceDetection/prepare dataset/data_augmentation.py data is being augmented using albumentations library [3]

NOTE : you might face some difficulties with albumentations as it depends on older versions of opencv so feel free to use any other data augmentation library


## 4 - Build the model

in /FaceDetection/prepare dataset/custom_model.py you will find the class that instantiate the model and defines the custom loss (regression loss)

as the bounding box loss is a regression problem where it is needed to minimize the difference between the coordinates predicted and labeled

## 5 -  Train the model

in /FaceDetection/prepare dataset/training.ipynb a model is created , compiled and trained on the loaded dataset

NOTE : it might make the code less clean to put the custom model and training in the prepare dataset folder

but this will be less painful to import functions from other folders

there are many ways to do so but I found this the simplest one in this case


## 5.1 - Evaluate the performance
feel free to test the model on any dataset you would like

I decided to test the model in real time (this will be my metrics of evalutaion)

## 6 - real time test

in real_time_test.py you will find the code that access the web cam then feed those pics to the model to get the bounding box coordinates



# Resources:-
## [1] Nicholas renotte youtube channel : https://www.youtube.com/watch?v=N_W4EYtsa10
## [2] labelme :  https://github.com/wkentaro/labelme
## [3] albumentations : https://albumentations.ai/
## [4] VGG16 : https://keras.io/api/applications/vgg/
