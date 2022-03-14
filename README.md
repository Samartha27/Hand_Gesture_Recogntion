# OpenCV_Calculator

## Description
Hand gesture recognition is an important way of communicating among humans and also between human and a machine. This has a wide range of applications, from interacting with robots, to interacting with laptops, mobile phones and helping specially abled people to express themselves. 

Our goal in this project is to build and apply various models which recognize hand gestures and use it to make an onscreen real time calculator. The model is trained to recognize 18 classes with each class having 900 training images. 

We have trained the dataset on various standard architectures like VGG, Inception, Resnet, Squeezenet, Alexnet and Densenet and obtained good accuracies above 98% for all the models.

## Process
We need to first segment the hand from the video sequence, by removing the background, motion detection and then extracting the contour of the hand. Using these real-time generated masks we wanted to predict the gestures made using our various architectures. We then give these masked images as inputs to the saved model to obtain the prediction and compute the operation. Since the model is being trained on 3 channel images, we have to stack the masked images in order to run the saved model. 

## Dataset
The dataset consists of around 21000 images for 18 different gesture classes with each gesture containing around 900 training images and 300 testing images. The images in the dataset are 50 x 50 pixels.

The link for the dataset is as follows:
https://www.kaggle.com/aryarishabh/hand-gesture-recognition-dataset

<img src="images/labels.png" width="720" >
