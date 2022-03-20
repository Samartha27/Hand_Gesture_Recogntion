# Realtime Hand Gesture Recognition

## Team Members
Jnana Rama Krishna D and Samartha Ramkumar

[Video presentation](https://drive.google.com/file/d/1hJuggP78K1OOjdf7JcgZBRIgGPBR321C/view?ts=62302ea3)

## Problem statement
Hand gesture recognition is an important way of communicating among humans, also between human and a machine. This has a wide range of applications, from interacting with robots, to interacting with laptops, mobile phones and helping specially abled people to express themselves. 

Our goal in this project is to build various CNN models using transfer learning and use them to recognize various real time hand gestures. The models are trained to recognize 18 classes with each class having 900 training images. 


## Dataset
The dataset was downloaded from [Kaggle](https://www.kaggle.com/aryarishabh/hand-gesture-recognition-dataset) and it consists of around 21000 images for 18 different gesture classes with each gesture containing around 900 training images and 300 validation images. The resolution of the images in the dataset are 50 x 50 pixels, which is really low.


## Data Augmentation

Even though the training images look like a masked image, they're  3 channel images and have a size of `50x50`. The background of the hand is black colored and the hand in the foreground is of white in color. Each image is resized to `256 x 256` as the image size has to be compatible with the models that are trained on the PyTorch. The images are normalized using mean as ```[0.485, 0.456, 0.406]``` and standard deviation as ```[0.229, 0.224, 0.225]``` as they're the trained on ImageNet and the ImageNet has the exact same mean and standard deviation. We've tried to increase variation in the training data by using random resize crop and horizontal flip and used the center crop for the validation dataset to generate some randomness in the images that the model is being validated.

# Model preparation 

## Transfer Learning

We've used transfer learning for feature extraction, where we've mainly froze the weights of the network in all the layers except the final fully connected layer. The last layer is replaced with a fully connected layer based on the number of classes we're using (18 in our case) and the model is trained again so that these last layer weights are updated, which significantly can reduce the training times of the heavy classification models that are being used and could help the model to generalize well as it was being trained on many features. As each networks last layer is different we need to modify the last layer based on the inputs of the last layer of that particular model. All the models used are pretrained in pytorch using the ImageNet dataset. We've  trained all the models with 6 epochs considering the huge training time we've encoured while experimenting with models. The model weights are later saved on to run a realtime handgesture analysis. 

## Model training and experimentation 

We've used the AlexNet, MobileNetV2, ResNet18 as the backbones and have trained the models with various batch sizes 64, 128, 256. The learning rate was 0.001 and Stochastic Gradient Descent(SGD) was used as the optimizer. CrossEntropyLoss was the criterion used to find the loss in each step as we're basically solving a multi class classification problem. Each model is trained on a NVIDIA  GTX 1050 Ti  GPU.

## Analysis of the results  

As the models take huge amount of time to train, we've mainly used only "batch size" as the varying parameter in all the experiments annd analysed the training times for each experiment. Apart from that we've analysed the training times of all the networks with various batch sizes. The batch sizes used are 64, 128 and 256, for both training, validation loops.

<p align="center">
    i) Validation accuracy for AlexNet
</p>

<p align="center">
    <img src="./assets/alexnet.png" >       
</p>

<p align="center">
    ii) Validation accuracy for MobileNetV2
</p>

<p align="center">
    <img src="./assets/mobilenetv2.png"> 
 </p>  
 
<p align="center">
   iii) Validation accuracy for ResNet18
</p>

<p align="center">   
    <img src="./assets/resnet.png">
</p>

All the models converged with less number of epochs, as the dataset has little to less variation in the training images. But, among the models used, MobieNetV2 converged relatively with less number of epochs followed by ResNet and AlexNet. Among all the models AlexNet took more number of epochs to converge.

We've achieved the best accuracy of `99.5926%` among all the experiments, using a MobileNetV2 with a batch size of 128. The same model's weights are saved in ordder to run the real time hand gesture recognition.

<p align="center">   
    <img src="./assets/bestaccuracy.jpeg" width="400" >
</p>

We've also analysed the training time taken by various models, and have observed that the training time almost remained same from the batch size of 128 in all the models. The MobileNet model has taken the highest training time compared to other two models, where as the AlexNet took the least training times.

<p align="center">   
    <img src="./assets/trainingtimes.jpeg" width="400" >
</p>


# Real time hand gesture recognition
We've used the best performiing model(MobileNetV2) after our experiments and used it's weights to predict the output for each frame in a video stream. Once we capture a single from the camera, we pass that frame to segment the hand region in the frame.

<p align="center">   
    <img src="./assets/flow.jpg">
</p>

## Segmenting hand from a frame

 - Background subtraction
    
    We used running average technique so that the system looks at a scene for 30 frames and computes the running average over the current, previous frames. Once this is done, we place the hand in the designated window so that the system knows the new pixels correspond to the new object entered into the scene and is at the foreground of the scene. 
   
    To do this, we used the concept of running averages. We make our system to look over a particular scene for 30 frames. During this period, we compute the running average over the current frame and the previous frames. After figuring out the background, we bring in our hand and make the system understand that our hand is a new entry into the background, which means it becomes the foreground object. Now we use the hand to find the absolute difference between the background scene (changes over time) and the current frame (contains our hand) which results in a image which holds only the newly entered object in the scene, which is our hand.

 - Motion detection and thresholding

    We use a threshold parameter on the difference image to filter out only the hand and set all other objects into the background

 - Contour extraction

    We find the contours in the thresholded image and obtain the contour with maximum area as the hand. 

After segmenting the hand from the background, the input image looks like the following : 
<p align="center">   
    <img src="./assets/masked_input.jpeg" width="400" >
</p>


## Prediction using model
Once we get the segmented hand image, we pass the image to our MobileNetV2 model to predict the class it belongs to and the class number is displayed on the window.


# Discussion

## Problems faced

The problem we faced was that the dataset used was undersized and it has less variation among them. To use the dataset with sophisticated large models like MobileNet v2, AlexNet,etc the images had to be resized nearly 5 times to their original size which although proved good to go while training, but when it came to the implementation on the webcam the model was not able to generalize well. The accuracy obtained could have been better in the real time scenarios.
  
<p align="center"> 
  <img src="./assets/output1.jpeg" width="400" >     <img src="./assets/output2.jpeg" width="400" >
</p>
    
<p align="center">
    <img src="./assets/output3.jpeg" width="400" >       <img src="./assets/output4.jpeg" width="400" >
</p>
 

Due to the limited availability of GPU computation resources we had to run the dataset containing 21000 thousand images(although the image was undersized) and  on Google colab and kaggle notebooks and fine tuning the pre-trained models was very challenging. We spent a significant amount of time with the local CUDA crashing while training with the datasets. 


## Next steps and future work

Even though the model gave good accuracies on the vaildation set, it struggles to perform well in the real world. We would like to explore more sophisticated data augmentation methods to create a dataset with sufficient diversity for effective generalization of the images on previously unseen data. Also, we would need better ways to segment our hand in the foreground. Along with that we can work in the direction of involving optical flow in the pipeline so that the model works on not only static images, but also actions generated from gestures.


## How is our approach unique ? 
The previous work employed a simple baseline CNN model for training and obtained the accuracies of around 98%. We have changed the backbone of the network and made use of transfer learning to fine tune the model based on our needs to obtain better accuracies in the range of 99%. We have also made a lightweight CNN model while playing around with models. 
By employing sophisticated networks we were able to  obtain better accuracies. However deployment at real time proved challenging due to technical limitations.

References : 
1) [Presentation for the project](https://docs.google.com/presentation/d/13LD8DBz0BHih0IjaJgoh5blaWbaGZDlG/edit#slide=id.p1)
2) [Github Reference used](https://github.com/rishabh-arya/Gesture-controlled-opencv-calculator)
3) [Transfer learning fine tuning in pytorch](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)
4) [Transfer learning](https://cs231n.github.io/transfer-learning/)
