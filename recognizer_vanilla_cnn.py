import numpy as np
import pandas as pd
import os
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor,Lambda
from torchvision.io import read_image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
# %matplotlib inline
import cv2 

##### RECOGNIZER WITH CUSTOM VANILA CNN MODEL


data_dir = "./data"
model_name = "mynn"
num_classes = 18
batch_size = 128

learning_rate = 0.0001
input_size = 64

class mynn(nn.Module):
    
    def __init__(self, output_dim):
        super(mynn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels = 32, kernel_size = 3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels = 64, kernel_size = 3, stride=2)
        self.max_pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout2d(p=0.15)
            
        self.conv3 = nn.Conv2d(in_channels=64, out_channels = 64, kernel_size = 3, stride=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels = 128, kernel_size = 3, stride=1)
        self.max_pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout2d(p=0.25)
        
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)
            
        pass

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.max_pool1(x)
        x = self.dropout1(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.max_pool2(x)
        x = self.dropout2(x)

        x = torch.flatten(x,start_dim = 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = mynn(18)
model.load_state_dict(torch.load("/home/agent/Desktop/cv_samartha/saved_state_gesture_model.pt"))
model.eval()



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    model.cuda()

print("model loaded")

bg = None
global loaded_model

# Function - To find the running average over the background
def run_avg(image, accumWeight):
	global bg
	if bg is None:
		bg = image.copy().astype("float")
		return

	cv2.accumulateWeighted(image, bg, accumWeight)

# Function - To segment the region of hand in the image
def segment(image, threshold=30):
	global bg
	# find the absolute difference between background and current frame
	diff = cv2.absdiff(bg.astype("uint8"), image)
	#cv2.imshow("diff = grey - bg",diff)
	cv2.imshow("grey",image)
	# threshold the diff image so that we get the foreground
	thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
	(cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if len(cnts) == 0:
		return
	else:
		segmented = max(cnts, key=cv2.contourArea)
		return (thresholded, segmented)

	

tfms = transforms.Compose([
		#transforms.Resize(256),	
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def count(thresholded, segmented):
	
	img = cv2.resize(thresholded,(256,256))
	img = np.stack((img,)*3, axis=-1)
	img = img.astype(np.float32)
	img = tfms(img)
	img = img.permute(2, 1, 0)
	
	img = torch.unsqueeze(img, 0)
	img = img.permute(0, 3, 1, 2)
	img = img.cuda()
	outputs = model(img)
	_, preds = torch.max(outputs, 1)
	predicted_class = preds.item()
	return predicted_class


# Main function
if __name__ == "__main__":

	accumWeight = 0.5

	# get the reference to the webcam
	camera = cv2.VideoCapture(0)

	# region of interest (ROI) coordinates
	top, right, bottom, left = 10, 350, 225, 590

	# initialize num of frames
	num_frames = 0

	# calibration indicator
	calibrated = False

	# keep looping, until interrupted
	while(True):
		# get the current frame 
		(ret, frame) = camera.read()
		#print(frame.shape)		#shape of the frame (480, 640, 3)
		# resize the frame
		#frame = imutils.resize(frame, width=700)

		# flip the frame so that it is not the mirror view
		frame = cv2.flip(frame, 1)	#shape of the frame (480, 640, 3)
		print(frame.shape)
		
		clone = frame.copy()

		# get the height and width of the frame
		(height, width) = frame.shape[:2]

		# get the ROI
		roi = frame[top:bottom, right:left]

		# convert the roi to grayscale and blur it
		gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (7, 7), 0)

		# to get the background, keep looking till a threshold is reached
		# so that our weighted average model gets calibrated
		if num_frames < 30:
			run_avg(gray, accumWeight)
			if num_frames == 1:
				print ("Calibration in progress !!!")
			elif num_frames == 29:
				print ("Finished calibration!")
		else:
			# segment the hand region
			hand = segment(gray)

			# check whether hand region is segmented
			if hand is not None:

				(thresholded, segmented) = hand
				# draw the segmented region and display the frame
				cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))

				# count the number of fingers
				
				fingers = count(thresholded, segmented)
				
				print(fingers)
				cv2.putText(clone, str(fingers), (200, 45), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,255), 2)
				# show the thresholded image
				cv2.imshow("Thesholded", thresholded)

		# draw the segmented hand
		cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

		# increment the number of frames
		num_frames += 1

		# display the frame with segmented hand
		cv2.imshow("Video Feed", clone)

		# observe the keypress by the user
		keypress = cv2.waitKey(1) & 0xFF

		# if the user has pressed "q", then stop looping
		if keypress == ord("q"):
			break

# free up memory
camera.release()
cv2.destroyAllWindows()

