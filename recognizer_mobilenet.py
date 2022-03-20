import cv2
import imutils
import numpy as np
from sklearn.metrics import pairwise
import os
import glob
import torch
import os
from torchvision import datasets, models, transforms
import torch.nn as nn

PATH = "/home/agent/Desktop/cv_samartha/mobilenetv2128.pt"

data_dir = "./test_data"

model = models.mobilenet_v2()

num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 18)
input_size = 224

model.load_state_dict(torch.load(PATH))
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









