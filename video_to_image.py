import cv2
import numpy as np
import time
import os
import argparse
from datetime import date

parser = argparse.ArgumentParser()

parser.add_argument("--video_path",help="give a video_path")
parser.add_argument("--img_path",default="./output_images/",help="give a image path ")
parser.add_argument("--resize",default="default",help="to resize --resize=1 and entry x and y")
parser.add_argument("--x")
parser.add_argument("--y")
parser.add_argument("--fps",default=20,help="to change fps value")
parser.add_argument("--image_name",default="x",help="to change images' name")


veri = parser.parse_args()

cap = cv2.VideoCapture(str(veri.video_path))

name=""



if(veri.image_name=="x"):
  name="image"
else:
  name=str(veri.image_name)

if(veri.img_path=="./output_images/"):

  file_path = "./output_images"+"_"+str(int((time.time())))
  os.mkdir(file_path)
  veri.img_path=file_path


i=0

t=float(1/int(veri.fps))


print("video_path:",veri.video_path,"image_saving_path:",veri.img_path,"FPS:",veri.fps,"image_saving_name:",veri.image_name)


while(cap.isOpened()):
  path = str(veri.img_path)
  ret, frame = cap.read()

  if(veri.resize=="default"):
	ret, frame = cap.read()
	xx = frame.shape[1]
	yy = frame.shape[0]
  else:
	xx = int(veri.x)
	yy = int(veri.y)



  if (ret == False):
    break

  frame = cv2.resize(frame,(xx,yy), interpolation = cv2.INTER_AREA)
  cv2.imwrite(path+"/"+name+"_"+str(i)+".jpg",frame)
  time.sleep(t)
  i+=1
  cv2.imshow('frame',frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break



print(str(i)+" "+"images_saved")