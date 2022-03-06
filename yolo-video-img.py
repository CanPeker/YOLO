import YOLO
import argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--image_path",help="give image_path")
parser.add_argument("--video_path",help="give video_path")
parser.add_argument("--config_path",help="give config_path",default="./baslayalim/yolov4-baslayalim.cfg")
parser.add_argument("--weights_path",help="give weights_path",default="./backup/baslayalim/yolov4-baslayalim_last.weights")
parser.add_argument("--data_path",help="give data_path",default="./baslayalim/baslayalim.data")
parser.add_argument("--type",help="1:image processing 2:Video processing")

veri = parser.parse_args()


model = YOLO.yolo_model(veri.config_path,veri.weights_path,veri.data_path)


if(veri.type=="1"):

    frame_read = cv2.imread(veri.image_path)

    x = frame_read.shape[0]
    y = frame_read.shape[1]

    detected_frame = model.yolo_detection(frame_read,x,y)

    cv2.imshow("Final_Frame",detected_frame)
    cv2.waitKey(0)


if(veri.type=="2"):

    cap = cv2.VideoCapture(veri.video_path)

    while True:

        _,frame_read = cap.read()
        x = frame_read.shape[0]
        y = frame_read.shape[1]

        detected_frame = model.yolo_detection(frame_read, x, y)

        cv2.imshow("Final_Frame", detected_frame)
        cv2.waitKey(1)


else:
    print("TYPE EROR !!")

print("END")