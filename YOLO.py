from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet


class yolo_model:

    def __init__(self, cfg, wghts, data):
        self.configPath = cfg
        self.weightPath = wghts
        self.metaPath = data
        self.netMain = None
        self.metaMain = None
        self.altNames = None
        self.model()

    def model(self):
        if not os.path.exists(self.configPath):
            raise ValueError("Invalid config path `" +
                             os.path.abspath(self.configPath) + "`")
        if not os.path.exists(self.weightPath):
            raise ValueError("Invalid weight path `" +
                             os.path.abspath(self.weightPath) + "`")
        if not os.path.exists(self.metaPath):
            raise ValueError("Invalid data file path `" +
                             os.path.abspath(self.metaPath) + "`")
        if self.netMain is None:
            self.netMain = darknet.load_net_custom(self.configPath.encode(
                "ascii"), self.weightPath.encode("ascii"), 0, 1)  # batch size = 1
        if self.metaMain is None:
            self.metaMain = darknet.load_meta(self.metaPath.encode("ascii"))
        if self.altNames is None:
            try:
                with open(self.metaPath) as metaFH:
                    metaContents = metaFH.read()
                    import re
                    match = re.search("names *= *(.*)$", metaContents,
                                      re.IGNORECASE | re.MULTILINE)
                    if match:
                        result = match.group(1)
                    else:
                        result = None
                    try:
                        if os.path.exists(result):
                            with open(result) as namesFH:
                                namesList = namesFH.read().strip().split("\n")
                                self.altNames = [x.strip() for x in namesList]
                    except TypeError:
                        pass
            except Exception:
                pass

    def convertBack(self, x, y, w, h):
        xmin = int(round(x - (w / 2)))
        xmax = int(round(x + (w / 2)))
        ymin = int(round(y - (h / 2)))
        ymax = int(round(y + (h / 2)))
        return xmin, ymin, xmax, ymax

    def cvDrawBoxes(self, detections, img):
        for detection in detections:
            x, y, w, h = detection[2][0], \
                         detection[2][1], \
                         detection[2][2], \
                         detection[2][3]
            xmin, ymin, xmax, ymax = self.convertBack(
                float(x), float(y), float(w), float(h))
            pt1 = (xmin, ymin)
            pt2 = (xmax, ymax)
            cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
            cv2.putText(img,
                        detection[0].decode() +
                        " [" + str(round(detection[1] * 100, 2)) + "]",
                        (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        [0, 255, 0], 2)
        return img

    def yolo_detection(self, frame_read, frame_height, frame_width):

        darknet_image = darknet.make_image(frame_width, frame_height, 3)
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (frame_width,frame_height),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

        detections = darknet.detect_image(self.netMain, self.metaMain, darknet_image, thresh=0.35)
        image = self.cvDrawBoxes(detections, frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image
