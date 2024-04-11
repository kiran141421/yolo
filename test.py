import requests
from image_object_detection import yolo_object_detection
import cv2 as cv
import base64
import pandas as pd
import numpy as np

image = cv.imread("images/people.jpg")
img_np = np.array(image)

class_names = yolo_object_detection(img_np)
print(class_names)




