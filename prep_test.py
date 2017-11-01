import PIL
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import os
from sklearn.preprocessing import LabelEncoder

path = os.getcwd() + '/dataset/test_stg2/'
filenames = []
for file in os.listdir(path):
    if file.endswith('.jpg'):
        filenames += [path + file]

for fn in filenames:
    img = PIL.Image.open(fn)
    img = img.resize((32, 32))
    img.save(fn)
    arr = fn.split('/')
    print(fn)
