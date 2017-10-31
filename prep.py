import PIL
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import os
from sklearn.preprocessing import LabelEncoder

path = os.getcwd() + '/dataset/train/'
folder_names = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
labels = []
filenames = []
for fn in folder_names:
    new_path = path + fn
    for file in os.listdir(new_path):
        if file.endswith('.jpg'):
            filenames += [new_path + '/' + file]

np.random.shuffle(filenames)

count = 0
for fn in filenames:
    count += 1
    img = PIL.Image.open(fn)
    img = img.resize((32, 32))
    img.save(os.getcwd() + '/dataset/new_train/' + str(count) + '.jpg')
    arr = fn.split('/')
    print(fn + "\t" + arr[len(arr) - 2])
    labels += [arr[len(arr) - 2]]

label = pd.DataFrame({
    'label': labels
})
label.to_csv('label.csv')
