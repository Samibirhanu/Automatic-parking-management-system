import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import xml.etree.ElementTree as xet

data_frame = pd.read_csv('lables.csv')
data_frame.head()


filename = data_frame['filepath'][0]
def getFilename(filename):
    filename_image = xet.parse(filename).getroot().find('filename').text
    filepath_image = os.path.join('./images', filename_image)
    return filepath_image

# print(getFilename(filename))

image_path = list(data_frame['filepath'].apply(getFilename))
# print(image_path)

file_path = image_path[0]

img = cv2.imread(file_path)

# cv2.namedWindow('example', cv2.WINDOW_NORMAL)
# cv2.imshow('Example image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.rectangle(img,(856,665),(1218,751),(0,255,0),3)
# cv2.namedWindow('example', cv2.WINDOW_NORMAL)
# cv2.imshow('Example image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Data preprocessing

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array

lables = data_frame.iloc[:,1:].values
# print(lables)

# ind = 0
data = []
output = []
for ind in range(len(image_path)):
    image = image_path[ind]
    img_arr = cv2.imread(image)
    h,w,d = img_arr.shape

    # preprocessing
    load_image = load_img(image, target_size=(224,224))
    load_image_arr = img_to_array(load_image)
    norm_load_image_arr = load_image_arr/255.0 # normalization
     # normalization to lables
    xmin,xmax,ymin,ymax = lables[ind] 
    nxmin,nxmax = xmin/w,xmax/w
    nymin,nymax = ymin/h,ymax/h
    lable_norm = (nxmin,nxmax,nymin,nymax)   # normalized output
    #append
    data.append(norm_load_image_arr)
    output.append(lable_norm)
    # print(lable_norm)
    # normalization to lables
    # print(lables[0])

# print(data)
# print(output)

x = np.array(data, dtype = np.float32)
y = np.array(output, dtype = np.float32)

# print(x.shape,y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, random_state=0)
x_train.shape, x_test.shape, y_train.shape, y_test.shape

# deep learning model
from tensorflow.keras.applications import MobileNetV2, InceptionV3, InceptionResNetV2
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model
import tensorflow as tf

inceprion_resnet = InceptionResNetV2(weights = "imagenet", include_top = False, input_tensor = Input(shape = (224, 224, 3)))
inceprion_resnet.trainable = False

headmodel = inceprion_resnet.output
headmodel = Flatten()(headmodel)
headmodel = Dense(500, activation = "relu")(headmodel)
headmodel = Dense(250, activation = "relu")(headmodel)


headmodel = Dense(4, activation = 'sigmoid')(headmodel)
