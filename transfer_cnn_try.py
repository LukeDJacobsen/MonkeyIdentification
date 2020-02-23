#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 18:15:05 2020

@author: lukejacobsen
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

from keras.preprocessing.image import ImageDataGenerator

#dl libraraies
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau

# specifically for cnn
from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

import tensorflow as tf
import random as rn
import cv2                  

from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50

###########################
#Importing photos
###########################
os.getcwd()
os.listdir()

X = []
Z = []

Xv = []
Yv = []


t0_dir='training/n0'
t1_dir='training/n1'
t2_dir='training/n2'
t3_dir='training/n3'
t4_dir='training/n4'
t5_dir='training/n5'
t6_dir='training/n6'
t7_dir='training/n7'
t8_dir='training/n8'
t9_dir='training/n9'

v0_dir='validation/n0'
v1_dir='validation/n1'
v2_dir='validation/n2'
v3_dir='validation/n3'
v4_dir='validation/n4'
v5_dir='validation/n5'
v6_dir='validation/n6'
v7_dir='validation/n7'
v8_dir='validation/n8'
v9_dir='validation/n9'

def create_train(monkey_type, directory, image_size):
    for image in os.listdir(directory):
      path = os.path.join(directory,image)
      img = cv2.imread(path)
      img = cv2.resize(img, (image_size, image_size))
      label=assign_label(img,monkey_type)
      X.append(np.array(img))
      Z.append(str(label))

create_train('mantled_howler', t0_dir, image_size = 150)
create_train('patas_monkey',t1_dir, image_size = 150)
create_train('bald_uakari',t2_dir, image_size = 150)
create_train('japanese_macaque',t3_dir, image_size = 150)
create_train('pygmy_marmoset',t4_dir, image_size = 150)
create_train('white_headed_capuchin',t5_dir, image_size = 150)
create_train('silvery_marmoset',t6_dir, image_size = 150)
create_train('common_squirrel_monkey',t7_dir, image_size = 150)
create_train('black_headed_night_monkey',t8_dir, image_size = 150)
create_train('nilgiri_langur',t9_dir, image_size = 150)
le=LabelEncoder()
Y=le.fit_transform(Z)
Y=to_categorical(Y,10)
X=np.array(X)
X=X/255

#########################
#look at a few
#########################

plt.imshow(X[4])
plt.imshow(X[305])
plt.imshow(X[500])
plt.imshow(X[501])

#########################
#train model
#########################

datagen.fit(X)


res50 = Sequential()
res50.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
res50.add(Dense(10, activation='softmax'))

# Say not to train first layer (ResNet) model. It is already trained
res50.layers[0].trainable = False

res50.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

res50.summary()

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

epochs=10
batch_size=32

 model.fit_generator(datagen.flow(X, Y, batch_size=batch_size),
                              epochs = 10, validation_data = (x_test,y_test),
                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size)

import sklearn as ski
ski.cross_val_score(my_new_model, X, Y, cv=3)