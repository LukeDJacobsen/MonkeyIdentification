#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 10:38:50 2020

@author: lukejacobsen
"""

# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
 
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
%matplotlib inline  
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

#preprocess.
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

# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2                  
import numpy as np  
from tqdm import tqdm
import os                   
from random import shuffle  
from zipfile import ZipFile
from PIL import Image

#TL pecific modules
from keras.applications.vgg16 import VGG16

###############################################################################
#importing data
###############################################################################
print(os.listdir('validation'))
X=[]
Z=[]

IMG_SIZE=300

t0_dir='validation/n0'
t1_dir='validation/n1'
t2_dir='validation/n2'
t3_dir='validation/n3'
t4_dir='validation/n4'
t5_dir='validation/n5'
t6_dir='validation/n6'
t7_dir='validation/n7'
t8_dir='validation/n8'
t9_dir='validation/n9'

def assign_label(img,monkey_type):
    return monkey_type

def make_train_data(monkey_type,DIR):
    for img in tqdm(os.listdir(DIR)):
        label=assign_label(img,monkey_type)
        path = os.path.join(DIR,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        
        X.append(np.array(img))
        Z.append(str(label))
        
make_train_data('mantled_howler',t0_dir)
print(len(X))
make_train_data('patas_monkey',t1_dir)
print(len(X))
make_train_data('bald_uakari',t2_dir)
print(len(X))
make_train_data('japanese_macaque',t3_dir)
print(len(X))
make_train_data('pygmy_marmoset',t4_dir)
print(len(X))
make_train_data('white_headed_capuchin',t5_dir)
print(len(X))
make_train_data('silvery_marmoset',t6_dir)
print(len(X))
make_train_data('common_squirrel_monkey',t7_dir)
print(len(X))
make_train_data('black_headed_night_monkey',t8_dir)
print(len(X))
make_train_data('nilgiri_langur',t9_dir)
print(len(X))

#######################################
#Visualize a few images 
#######################################

fig,ax=plt.subplots(3,2)
fig.set_size_inches(8,8)
for i in range(5):
    for j in range(2):
        l=rn.randint(0,len(Z))
        ax[i,j].imshow(X[l])
        ax[i,j].set_title('Monkey: '+Z[l])    
plt.tight_layout()


########################################
#Encoding
########################################
le=LabelEncoder()
Y=le.fit_transform(Z)
Y=to_categorical(Y,10)
X=np.array(X)
X=X/255

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)

np.random.seed(42)
rn.seed(42)

########################################
#Training model
########################################
base_model=VGG16(include_top=False, weights='imagenet', 
                 input_shape=(300,300,3), pooling='avg')

base_model.summary()


model=Sequential()
model.add(base_model)

model.add(Dense(256,activation='relu'))
model.add(Dense(10,activation='softmax'))

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


datagen.fit(x_train)

#I don't know what's happening in red_lr
epochs=10
batch_size=32
red_lr=ReduceLROnPlateau(monitor='val_acc', factor=0.1, epsilon=0.0001, patience=2, verbose=1)
 

model.summary()

#fit model 
base_model.trainable=False

model.compile(optimizer=Adam(lr=1e-4),loss='categorical_crossentropy',metrics=['accuracy'])

# =============================================================================
# History = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
#                               epochs = 10, validation_data = (x_test,y_test),
#                               verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size)
# 
# =============================================================================
plt.plot(History.history['acc'])
plt.plot(History.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

#####################################
#post model training visualizations
#####################################

#####################################
#Try with other Monkey pictures
#####################################

