
#importing Keras, Library for deep learning 
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import  MaxPooling2D,Conv2D,ZeroPadding2D,ZeroPadding3D
from keras.utils import np_utils
from keras.preprocessing.image import  img_to_array


import numpy as np

# Image manipulations and arranging data
import os
from PIL import Image
from os import walk,path


from sklearn.cross_validation import train_test_split


def unpickle(file):
 '''Load byte data from file'''
 import pickle
 with open(file, 'rb') as f:
  data = pickle.load(f, encoding='latin-1')
  return data


def load_cifar10_data(data_dir):
 '''Return train_data, train_labels, test_data, test_labels
 The shape of data is 32 x 32 x3'''
 train_data = None
 train_labels = []
 for i in range(1, 6):
  data_dic = unpickle(data_dir + "/data_batch_{}".format(i))
  if i == 1:
   train_data = data_dic['data']
  else:
   train_data = np.vstack((train_data, data_dic['data']))
  train_labels += data_dic['labels']

 test_data_dic = unpickle(data_dir + "/test_batch")
 test_data = test_data_dic['data']
 test_labels = test_data_dic['labels']

 train_data = train_data.reshape((len(train_data), 3, 32, 32))
 train_data = np.rollaxis(train_data, 1, 4)
 train_labels = np.array(train_labels)

 test_data = test_data.reshape((len(test_data), 3, 32, 32))
 test_data = np.rollaxis(test_data, 1, 4)
 test_labels = np.array(test_labels)

 return train_data, train_labels, test_data, test_labels

data_dir = 'C:/Users/584061/Desktop/cifar-10-batches-py'
train_data, train_labels, test_data, test_labels = load_cifar10_data(data_dir)


batch_size=32
nb_classes=200#len(classes)
nb_epoch=20
nb_filters=32
nb_pool=2
nb_conv=3


from keras.layers.normalization import BatchNormalization

model= Sequential()

model.add(Conv2D(64, (3, 3), padding="same",input_shape=train_data.shape[1:]))
#model.add(Conv2D(64, (3, 3), padding="same",input_shape=x_train.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D((2,2), strides=(2,2)))

## first set of CONV => RELU => POOL layers
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2), strides=(2,2)))
#model.add(Dropout(0.5));
# second set of CONV => RELU => POOL layers
model.add(Conv2D(256, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D((2,2), strides=(2,2)))

# Third set of CONV => RELU => POOL layers
#model.add(Conv2D(512, (3, 3), padding="same"))
#model.add(Activation("relu"))
#model.add(MaxPooling2D((2,2), strides=(2,2)))


# first (and only) set of FC => RELU layers
model.add(Flatten())
#model.add(Dense(100))
#model.add(Activation("relu"))
#
#model.add(Dense(3000))
#model.add(Activation("relu"))
## softmax classifier
model.add(Dense(10))#nb_classes))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
model.summary()
nb_epoch=25;
batch_size=5;
Y_train_labels=np_utils.to_categorical(train_labels,10)

Y_test_labels=np_utils.to_categorical(test_labels,10)
#model.fit(x_train,Y_train,batch_size=batch_size,epochs=nb_epoch,verbose=1,validation_data=(x_val, Y_val))
model.fit(train_data,Y_train_labels,batch_size=batch_size,epochs=nb_epoch,verbose=1)
predictions = model.predict_classes(test_data,verbose=1)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

print(f1_score(test_labels , predictions, average="macro"))
print(precision_score(test_labels, predictions, average="macro"))
print(recall_score(test_labels, predictions, average="macro"))    
print(classification_report(test_labels, predictions))
