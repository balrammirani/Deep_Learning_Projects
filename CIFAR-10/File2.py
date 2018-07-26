# Improved the model and reached 22.79% CNN error for validation set
# This model promises precision of 0.78
#importing Keras, Library for deep learning 
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import  MaxPooling2D,Conv2D,ZeroPadding2D,ZeroPadding3D
from keras.utils import np_utils
from keras.preprocessing.image import  img_to_array
import numpy as np

# Image manipulations and arranging data
from PIL import Image
from os import walk,path
from keras.callbacks import History

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
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




data_dir = '/../Desktop/cifar-10-batches-py'
train_data, train_labels, test_data, test_labels = load_cifar10_data(data_dir)

train_data=train_data - np.mean(train_data) / train_data.std() #normalizing the data
test_data=test_data - np.mean(test_data) / test_data.std()

(x_train, x_val, y_train, y_val) = train_test_split(train_data,train_labels, test_size=0.30, random_state=42)

batch_size=32
nb_classes=len(set(train_labels))
nb_epoch=20
nb_filters=32
nb_pool=2
nb_conv=3


from keras.layers.normalization import BatchNormalization

model= Sequential()

## first set of CONV => RELU => POOL layers
model.add(Conv2D(64, (3, 3),padding="same",activation = "relu",input_shape=train_data.shape[1:]))
model.add(Conv2D(64, (3, 3), padding="same",activation = "relu"))
model.add(BatchNormalization())
#model.add(Activation("relu"))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.3))

## second set of CONV => RELU => POOL layers
model.add(Conv2D(128, (3, 3), padding="same",activation = "relu"))
model.add(Conv2D(128, (3, 3), padding="same",activation = "relu"))
#model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2), strides=(2,2)))

### third set of CONV => RELU => POOL layers
model.add(Conv2D(256, (3, 3), padding="same",activation = "relu"))
model.add(Conv2D(256, (3, 3), padding="same",activation = "relu"))
##model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2), strides=(2,2)))

## fourth set of CONV => RELU => POOL layers
model.add(Conv2D(512, (3, 3), padding="same",activation = "relu"))
model.add(Conv2D(512, (3, 3), padding="same",activation = "relu"))
##model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2), strides=(2,2)))


model.add(Flatten())
model.add(Dense(nb_classes,activation = "softmax"))
#model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
model.summary()
nb_epoch=50;
batch_size=5;
Y_train_labels=np_utils.to_categorical(y_train,10)
Y_val_labels=np_utils.to_categorical(y_val,10)

Y_test_labels=np_utils.to_categorical(test_labels,10)
#model.fit(x_train,Y_train,batch_size=batch_size,epochs=nb_epoch,verbose=1,validation_data=(x_val, Y_val))
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=4)
seed=123
history = History()
result=model.fit(x_train,Y_train_labels,batch_size=batch_size,epochs=25,verbose=1,validation_data=(x_val, Y_val_labels),callbacks=[early_stopping,history])
predictions = model.predict_classes(test_data,verbose=1)

# Final evaluation of the model
scores =model.evaluate(x_val, Y_val_labels, verbose=0)
print("CNN error: % .2f%%" % (100-scores[1]*100))

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

print(f1_score(test_labels , predictions, average="macro"))
print(precision_score(test_labels, predictions, average="macro"))
print(recall_score(test_labels, predictions, average="macro"))    
print(classification_report(test_labels, predictions))
# model.save('cifar_0_2.h5')
# from keras.models import load_model
# model = load_model("cifar_0_1.h5")

import matplotlib.pyplot as plt
#Plot the Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)

#Plot the Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
