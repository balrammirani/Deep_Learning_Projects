#V0.1 -- 
#F1_score - 0.02538435073
#Precision - 0.0265963526208
#recall - 0.0324287518038

#importing Keras, Library for deep learning 
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import  img_to_array
from keras import backend as K
K.set_image_dim_ordering('th')

import numpy as np

import os
from PIL import Image
from os import walk,path

theano.config.optimizer="None"
#Sklearn to modify the data

from sklearn.cross_validation import train_test_split
os.chdir("D:\AV_Comp\Image Classification\images");

# input image dimensions
m,n = 75

path1="input";
path2="data";

classes=os.listdir(path2)
dir_path ="D:\AV_Comp\Image Classification\images"
x=[]
labels=[]
for (dirpath, dirnames, filenames) in walk(dir_path):
    for filename in filenames: 
	# load the image, pre-process it, and store it in the data list
        if "._" not in filename:
            im=Image.open(path.join(dirpath,filename));
            im=im.convert(mode='RGB')
            imrs=im.resize((m,n))
            imrs=img_to_array(imrs)/255;
            imrs=imrs.transpose(2,0,1);
            imrs=imrs.reshape(3,m,n);
            x.append(imrs)
            label = dirpath.split(".")[-1]
            labels.append(label)
        
x=np.array(x);
labels=np.array(labels);

batch_size=32
nb_classes=200#len(classes)
nb_epoch=20
nb_filters=32
nb_pool=2
nb_conv=3

(x_train, x_val, y_train, y_val) = train_test_split(x,labels, test_size=0.25, random_state=42)

(x_train, x_test, y_train, y_test) = train_test_split(x_train,y_train, test_size=0.25, random_state=42)

#x_train, x_test, y_train, y_test= train_test_split(x,labels,test_size=0.2,random_state=4)

uniques, id_train=np.unique(y_train,return_inverse=True)
Y_train=np_utils.to_categorical(id_train,nb_classes)
uniques, id_test=np.unique(y_test,return_inverse=True)
Y_test=np_utils.to_categorical(id_test,nb_classes)

uniques, id_val=np.unique(y_val,return_inverse=True)
Y_val=np_utils.to_categorical(id_val,nb_classes)


model= Sequential()
model.add(Convolution2D(nb_filters,nb_conv,nb_conv,border_mode='same',input_shape=x_train.shape[1:]))
model.add(Activation('relu'));
model.add(Convolution2D(nb_filters,nb_conv,nb_conv));
model.add(Activation('relu'));
model.add(MaxPooling2D(pool_size=(nb_pool,nb_pool)));
model.add(Dropout(0.5));
model.add(Flatten());
model.add(Dense(128));
model.add(Dropout(0.5));
model.add(Dense(nb_classes));
model.add(Activation('softmax'));
model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

nb_epoch=12;
batch_size=5;
model.fit(x_train,Y_train,batch_size=batch_size,nb_epoch=nb_epoch,verbose=1,validation_data=(x_val, Y_val))

predictions = model.predict_classes(x_test,verbose=1)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

print(f1_score(id_test, predictions, average="macro"))
print(precision_score(id_test, predictions, average="macro"))
print(recall_score(id_test, predictions, average="macro"))    
