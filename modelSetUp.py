import tensorflow as tf
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Activation, Dropout

import pandas as pd
import numpy as np

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
labels = train.iloc[:,0].values.astype('int32')
X_train = train.iloc[:,1:].values.astype('float32')
X_test = (pd.read_csv('../input/test.csv').values).astype('float32')

y_train = np_utils.to_categorical(labels)

scale = np.max(X_train)
X_train /= scale
X_test /= scale

mean = np.std(X_train)
X_train -= mean
X_test -= mean

input_dim = X_train.shape[1]
nb_classes = y_train.shape[1]

model = Sequential()
model.add(Dense(128, input_dim=input_dim))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

print(X_train.shape)
