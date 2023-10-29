from keras import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

train = pd.read_csv('iris.data', header=None)
test = pd.read_csv('bezdekIris.data', header=None)
X_train = train.iloc[:, :4]
X_test = test.iloc[:, :4]

input_strings = train.iloc[:, 4]
label_encoder = LabelEncoder()
input_labels = label_encoder.fit_transform(input_strings)

#pre-process data, helps with speed I guess :P
    #create standard deviation of 1
scale = np.max(X_train)
X_train /= scale
X_test /= scale
    #create mean of 0
mean = np.mean(X_train)
X_train -= mean
X_test -= mean

y_train = to_categorical(input_labels)

#input = [1, 4]
#output = [1, 3]

model = Sequential()
model.add(Dense(4, input_shape=(4,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100)

model.evaluate(X_test, y_test)






