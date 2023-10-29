from keras import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten
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
y_train = to_categorical(input_labels)

output_strings = test.iloc[:, 4]
output_labels = label_encoder.fit_transform(output_strings)
y_test = to_categorical(output_labels)

#pre-process data, helps with speed I guess :P
    #create standard deviation of 1
scale = np.max(X_train)
X_train /= scale
X_test /= scale
    #create mean of 0
mean = np.mean(X_train)
X_train -= mean
X_test -= mean



#input = [1, 4]
#output = [1, 3]
print(X_train.shape)

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(4,)))
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(16, activation='swish'))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=1000)

print(model.evaluate(X_test, y_test))






