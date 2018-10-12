# Digit Recognition using ANN
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

training_set = pd.read_csv('train.csv')
X_test = pd.read_csv('test.csv')
X_train = training_set.drop('label', 1)
y_train = training_set.iloc[:, 0]
y_train = pd.DataFrame(y_train)

onehotencoder = OneHotEncoder(categorical_features = [0])
y_train = onehotencoder.fit_transform(y_train).toarray()

X_train = np.array(X_train)
X_test = np.array(X_test)

classifier = Sequential()

# Adding the input layer and the first hidden layer
# add a hidden and an input layer to the model using the demse fn
# 'units' = hidden layer, kernel_initializer = 'uniform' is used to initialize the weights randomly
# and unifromly, activation = 'relu' is used to set the activation fn as rectifier
# input_dim = 11 is used to set the number of input neurons as the independent variables
classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu', input_dim = 784))

# Adding the second hidden layer
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))
# Adding the output layer
classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'softmax'))

sgd = SGD(lr=0.001, decay=1e-06, momentum=0.9, nesterov=True)
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'mse', metrics = ['categorical_accuracy'])
# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 30, epochs = 30)

y_pred = classifier.predict(X_test)

y_pred = np.argmax(y_pred, axis = 1)