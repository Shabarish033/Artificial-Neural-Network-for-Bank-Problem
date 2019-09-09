#%% -*- coding: utf-8 -*-
#%% Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder #Used in data preprocessing for encoding input data
from sklearn.model_selection import train_test_split #Data Preprocessing - Splitting data set
from sklearn.preprocessing import StandardScaler #Standardizing data
import keras
from keras.models import Sequential #ANN Builders
from keras.layers import Dense #Defining Layers within the ANN
from sklearn.metrics import confusion_matrix #To compute the efficiency of predictions
#%%Data Preprocessing
#Reading Inputs
data = pd.read_csv('Churn_Modelling.csv')
X = data.iloc[:,3:13]
Y = data.iloc[:,-1]

#Encoding Columns of data
LabelEncoder_X1 = LabelEncoder()
X.iloc[:,1] = LabelEncoder_X1.fit_transform(X.iloc[:,1])
LabelEncoder_X2 = LabelEncoder()
X.iloc[:,2] = LabelEncoder_X2.fit_transform(X.iloc[:,2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] #This is done to avoid dummy variable trap

#Test and Training datasets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y, test_size=1/5, random_state=0)

# Scaling data
ss = StandardScaler()
Xtrain = ss.fit_transform(Xtrain)
Xtest = ss.fit_transform(Xtest)

#%% Creating the ANN
classifier = Sequential()
#1st Input Layer
classifier.add(Dense(units=6, activation='relu', kernel_initializer='glorot_uniform', input_dim=11))
#2nd Input Layer
classifier.add(Dense(units=6, activation='relu', kernel_initializer='glorot_uniform'))
#Output Layer
classifier.add(Dense(units=1, activation='sigmoid', kernel_initializer='glorot_uniform'))
#Compiling
classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#Fitting the Neural Network
classifier.fit(Xtrain, Ytrain, batch_size=10,epochs=100)
# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
Ypred = classifier.predict(Xtest)
Ypred = (Ypred > 0.5)

#%% Making the Confusion Matrix #Efficiency of the ANN
cm = confusion_matrix(Ytest, Ypred)
